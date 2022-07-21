//===-- PMRobustness.cpp - xxx -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a modified version of ThreadSanitizer.cpp, a part of a race detector.
//
// The tool is under development, for the details about previous versions see
// http://code.google.com/p/data-race-test
//
// The instrumentation phase is quite simple:
//   - Insert calls to run-time library before every memory access.
//      - Optimizations may apply to avoid instrumenting some of the accesses.
//   - Insert calls at function entry/exit.
// The rest is handled by the run-time library.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/CFG.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "andersen/include/AndersenAA.h"
//#include "llvm/Analysis/AliasAnalysis.h"

#include <cassert>

using namespace llvm;

#define CDS_DEBUG
#define DEBUG_TYPE "CDS"
#include <llvm/IR/DebugLoc.h>

// TODO: need to change data structure?
typedef MapVector<Value *, struct VarState> state_t;

enum class PMState {
	UNKNOWN,
	FLUSHED,
	UNFLUSHED,
	CLWB
};

struct VarState {
	PMState s = PMState::FLUSHED;
	bool escaped = false;
};

void printVarState(VarState * state) {
	errs() << "<";
	if (state->s == PMState::UNKNOWN)
		errs() << "Unknown,";
	else if (state->s == PMState::FLUSHED)
		errs() << "Flushed,";
	else if (state->s == PMState::UNFLUSHED)
		errs() << "Unflushed,";
	else
		errs() << "CLWB,";

	if (state->escaped) errs() << "escaped>";
	else errs() << "captured>";
}

namespace {
	struct PMRobustness : public FunctionPass {
		PMRobustness() : FunctionPass(ID) {}
		StringRef getPassName() const override;
		bool runOnFunction(Function &F) override;
		void getAnalysisUsage(AnalysisUsage &AU) const override;

		void analyze(Function &F);
		void copyState(state_t * src, state_t * dst);
		void copyMergedState(SmallPtrSetImpl<BasicBlock *> * src_list, state_t * dst);
		bool update(state_t * map, Instruction * I);

		static char ID;
		//AliasAnalysis *AA;
		AndersenAAResult *AA;

	private:
		void printMap(state_t * map);
		void test();

		std::vector<Value *> value_list;
		DenseMap<const Instruction *, state_t * > States;
	};
}

StringRef PMRobustness::getPassName() const {
	return "PMRobustness";
}

bool PMRobustness::runOnFunction(Function &F) {
	//AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
	AA = &getAnalysis<AndersenAAWrapperPass>().getResult();

	analyze(F);

	return true;
}

void PMRobustness::analyze(Function &F) {
	if (F.getName() == "main") {
		//F.dump();
		//errs() << "\n--------\n\n";

		std::list<BasicBlock *> WorkList;

		// LLVM allows duplicate predecessors: https://stackoverflow.com/questions/65157239/llvmpredecessors-could-return-duplicate-basic-block-pointers
		DenseMap<const BasicBlock *, SmallPtrSet<BasicBlock *, 8> *> block_predecessors;
		DenseMap<const BasicBlock *, SmallPtrSet<BasicBlock *, 8> *> block_successors;

		WorkList.push_back(&F.getEntryBlock());
		while (!WorkList.empty()) {
			BasicBlock *block = WorkList.front();
			WorkList.pop_front();

			bool changed = false;
			BasicBlock::iterator prev = block->begin();
			for (BasicBlock::iterator it = block->begin(); it != block->end();) {
				MapVector<Value *, struct VarState> * state = States[&*it];

				if (state == NULL) {
					state = new MapVector<Value *, struct VarState>();
					States[&*it] = state;
				}

				// Build state from predecessors' states
				if (it == block->begin()) {
					// First instruction; take the union of predecessors' states
					if (BasicBlock *pred = block->getUniquePredecessor()) {
						// Unique predecessor; copy the state of last instruction in the pred block
						Instruction * last = pred->getTerminator();
						copyState(States[last], state);
					} else {
						// Multiple predecessors
						SmallPtrSet<BasicBlock *, 8> *pred_list = block_predecessors[block];
						if (pred_list == NULL) {
							pred_list = new SmallPtrSet<BasicBlock *, 8>();
							block_predecessors[block] = pred_list;
							for (BasicBlock *pred : predecessors(block)) {
								pred_list->insert(pred);
							}
						}

						copyMergedState(pred_list, state);

						/*
						errs() << "block: " << *block << "\n";
						for (BasicBlock *pred : *pred_list) {
							errs() << "pred\t";
							errs() << *pred << "\n";
						}
						errs() << "----\n\n";
						*/
					}
				} else {
					// Copy the previous instruction's state
					copyState(States[&*prev], state);
				}

				if (update(state, &*it))
					changed = true;

				prev = it;
				it++;
			}

			if (changed) {
				SmallPtrSet<BasicBlock *, 8> *succ_list = block_successors[block];
				if (succ_list == NULL) {
					succ_list = new SmallPtrSet<BasicBlock *, 8>();
					block_successors[block] = succ_list;
					for (BasicBlock *pred : successors(block)) {
						succ_list->insert(pred);
					}
				}

				for (BasicBlock *succ : *succ_list) {
					WorkList.push_back(succ);
				}
			}
		}
	}
}

void PMRobustness::copyState(state_t * src, state_t * dst) {
	for (state_t::iterator it = src->begin(); it != src->end(); it++)
		(*dst)[it->first] = it->second;
}

void PMRobustness::copyMergedState(SmallPtrSetImpl<BasicBlock *> * src_list, state_t * dst) {
	state_t map;
	for (BasicBlock *pred : *src_list) {
		state_t *s = States[pred->getTerminator()];
		for (state_t::iterator sit = s->begin(); sit != s->end(); sit++) {
			state_t::iterator ValueA = map.find(sit->first);
			if (ValueA != map.end()) {
				// key-value pair found
				// TODO: Using & can be more efficient; FLUSHED: 0x10; CLWB: 0x1; UNFLUSHED: 0x0;
				if (ValueA->second.s == PMState::UNFLUSHED || sit->second.s == PMState::UNFLUSHED) {
					map[sit->first].s = PMState::UNFLUSHED;
				} else if (ValueA->second.s == PMState::CLWB || sit->second.s == PMState::CLWB) {
					map[sit->first].s = PMState::CLWB;
				} else {
					map[sit->first].s = PMState::FLUSHED;
				}
				
				map[sit->first].escaped = ValueA->second.escaped || sit->second.escaped;
			} else {
				map[sit->first] = sit->second;
			}
		}
	}

	copyState(&map, dst);
}

bool PMRobustness::update(state_t * map, Instruction * I) {
	bool ret = true;
	if (AllocaInst *AI = dyn_cast<AllocaInst>(I)) {
		struct VarState state;
		Value * Addr = dyn_cast<Value>(AI);
		(*map)[Addr] = state;

		// Debug
		value_list.push_back(Addr);
		//errs() << *AI << "\n";
	} else if (StoreInst * SI = dyn_cast<StoreInst>(I)) {
		Value * Addr = SI->getPointerOperand();
		// errs() << "Addr: " << *Addr << "is alloca: " << "\n";
		(*map)[Addr].s = PMState::UNFLUSHED;
	} else {
		ret = false;
	}

	if (ret) {
		errs() << "After " << *I << "\n";
		printMap(map);
	}

	return ret;
}

void PMRobustness::getAnalysisUsage(AnalysisUsage &AU) const {
	AU.setPreservesAll();
	//AU.addRequired<AAResultsWrapperPass>();
	AU.addRequired<AndersenAAWrapperPass>();
}

void PMRobustness::printMap(state_t * map) {
	for (unsigned i = 0; i < value_list.size(); i++) {
		struct VarState *v = &(*map)[value_list[i]];
		errs() << *value_list[i] << ": ";
		printVarState(v);
		errs() << "\n";
	}
	errs() << "\n----\n";
}

void PMRobustness::test() {
	MapVector<int *, int> map;
	int x;
	int y;
	map[&y] = 2;
	//errs() << "map[x]: " << map[&x] << "\n";
	if (map.find(&x) == map.end()) {
		errs() << "x not found\n";
	} else {
		errs() << "x found\n";	
	}

	//errs() << "map[y]: " << map[&y] << "\n";
	if (map.find(&y) == map.end()) {
		errs() << "y not found\n";
	} else {
		errs() << "y found\n";	
	}

}

char PMRobustness::ID = 0;
static RegisterPass<PMRobustness> X("hello", "Persistent Memory Robustness Analysis Pass");

// Automatically enable the pass.
static void registerPMRobustness(const PassManagerBuilder &,
							legacy::PassManagerBase &PM) {
	PM.add(new PMRobustness());
}
/* Enable the pass when opt level is greater than 0 */
static RegisterStandardPasses 
	RegisterMyPass1(PassManagerBuilder::EP_OptimizerLast,
registerPMRobustness);

/* Enable the pass when opt level is 0 */
static RegisterStandardPasses 
	RegisterMyPass2(PassManagerBuilder::EP_EnabledOnOptLevel0,
registerPMRobustness);
