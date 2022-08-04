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
#include "llvm/Transforms/Utils.h"
#include "andersen/include/AndersenAA.h"
//#include "llvm/Analysis/AliasAnalysis.h"

#include <cassert>

using namespace llvm;

#define PMROBUST_DEBUG
#define DEBUG_TYPE "PMROBUST_DEBUG"
#include <llvm/IR/DebugLoc.h>

/**
 * Object with 3 fields: vector -> [object status, field 1, field 2, field 3]
 * TODO: may need to track the size of fields
 * TODO: need to change data structure?
 **/
typedef DenseMap<Value *, std::vector<struct VarState>> state_t;

enum class PMState {
	UNFLUSHED = 0x1,
	CLWB = 0x11,
	FLUSHED = 0x111
};

struct VarState {
	PMState s = PMState::FLUSHED;
	bool escaped = false;
};

void printVarState(VarState &state) {
	errs() << "<";
	if (state.s == PMState::UNFLUSHED)
		errs() << "Unflushed,";
	else if (state.s == PMState::CLWB)
		errs() << "CLWB,";
	else
		errs() << "Flushed,";

	if (state.escaped) errs() << "escaped>";
	else errs() << "captured>";
}

namespace {
	struct PMRobustness : public FunctionPass {
		PMRobustness() : FunctionPass(ID) {}
		StringRef getPassName() const override;
		bool runOnFunction(Function &F) override;
		void getAnalysisUsage(AnalysisUsage &AU) const override;
		void analyze(Function &F);

		static char ID;
		//AliasAnalysis *AA;
		AndersenAAResult *AA;

	private:
		void copyState(state_t * src, state_t * dst);
		void copyMergedState(SmallPtrSetImpl<BasicBlock *> * src_list, state_t * dst);
		bool update(state_t * map, Instruction * I);

		bool isPMAddr(Value * addr) { return true; }
		bool mayInHeap(Value * addr);
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
				state_t * state = States[&*it];

				if (state == NULL) {
					state = new state_t();
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
		for (state_t::iterator it = s->begin(); it != s->end(); it++) {
			state_t::iterator item = map.find(it->first);
			if (item != map.end()) {
				// (Loc, vector<VarState>) pair found
				std::vector<struct VarState> &var_state_listA = item->second;
				std::vector<struct VarState> &var_state_listB = it->second;
				if (var_state_listA.size() < var_state_listB.size()) {
					unsigned old_size = var_state_listA.size();
					var_state_listA.resize(var_state_listB.size());
					// Merge states already in listA
					for (unsigned i = 0; i < old_size; i++) {
						int tmp = static_cast<int>(var_state_listA[i].s) & static_cast<int>(var_state_listB[i].s);
						var_state_listA[i].s = static_cast<PMState>(tmp);
						var_state_listA[i].escaped |= var_state_listB[i].escaped;
					}

					// Copy states from listB to listA for the remaining indices
					for (unsigned i = old_size; i < var_state_listB.size(); i++) {
						var_state_listA[i] = var_state_listB[i];
					}
				} else {
					// Merge states in listB to states in listA
					for (unsigned i = 0; i < var_state_listB.size(); i++) {
						int tmp = static_cast<int>(var_state_listA[i].s) & static_cast<int>(var_state_listB[i].s);
						var_state_listA[i].s = static_cast<PMState>(tmp);
						var_state_listA[i].escaped |= var_state_listB[i].escaped;
					}
				}
			} else {
				map[it->first] = it->second;
			}
		}
	}

	copyState(&map, dst);
}

bool PMRobustness::update(state_t * map, Instruction * I) {
	bool ret = true;

	/* Rule 1: x.f = v => x.f becomes dirty */
	if (StoreInst * SI = dyn_cast<StoreInst>(I)) {
		Value * Addr = SI->getPointerOperand();

		// TODO: Address check to be implemented
		if (!isPMAddr(Addr))
			return false;

		//errs() << "Addr: " << *Addr << "\n";
		//errs() << "SI" << *SI << "\n";
		if (GetElementPtrInst * GEP = dyn_cast<GetElementPtrInst>(Addr)) {
			Value * BaseAddr = GEP->getPointerOperand();
			std::vector<struct VarState> &var_states = (*map)[BaseAddr];

			unsigned LastOperandIdx = GEP->getNumOperands() - 1;
			Value *LastOperand = GEP->getOperand(LastOperandIdx);
			if (ConstantInt *offset = dyn_cast<ConstantInt>(LastOperand)) {
#ifdef PMROBUST_DEBUG
				if (var_states.size() == 0) {
					value_list.push_back(BaseAddr);
				}
#endif
				unsigned index = offset->getZExtValue() + 1;	// index = GEP offset + 1
				if (var_states.size() < index + 1)
					var_states.resize(index + 1);

				var_states[index].s = PMState::UNFLUSHED;
				var_states[0].s = PMState::UNFLUSHED;
/*
				errs() << "SI" << *SI << "\n";
				errs() << "store Addr -> " << *Addr << "\t";
				errs() << "GEP Base Addr -> " << *BaseAddr << "\n";
				errs() << "var size -> " << var_states.size() << "\n";
*/
			} else {
				assert("Non constant offset\n");
			}
		} else if (isa<AllocaInst>(Addr)){
			assert(false && "Non-promotable AllocaInst Encountered; fix it here!\n");
		} else {
			std::vector<struct VarState> &var_states = (*map)[Addr];
			if (var_states.size() == 0) {
				struct VarState state;
				state.s = PMState::UNFLUSHED;
				var_states.push_back(state);

#ifdef PMROBUST_DEBUG
				value_list.push_back(Addr);
#endif
			} else if (var_states.size() == 1) {
				var_states[0].s = PMState::UNFLUSHED;
			} else
				assert(false && "Store to an object with several fields\n");
		}

		Value * Val = SI->getValueOperand();
		/* Rule 2: *x = p (where x is a heap address) => all fields of p escapes */
		if (Val->getType()->isPointerTy()) {
			if (mayInHeap(Addr)) {
				std::vector<struct VarState> &val_var_states = (*map)[Val];
				for (unsigned i = 0; i < val_var_states.size(); i++) {
					val_var_states[i].escaped = true;
				}
			}
		}
	} else {
		ret = false;
	}

	if (ret) {
		//errs() << "After " << *I << "\n";
		//printMap(map);
	}

	return ret;
}

void PMRobustness::getAnalysisUsage(AnalysisUsage &AU) const {
	AU.setPreservesAll();
	//AU.addRequired<AAResultsWrapperPass>();
	AU.addRequired<AndersenAAWrapperPass>();
}

/** Simple may-analysis for checking if an address is in the heap
 *  TODO: may need more sophisticated checks
 **/
bool PMRobustness::mayInHeap(Value * addr) {
	if (GetElementPtrInst * GEP = dyn_cast<GetElementPtrInst>(addr)) {
		Value * BaseAddr = GEP->getPointerOperand();

		for (auto &u : BaseAddr->uses()) {
			if (isa<AllocaInst>(u)) {
				return false;
			}
		}
	} else {
		for (auto &u : addr->uses()) {
			if (isa<AllocaInst>(u)) {
				return false;
			}
		}
	}

	// Address may be in the heap. We don't know for sure.
	return true;
}

void PMRobustness::printMap(state_t * map) {
	for (unsigned i = 0; i < value_list.size(); i++) {
		Value * val = value_list[i];
		std::vector<struct VarState> &v = (*map)[val];
		//errs() << "var => " << val << "\t";
		//errs() << "var size => " << v->size() << "\n";

		if (v.size() == 1) {
			errs() << *val << ": ";
			printVarState( v[0] );
			errs() << "\n";
		} else {
			for (unsigned j = 0; j < v.size(); j++) {
				errs() << *val << ", " << j << ": ";
				printVarState( v[j] );
				errs() << "\n";
			}
		}
	}
	errs() << "\n----\n";
}

void PMRobustness::test() {
	std::vector<struct VarState> tmp;
	tmp.resize(5);
	tmp[3].s = PMState::UNFLUSHED;
	//printVarState(tmp[3]);

	std::vector<struct VarState> tmp2;
	tmp2 = tmp;
	printVarState(tmp2[3]);
	errs() << "\n\n";
	/*
	DenseMap<int *, int> map;
	int x;
	int y;
	map[&y] = 2;
	if (map.find(&x) == map.end()) {
		errs() << "x not found\n";
	} else {
		errs() << "x found\n";	
	}

	if (map.find(&y) == map.end()) {
		errs() << "y not found\n";
	} else {
		errs() << "y found\n";	
	}*/
}

char PMRobustness::ID = 0;
static RegisterPass<PMRobustness> X("pmrobust", "Persistent Memory Robustness Analysis Pass");

// Automatically enable the pass.
static void registerPMRobustness(const PassManagerBuilder &,
							legacy::PassManagerBase &PM) {
	PM.add(createPromoteMemoryToRegisterPass());
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
