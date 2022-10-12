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

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Attributes.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils.h"
#include "PMRobustness.h"
//#include "andersen/include/AndersenAA.h"
//#include "llvm/Analysis/AliasAnalysis.h"

#define PMROBUST_DEBUG
#define DEBUG_TYPE "PMROBUST_DEBUG"
#include <llvm/IR/DebugLoc.h>

namespace {
	struct PMRobustness : public FunctionPass {
		PMRobustness() : FunctionPass(ID) {}
		StringRef getPassName() const override;
		bool doInitialization(Module &M) override;
		bool runOnFunction(Function &F) override;
		void getAnalysisUsage(AnalysisUsage &AU) const override;
		void analyze(Function &F);

		static char ID;
		//AliasAnalysis *AA;
		//AndersenAAResult *AA;

	private:
		void copyState(state_t * src, state_t * dst);
		void copyMergedState(SmallPtrSetImpl<BasicBlock *> * src_list, state_t * dst);
		bool update(state_t * map, Instruction * I);
		bool processAtomic(state_t * map, Instruction * I);
		bool processMemIntrinsic(state_t * map, Instruction * I);
		bool processLoad(state_t * map, Instruction * I);
		bool processStore(state_t * map, Instruction * I);
		//bool processFlush(state_t * map, Instruction * I, NVMOP op);
		bool processFlushWrapperFunction(state_t * map, Instruction * I);

		// TODO: Address check to be implemented
		bool skipFunction(Function &F);
		bool isPMAddr(const Value * Addr) { return true; }
		bool mayInHeap(const Value * Addr);

		void decomposeAddress(DecomposedGEP &DecompGEP, Value *Addr, const DataLayout &DL);
		unsigned getMemoryAccessSize(Value *Addr, const DataLayout &DL);
		unsigned getFieldSize(Value *Addr, const DataLayout &DL);
		NVMOP whichNVMoperation(Instruction *I);
		NVMOP whichNVMoperation(StringRef flushtype);
		NVMOP analyzeFlushType(Function &F);
		bool isFlushWrapperFunction(Instruction *I);

		const Value * GetLinearExpression(
		    const Value *V, APInt &Scale, APInt &Offset, unsigned &ZExtBits,
		    unsigned &SExtBits, const DataLayout &DL, unsigned Depth,
		    AssumptionCache *AC, DominatorTree *DT, bool &NSW, bool &NUW
		);

		bool DecomposeGEPExpression(const Value *V, DecomposedGEP &Decomposed,
			const DataLayout &DL, AssumptionCache *AC, DominatorTree *DT
		);

		void printMap(state_t * map);
		void test();

		std::vector<const Value *> value_list;
		DenseMap<const Instruction *, state_t *> States;

		std::vector<BasicBlock *> unprocessed_blocks;
		std::list<BasicBlock *> WorkList;

		unsigned MaxLookupSearchDepth = 100;
	};
}

StringRef PMRobustness::getPassName() const {
	return "PMRobustness";
}

bool PMRobustness::doInitialization (Module &M) {
	GlobalVariable *global_annos = M.getNamedGlobal("llvm.global.annotations");
	if (global_annos) {
		ConstantArray *a = cast<ConstantArray>(global_annos->getOperand(0));
		for (unsigned i=0; i < a->getNumOperands(); i++) {
			ConstantStruct *e = cast<ConstantStruct>(a->getOperand(i));
			if (Function *fn = dyn_cast<Function>(e->getOperand(0)->getOperand(0))) {
				StringRef anno = cast<ConstantDataArray>(cast<GlobalVariable>(e->getOperand(1)->getOperand(0))->getOperand(0))->getAsCString();
				std::pair<StringRef, StringRef> Split = anno.split(":");
				fn->addFnAttr(Split.first, Split.second);
			}
		}
	}

	return true;
}

bool PMRobustness::runOnFunction(Function &F) {
	if (skipFunction(F))
		return false;

	//AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
	//AA = &getAnalysis<AndersenAAWrapperPass>().getResult();

	analyze(F);

	return true;
}

void PMRobustness::analyze(Function &F) {
	if (true) {
		//errs() << "\n------\n" << F.getName() << "\n";
		//F.dump();

		// LLVM allows duplicate predecessors: https://stackoverflow.com/questions/65157239/llvmpredecessors-could-return-duplicate-basic-block-pointers
		DenseMap<const BasicBlock *, SmallPtrSet<BasicBlock *, 8> *> block_predecessors;
		DenseMap<const BasicBlock *, SmallPtrSet<BasicBlock *, 8> *> block_successors;

		WorkList.push_back(&F.getEntryBlock());
		while (!WorkList.empty()) {
			BasicBlock *block = WorkList.front();
			WorkList.pop_front();

			//errs() << "processing block: "<< block << "\n";
			bool changed = false;
			BasicBlock::iterator prev = block->begin();
			for (BasicBlock::iterator it = block->begin(); it != block->end();) {
				state_t * state = States[&*it];

				if (state == NULL) {
					state = new state_t();
					States[&*it] = state;
					changed |= true;
				}

				// Build state from predecessors' states
				if (it == block->begin()) {
					// First instruction; take the union of predecessors' states
					if (BasicBlock *pred = block->getUniquePredecessor()) {
						// Unique predecessor; copy the state of last instruction in the pred block
						state_t * prev_s = States[pred->getTerminator()];
						if (prev_s == NULL)
							WorkList.push_back(pred);
						else
							copyState(prev_s, state);
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
					changed |= true;

				prev = it;
				it++;
			}

			if (changed) {
				SmallPtrSet<BasicBlock *, 8> *succ_list = block_successors[block];
				if (succ_list == NULL) {
					succ_list = new SmallPtrSet<BasicBlock *, 8>();
					block_successors[block] = succ_list;
					for (BasicBlock *succ : successors(block)) {
						succ_list->insert(succ);
					}
				}

				for (BasicBlock *succ : *succ_list) {
					WorkList.push_back(succ);
				}
			}
		}
	}
}

// DenseMap<Value *, ob_state_t *> state_t;
void PMRobustness::copyState(state_t * src, state_t * dst) {
	for (state_t::iterator it = src->begin(); it != src->end(); it++) {
		ob_state_t *object_state = (*dst)[it->first];
		if (object_state == NULL) {
			object_state = new ob_state_t(it->second);
			(*dst)[it->first] = object_state;
		} else {
//			if (object_state->size != it->second->size) {
//				errs() << "dst size: " << object_state->size << "\t";
//				errs() << "src size: " << it->second->size << "\n";
//			}
			object_state->copyFrom(it->second);
		}
	}
}

void PMRobustness::copyMergedState(SmallPtrSetImpl<BasicBlock *> * src_list, state_t * dst) {
	for (state_t::iterator it = dst->begin(); it != dst->end(); it++)
		(*dst)[it->first]->size = 0;

	for (BasicBlock *pred : *src_list) {
		state_t *s = States[pred->getTerminator()];
		if (s == NULL) {
			WorkList.push_back(pred);
			continue;
		}

		//DenseMap<Value *, ob_state_t *> state_t;
		for (state_t::iterator it = s->begin(); it != s->end(); it++) {
			state_t::iterator item = dst->find(it->first);
			if (item != dst->end()) {
				// (Loc, vector<VarState>) pair found
				ob_state_t *A = item->second;
				ob_state_t *B = it->second;

				if (A->getSize() == 0) {
					A->setSize(B->getSize());
					A->copyFrom(B);
				} else {
					A->mergeFrom(B);
				}
			} else {
				(*dst)[it->first] = new ob_state_t(it->second);
				//map[it->first] = it->second;
			}
		}
	}
}

bool PMRobustness::update(state_t * map, Instruction * I) {
	bool updated = false;

	if (I->isAtomic()) {
		updated = processAtomic(map, I);
	} else if (isa<LoadInst>(I)) {
		updated = processLoad(map, I);
	} else if (isa<StoreInst>(I)) {
		updated = processStore(map, I);
	} else if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
		if (isa<MemIntrinsic>(I)) {
			updated = processMemIntrinsic(map, I);
		} else if (isFlushWrapperFunction(I)) {
			updated |= processFlushWrapperFunction(map, I);
		} else {
			NVMOP op = whichNVMoperation(I);
			if (op == NVM_FENCE) {
				// TODO: fence operations
			} else if (op == NVM_CLWB || op == NVM_CLFLUSH) {
				// updated |= processFlush(map, I, op);
			}
		}
	}

	if (updated) {
		//errs() << "After " << *I << "\n";
		//printMap(map);
	}

	return updated;
}

bool PMRobustness::processAtomic(state_t * map, Instruction * I) {
	bool updated = false;
	//const DataLayout &DL = I->getModule()->getDataLayout();

	if (isa<StoreInst>(I)) {
		updated |= processStore(map, I);
	} else if (isa<LoadInst>(I)) {
		updated |= processLoad(map, I);
	} else if (AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(I)) {
		// Treat atomic_exchange as load + store
		// Treat other RMWs as store
		if (RMWI->getOperation() == AtomicRMWInst::Xchg)
			updated |= processLoad(map, I);

		updated |= processStore(map, I);
		//errs() << "Atomic RMW processed\n";
	} else if (AtomicCmpXchgInst *CASI = dyn_cast<AtomicCmpXchgInst>(I)) {
		errs() << "CASI not implemented yet\n";
	} else if (isa<FenceInst>(I)) {
		// Ignore for now
		//errs() << "FenseInst not implemented yet\n";
		//IRBuilder<> IRB(I);
		//getPosition(I, IRB, true);
	}

	return updated;
}

bool PMRobustness::processMemIntrinsic(state_t * map, Instruction * I) {
	bool updated = false;

    if (MemSetInst *M = dyn_cast<MemSetInst>(I)) {
		// TODO
		//errs() << "memset addr: " << M->getArgOperand(0) << "\t";
		//errs() << "const bit: " << *M->getArgOperand(1) << "\t";
		//errs() << "size: " << *M->getArgOperand(2) << "\n\n";
    } else if (MemTransferInst *M = dyn_cast<MemTransferInst>(I)) {
		// TODO
    }
	return updated;
}

bool PMRobustness::processStore(state_t * map, Instruction * I) {
	bool updated = false;
	IRBuilder<> IRB(I);
	const DataLayout &DL = I->getModule()->getDataLayout();
	Value *Addr = NULL;
	Value *Val = NULL;

	if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
		Addr = SI->getPointerOperand();
		Val = SI->getValueOperand();
	} else if (AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(I)) {
		Addr = RMWI->getPointerOperand();

		// Rule 2.1 only makes sense for atomic_exchange
		if (RMWI->getOperation() == AtomicRMWInst::Xchg)
			Val = RMWI->getValOperand();
	} else {
		return false;
	}

	// Rule 1: x.f = v => x.f becomes dirty
	if (!isPMAddr(Addr))
		return false;

	DecomposedGEP DecompGEP;
	decomposeAddress(DecompGEP, Addr, DL);
	unsigned offset = DecompGEP.getOffsets();

	if (DecompGEP.isArray) {
		//TODO: implement robustness checks for array
		//errs() << I->getFunction()->getName() << "\n";
		//errs() << "Array encountered\t";
		//errs() << "Addr: " << *Addr << "\n";
		//errs() << *I << "\n";
		//getPosition(I, IRB, true);
		//printDecomposedGEP(DecompGEP);
		//I->getParent()->dump();
	} else if (offset == UNKNOWNOFFSET || offset == VARIABLEOFFSET) {
		// TODO: treat it the same way as array
		/*
		errs() << "UNKNOWN offset encountered\t";
		errs() << "Addr: " << *Addr << "\n";
		errs() << *I << "\n";
		getPosition(I, IRB, true);
		I->getParent()->dump();
		*/
	} else {
		unsigned TypeSize = getMemoryAccessSize(Addr, DL);
		ob_state_t *object_state = (*map)[DecompGEP.Base];
		if (object_state == NULL) {
			object_state = new ob_state_t();
			(*map)[DecompGEP.Base] = object_state;
			updated |= true;
#ifdef PMROBUST_DEBUG
			value_list.push_back(DecompGEP.Base);
#endif
		}

		updated |= object_state->setDirty(offset, TypeSize);

		// Rule 2.1: *x = p (where x is a heap address) => all fields of p escapes
		// TODO: Val should be PM Addr
		if (Val && Val->getType()->isPointerTy() &&
			mayInHeap(DecompGEP.Base)) {
			DecomposedGEP ValDecompGEP;
			decomposeAddress(ValDecompGEP, Val, DL);
			unsigned offset = ValDecompGEP.getOffsets();

			if (ValDecompGEP.isArray) {
				// e.g. struct { int A; int arr[10]; };
				// StructOffset is the beginning address of arr.
				// TODO: can we compute the size of the arr field?
				ob_state_t *object_state = (*map)[ValDecompGEP.Base];
				if (object_state == NULL) {
					object_state = new ob_state_t();
					(*map)[ValDecompGEP.Base] = object_state;
					updated |= true;
				}

				// Mark the start byte of the arr as escaped
				unsigned startByte = ValDecompGEP.getStructOffset();
				updated |= object_state->setEscape(startByte, 1);
			} else if (offset == UNKNOWNOFFSET || offset == VARIABLEOFFSET) {
				// TODO: start working here
				// assert(false && "Fix me");
			} else {
				ob_state_t *object_state = (*map)[ValDecompGEP.Base];
				if (object_state == NULL) {
					object_state = new ob_state_t();
					(*map)[ValDecompGEP.Base] = object_state;
					updated |= true;
				}

				if (offset == 0) {
					// Mark the entire object as escaped
					// FIXME: the offset of the first field is also 0;
					// could not tell if the object or the first field escapes
					updated |= object_state->setEscape(0, object_state->getSize(), true);
				} else {
					// Only mark this field as escaped
					// Example: *x = &p->f;

					// Get the size of the field p->f
					unsigned TypeSize = getFieldSize(Val, DL);
					updated |= object_state->setEscape(offset, TypeSize);
				}
			}
		}
	}

	return updated;
}

bool PMRobustness::processLoad(state_t * map, Instruction * I) {
	bool updated = false;
	IRBuilder<> IRB(I);
	const DataLayout &DL = I->getModule()->getDataLayout();
	Value *Addr = NULL;

	if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
		Addr = LI->getPointerOperand();
	} else if (AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(I)) {
		Addr = RMWI->getPointerOperand();
	} else {
		return false;
	}

	if (!isPMAddr(Addr))
		return false;

	DecomposedGEP DecompGEP;
	decomposeAddress(DecompGEP, Addr, DL);
	unsigned offset = DecompGEP.getOffsets();

	if (DecompGEP.isArray) {
		//TODO: implement robustness checks for array
		//errs() << I->getFunction()->getName() << "\n";
		//errs() << "Array encountered\t";
		//errs() << "Addr: " << *Addr << "\n";
		//errs() << *I << "\n";
		//getPosition(I, IRB, true);
	} else if (offset == UNKNOWNOFFSET || offset == VARIABLEOFFSET) {
		// TODO: treat it the same way as array
	} else {
		if (I->isAtomic() && isa<LoadInst>(I)) {
			// Mark the address as dirty to detect interthread robustness violation
			// For Atomic RMW, this is already done in processStore
			unsigned TypeSize = getMemoryAccessSize(Addr, DL);

			ob_state_t *object_state = (*map)[DecompGEP.Base];
			if (object_state == NULL) {
				object_state = new ob_state_t();
				(*map)[DecompGEP.Base] = object_state;
				updated |= true;
#ifdef PMROBUST_DEBUG
				value_list.push_back(DecompGEP.Base);
#endif
			}

			updated |= object_state->setDirty(offset, TypeSize);
		}

		// Rule 2.2: x = *p (where p is a heap address) => x escapes
		if (I->getType()->isPointerTy() && mayInHeap(DecompGEP.Base)) {
			DecomposedGEP LIDecompGEP;
			decomposeAddress(LIDecompGEP, I, DL);
			unsigned offset = LIDecompGEP.getOffsets();

			if (LIDecompGEP.isArray) {
				assert(false && "Fix me");
			} else if (offset > 0) {
				assert(false && "fix me");
			}

			unsigned TypeSize = getMemoryAccessSize(Addr, DL);
			ob_state_t *object_state = (*map)[LIDecompGEP.Base];
			if (object_state == NULL) {
				object_state = new ob_state_t();
				(*map)[LIDecompGEP.Base] = object_state;
				updated |= true;
#ifdef PMROBUST_DEBUG
				value_list.push_back(I);
#endif
			}

			updated |= object_state->setEscape(0, object_state->getSize(), true);
		}
	}

	return updated;
}

bool PMRobustness::processFlushWrapperFunction(state_t * map, Instruction * I) {
	CallInst *callInst = cast<CallInst>(I);
	const DataLayout &DL = I->getModule()->getDataLayout();
	Function *callee = callInst->getCalledFunction();
	assert(callee);

	std::vector<StringRef> annotations;
	StringRef AttrValue = callee->getFnAttribute("myflush").getValueAsString();
	std::pair<StringRef, StringRef> Split = AttrValue.split("|");
	annotations.push_back(Split.first);

	while (!Split.second.empty()) {
		Split = Split.second.split("|");
		annotations.push_back(Split.first);
	}

	assert(callInst->arg_size() == annotations.size() &&
		"annotations should match the number of paramaters");
	Value *Addr = NULL;
	Value *FlushSize = NULL;
	for (unsigned i = 0; i < annotations.size(); i++) {
		StringRef &token = annotations[i];
		if (token == "addr") {
			Addr = callInst->getArgOperand(i);
		} else if (token == "size") {
			FlushSize = callInst->getArgOperand(i);
		} else {
			assert(false && "bad annotation");
		}
	}

	NVMOP FlushOp = NVM_UNKNOWN;
	if (callee->hasFnAttribute("flush_type")) {
		StringRef flushtype = callee->getFnAttribute("flush_type").getValueAsString();
		FlushOp = whichNVMoperation(flushtype);
	} else {
		FlushOp = analyzeFlushType(*callee);
	}
	assert(FlushOp == NVM_CLWB || FlushOp == NVM_CLFLUSH);

	DecomposedGEP DecompGEP;
	decomposeAddress(DecompGEP, Addr, DL);
	unsigned offset = DecompGEP.getOffsets();

	if (DecompGEP.isArray) {
		//TODO: implement robustness checks for array
	} else if (offset == UNKNOWNOFFSET || offset == VARIABLEOFFSET) {
		// TODO: treat it the same way as array
	} else {
		//unsigned TypeSize = getMemoryAccessSize(Addr, DL);
		ob_state_t *object_state = (*map)[DecompGEP.Base];
		if (object_state == NULL) {
			// TODO: to be solve in interprocedural analysis
			//errs() << "FLush an unknown address\n";
			assert(false && "Flush an unknown address");
		} else {
			unsigned size = cast<ConstantInt>(FlushSize)->getZExtValue();
			//errs() << "flush " << *DecompGEP.Base << " from " << offset << " to " << size << "\n";

			if (FlushOp == NVM_CLFLUSH)
				object_state->setFlush(offset, size);
			else if (FlushOp == NVM_CLWB)
				object_state->setClwb(offset, size);
		}
	}

	return true;
}

/*
bool PMRobustness::processFlush(state_t * map, Instruction * I, NVMOP op) {
	CallInst *callInst = cast<CallInst>(I);
	const DataLayout &DL = I->getModule()->getDataLayout();

	Value *Addr = callInst->getArgOperand(0);

	DecomposedGEP DecompGEP;
	decomposeAddress(DecompGEP, Addr, DL);
	unsigned offset = DecompGEP.getOffsets();

	if (DecompGEP.isArray) {
		//TODO: implement robustness checks for array
	} else if (offset == UNKNOWNOFFSET) {
		// TODO: treat it the same way as array
	} else {
		unsigned TypeSize = getMemoryAccessSize(Addr, DL);
		ob_state_t *object_state = (*map)[DecompGEP.Base];
		if (object_state == NULL) {
			// TODO: to be solve in interprocedural analysis
			errs() << "FLush an unknown address\n";
		} else {
			printDecomposedGEP(DecompGEP);
			object_state->setFlush(offset, -1);
		}
	}

	return true;
}*/

unsigned PMRobustness::getMemoryAccessSize(Value *Addr, const DataLayout &DL) {
	Type *OrigPtrTy = Addr->getType();
	Type *OrigTy = cast<PointerType>(OrigPtrTy)->getElementType();
	assert(OrigTy->isSized());
	unsigned TypeSize = DL.getTypeStoreSizeInBits(OrigTy);
    if (TypeSize != 8  && TypeSize != 16 &&
		TypeSize != 32 && TypeSize != 64 && TypeSize != 128) {
		//NumAccessesWithBadSize++;
		// Ignore all unusual sizes.

		assert(false && "Bad size access\n");
    }

	return TypeSize / 8;
}

unsigned PMRobustness::getFieldSize(Value *Addr, const DataLayout &DL) {
	Type *OrigPtrTy = Addr->getType();
	Type *OrigTy = cast<PointerType>(OrigPtrTy)->getElementType();
	assert(OrigTy->isSized());
	unsigned TypeSize = DL.getTypeStoreSizeInBits(OrigTy);

	return TypeSize / 8;
}


void PMRobustness::getAnalysisUsage(AnalysisUsage &AU) const {
	AU.setPreservesAll();
	//AU.addRequired<AAResultsWrapperPass>();
	//AU.addRequired<AndersenAAWrapperPass>();
}

bool PMRobustness::skipFunction(Function &F) {
	if (F.hasFnAttribute("myflush"))
		return false;

	return false;
}

/** Simple may-analysis for checking if an address is in the heap
 *  TODO: may need more sophisticated checks
 **/
bool PMRobustness::mayInHeap(const Value * Addr) {
	if (GetElementPtrInst * GEP = dyn_cast<GetElementPtrInst>((Value *)Addr)) {
		Value * BaseAddr = GEP->getPointerOperand();

		for (auto &u : BaseAddr->uses()) {
			if (isa<AllocaInst>(u)) {
				return false;
			}
		}
	} else {
		for (auto &u : Addr->uses()) {
			if (isa<AllocaInst>(u)) {
				return false;
			}
		}
	}

	// Address may be in the heap. We don't know for sure.
	return true;
}

void PMRobustness::decomposeAddress(DecomposedGEP &DecompGEP, Value *Addr, const DataLayout &DL) {
	unsigned MaxPointerSize = getMaxPointerSize(DL);
	DecompGEP.StructOffset = DecompGEP.OtherOffset = APInt(MaxPointerSize, 0);
	DecompGEP.isArray = false;
	bool GEPMaxLookupReached = DecomposeGEPExpression(Addr, DecompGEP, DL, NULL, NULL);

	if (GEPMaxLookupReached) {
		errs() << "GEP Max Lookup Reached\n";
#ifdef PMROBUST_DEBUG
		assert(false && "GEP Max Lookup Reached; debug\n");
#endif
	}
}

NVMOP PMRobustness::whichNVMoperation(Instruction *I) {
	IRBuilder<> IRB(I);

	if (CallInst *callInst = dyn_cast<CallInst>(I)) {
		if (callInst->isInlineAsm()) {
			InlineAsm *asmInline = dyn_cast<InlineAsm>(callInst->getCalledOperand());
			StringRef asmStr = asmInline->getAsmString();
			if (asmStr.contains("mfence") || asmStr.contains("sfence")) {
				//errs() << "mfence/sfence seen at\n";
				//getPosition(I, IRB, true);
				//errs() << *I << "\n";
				return NVM_FENCE;
			} else if (asmStr.contains("xsaveopt") || asmStr.contains("clflushopt")) {
				//errs() << "clflushopt seen at\n";
				//getPosition(I, IRB, true);
				//errs() << *I << "\n";
				return NVM_CLWB;
			} else if (asmStr.contains("clflush")) {
				//errs() << "clflush seen at\n";
				//getPosition(I, IRB, true);
				//errs() << *I << "\n";
				return NVM_CLFLUSH;
			}
		}
	}
	return NVM_UNKNOWN;
}

NVMOP PMRobustness::whichNVMoperation(StringRef flushtype) {
	if (flushtype.contains("mfence") || flushtype.contains("sfence")) {
		return NVM_FENCE;
	} else if (flushtype.contains("clflushopt")) {
		return NVM_CLWB;
	} else if (flushtype.contains("clflush")) {
		return NVM_CLFLUSH;
	}
	return NVM_UNKNOWN;
}

NVMOP PMRobustness::analyzeFlushType(Function &F) {
	NVMOP op = NVM_UNKNOWN;
	for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
		op = whichNVMoperation(&*I);
		if (op != NVM_UNKNOWN) {
			if (op == NVM_CLWB) {
				F.addFnAttr("flush_type", "clflushopt");
			} else if (op == NVM_CLFLUSH) {
				F.addFnAttr("flush_type", "clflush");
			}
			break;
		}
	}

	return op;
}

bool PMRobustness::isFlushWrapperFunction(Instruction *I) {
	if (CallInst *CI = dyn_cast<CallInst>(I)) {
		if (Function *callee = CI->getCalledFunction()) {
			if (callee->hasFnAttribute("myflush"))
				return true;
		}
	}

	return false;
}


//===----------------------------------------------------------------------===//
// GetElementPtr Instruction Decomposition and Analysis
//===----------------------------------------------------------------------===//
/// Copied from Analysis/BasicAliasAnalysis.cpp
/// Analyzes the specified value as a linear expression: "A*V + B", where A and
/// B are constant integers.
///
/// Returns the scale and offset values as APInts and return V as a Value*, and
/// return whether we looked through any sign or zero extends.  The incoming
/// Value is known to have IntegerType, and it may already be sign or zero
/// extended.
///
/// Note that this looks through extends, so the high bits may not be
/// represented in the result.
const Value *PMRobustness::GetLinearExpression(
	const Value *V, APInt &Scale, APInt &Offset, unsigned &ZExtBits,
	unsigned &SExtBits, const DataLayout &DL, unsigned Depth,
	AssumptionCache *AC, DominatorTree *DT, bool &NSW, bool &NUW) {

	assert(V->getType()->isIntegerTy() && "Not an integer value");
	// Limit our recursion depth.
	if (Depth == 6) {
		Scale = 1;
		Offset = 0;
		return V;
	}

	if (const ConstantInt *Const = dyn_cast<ConstantInt>(V)) {
		// If it's a constant, just convert it to an offset and remove the variable.
		// If we've been called recursively, the Offset bit width will be greater
		// than the constant's (the Offset's always as wide as the outermost call),
		// so we'll zext here and process any extension in the isa<SExtInst> &
		// isa<ZExtInst> cases below.
		Offset += Const->getValue().zextOrSelf(Offset.getBitWidth());
		assert(Scale == 0 && "Constant values don't have a scale");
		return V;
	}

	if (const BinaryOperator *BOp = dyn_cast<BinaryOperator>(V)) {
		if (ConstantInt *RHSC = dyn_cast<ConstantInt>(BOp->getOperand(1))) {
			// If we've been called recursively, then Offset and Scale will be wider
			// than the BOp operands. We'll always zext it here as we'll process sign
			// extensions below (see the isa<SExtInst> / isa<ZExtInst> cases).
			APInt RHS = RHSC->getValue().zextOrSelf(Offset.getBitWidth());

			switch (BOp->getOpcode()) {
			default:
				// We don't understand this instruction, so we can't decompose it any
				// further.
				Scale = 1;
				Offset = 0;
				return V;
			case Instruction::Or:
				// X|C == X+C if all the bits in C are unset in X.	Otherwise we can't
				// analyze it.
				if (!MaskedValueIsZero(BOp->getOperand(0), RHSC->getValue(), DL, 0, AC,
						 BOp, DT)) {
					Scale = 1;
					Offset = 0;
					return V;
				}
				LLVM_FALLTHROUGH;
			case Instruction::Add:
				V = GetLinearExpression(BOp->getOperand(0), Scale, Offset, ZExtBits,
						SExtBits, DL, Depth + 1, AC, DT, NSW, NUW);
				Offset += RHS;
				break;
			case Instruction::Sub:
				V = GetLinearExpression(BOp->getOperand(0), Scale, Offset, ZExtBits,
						SExtBits, DL, Depth + 1, AC, DT, NSW, NUW);
				Offset -= RHS;
				break;
			case Instruction::Mul:
				V = GetLinearExpression(BOp->getOperand(0), Scale, Offset, ZExtBits,
						SExtBits, DL, Depth + 1, AC, DT, NSW, NUW);
				Offset *= RHS;
				Scale *= RHS;
				break;
			case Instruction::Shl:
				V = GetLinearExpression(BOp->getOperand(0), Scale, Offset, ZExtBits,
						SExtBits, DL, Depth + 1, AC, DT, NSW, NUW);

				// We're trying to linearize an expression of the kind:
				//	 shl i8 -128, 36
				// where the shift count exceeds the bitwidth of the type.
				// We can't decompose this further (the expression would return
				// a poison value).
				if (Offset.getBitWidth() < RHS.getLimitedValue() ||
						Scale.getBitWidth() < RHS.getLimitedValue()) {
					Scale = 1;
					Offset = 0;
					return V;
				}

				Offset <<= RHS.getLimitedValue();
				Scale <<= RHS.getLimitedValue();
				// the semantics of nsw and nuw for left shifts don't match those of
				// multiplications, so we won't propagate them.
				NSW = NUW = false;
				return V;
			}

			if (isa<OverflowingBinaryOperator>(BOp)) {
				NUW &= BOp->hasNoUnsignedWrap();
				NSW &= BOp->hasNoSignedWrap();
			}
			return V;
		}
	}

	// Since GEP indices are sign extended anyway, we don't care about the high
	// bits of a sign or zero extended value - just scales and offsets.	The
	// extensions have to be consistent though.
	if (isa<SExtInst>(V) || isa<ZExtInst>(V)) {
		Value *CastOp = cast<CastInst>(V)->getOperand(0);
		unsigned NewWidth = V->getType()->getPrimitiveSizeInBits();
		unsigned SmallWidth = CastOp->getType()->getPrimitiveSizeInBits();
		unsigned OldZExtBits = ZExtBits, OldSExtBits = SExtBits;
		const Value *Result =
				GetLinearExpression(CastOp, Scale, Offset, ZExtBits, SExtBits, DL,
					Depth + 1, AC, DT, NSW, NUW);

		// zext(zext(%x)) == zext(%x), and similarly for sext; we'll handle this
		// by just incrementing the number of bits we've extended by.
		unsigned ExtendedBy = NewWidth - SmallWidth;

		if (isa<SExtInst>(V) && ZExtBits == 0) {
			// sext(sext(%x, a), b) == sext(%x, a + b)

			if (NSW) {
				// We haven't sign-wrapped, so it's valid to decompose sext(%x + c)
				// into sext(%x) + sext(c). We'll sext the Offset ourselves:
				unsigned OldWidth = Offset.getBitWidth();
				Offset = Offset.trunc(SmallWidth).sext(NewWidth).zextOrSelf(OldWidth);
			} else {
				// We may have signed-wrapped, so don't decompose sext(%x + c) into
				// sext(%x) + sext(c)
				Scale = 1;
				Offset = 0;
				Result = CastOp;
				ZExtBits = OldZExtBits;
				SExtBits = OldSExtBits;
			}
			SExtBits += ExtendedBy;
		} else {
			// sext(zext(%x, a), b) = zext(zext(%x, a), b) = zext(%x, a + b)

			if (!NUW) {
				// We may have unsigned-wrapped, so don't decompose zext(%x + c) into
				// zext(%x) + zext(c)
				Scale = 1;
				Offset = 0;
				Result = CastOp;
				ZExtBits = OldZExtBits;
				SExtBits = OldSExtBits;
			}
			ZExtBits += ExtendedBy;
		}

		return Result;
	}

	Scale = 1;
	Offset = 0;
	return V;
}

/// If V is a symbolic pointer expression, decompose it into a base pointer
/// with a constant offset and a number of scaled symbolic offsets.
///
/// The scaled symbolic offsets (represented by pairs of a Value* and a scale
/// in the VarIndices vector) are Value*'s that are known to be scaled by the
/// specified amount, but which may have other unrepresented high bits. As
/// such, the gep cannot necessarily be reconstructed from its decomposed form.
///
/// When DataLayout is around, this function is capable of analyzing everything
/// that GetUnderlyingObject can look through. To be able to do that
/// GetUnderlyingObject and DecomposeGEPExpression must use the same search
/// depth (MaxLookupSearchDepth). When DataLayout not is around, it just looks
/// through pointer casts.
bool PMRobustness::DecomposeGEPExpression(const Value *V,
		DecomposedGEP &Decomposed, const DataLayout &DL, AssumptionCache *AC,
		DominatorTree *DT) {
	// Limit recursion depth to limit compile time in crazy cases.
	unsigned MaxLookup = MaxLookupSearchDepth;
	//SearchTimes++;

	unsigned MaxPointerSize = getMaxPointerSize(DL);
	Decomposed.VarIndices.clear();
	do {
		// See if this is a bitcast or GEP.
		const Operator *Op = dyn_cast<Operator>(V);
		if (!Op) {
			// The only non-operator case we can handle are GlobalAliases.
			if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(V)) {
				if (!GA->isInterposable()) {
					V = GA->getAliasee();
					continue;
				}
			}
			Decomposed.Base = V;
			return false;
		}

		if (Op->getOpcode() == Instruction::BitCast ||
				Op->getOpcode() == Instruction::AddrSpaceCast) {
			V = Op->getOperand(0);
			continue;
		}

		const GEPOperator *GEPOp = dyn_cast<GEPOperator>(Op);
		if (!GEPOp) {
			if (const auto *Call = dyn_cast<CallBase>(V)) {
				// CaptureTracking can know about special capturing properties of some
				// intrinsics like launder.invariant.group, that can't be expressed with
				// the attributes, but have properties like returning aliasing pointer.
				// Because some analysis may assume that nocaptured pointer is not
				// returned from some special intrinsic (because function would have to
				// be marked with returns attribute), it is crucial to use this function
				// because it should be in sync with CaptureTracking. Not using it may
				// cause weird miscompilations where 2 aliasing pointers are assumed to
				// noalias.
				if (auto *RP = getArgumentAliasingToReturnedPointer(Call)) {
					V = RP;
					continue;
				}
			}

			// If it's not a GEP, hand it off to SimplifyInstruction to see if it
			// can come up with something. This matches what GetUnderlyingObject does.
			if (const Instruction *I = dyn_cast<Instruction>(V))
				// TODO: Get a DominatorTree and AssumptionCache and use them here
				// (these are both now available in this function, but this should be
				// updated when GetUnderlyingObject is updated). TLI should be
				// provided also.
				if (const Value *Simplified =
								SimplifyInstruction(const_cast<Instruction *>(I), DL)) {
					V = Simplified;
					continue;
				}

			Decomposed.Base = V;
			return false;
		}

		// Don't attempt to analyze GEPs over unsized objects.
		if (!GEPOp->getSourceElementType()->isSized()) {
			Decomposed.Base = V;
			return false;
		}

		unsigned AS = GEPOp->getPointerAddressSpace();
		// Walk the indices of the GEP, accumulating them into BaseOff/VarIndices.
		gep_type_iterator GTI = gep_type_begin(GEPOp);
		unsigned PointerSize = DL.getPointerSizeInBits(AS);
		// Assume all GEP operands are constants until proven otherwise.
		bool GepHasConstantOffset = true;

		// Detecing array
		const Value * firstIndex = *(GEPOp->op_begin() + 1);
		if (const ConstantInt *CIdx = dyn_cast<ConstantInt>(firstIndex)) {
			unsigned FieldNo = CIdx->getZExtValue();
			if (FieldNo != 0 || dyn_cast<ArrayType>(GEPOp->getSourceElementType())) {
				Decomposed.isArray = true;
				//return false;
			}
		}

		for (User::const_op_iterator I = GEPOp->op_begin() + 1, E = GEPOp->op_end();
				 I != E; ++I, ++GTI) {
			const Value *Index = *I;
			// Compute the (potentially symbolic) offset in bytes for this index.
			if (StructType *STy = GTI.getStructTypeOrNull()) {
				// For a struct, add the member offset.
				unsigned FieldNo = cast<ConstantInt>(Index)->getZExtValue();
				if (FieldNo == 0)
					continue;

				Decomposed.StructOffset +=
					DL.getStructLayout(STy)->getElementOffset(FieldNo);

				continue;
			}

			// For an array/pointer, add the element offset, explicitly scaled.
			if (const ConstantInt *CIdx = dyn_cast<ConstantInt>(Index)) {
				if (CIdx->isZero())
					continue;
				Decomposed.OtherOffset +=
					(DL.getTypeAllocSize(GTI.getIndexedType()) *
						CIdx->getValue().sextOrSelf(MaxPointerSize))
						.sextOrTrunc(MaxPointerSize);
				continue;
			}

			GepHasConstantOffset = false;

			APInt Scale(MaxPointerSize, DL.getTypeAllocSize(GTI.getIndexedType()));
			unsigned ZExtBits = 0, SExtBits = 0;

			// If the integer type is smaller than the pointer size, it is implicitly
			// sign extended to pointer size.
			unsigned Width = Index->getType()->getIntegerBitWidth();
			if (PointerSize > Width)
				SExtBits += PointerSize - Width;

			// Use GetLinearExpression to decompose the index into a C1*V+C2 form.
			APInt IndexScale(Width, 0), IndexOffset(Width, 0);
			bool NSW = true, NUW = true;
			const Value *OrigIndex = Index;
			Index = GetLinearExpression(Index, IndexScale, IndexOffset, ZExtBits,
																	SExtBits, DL, 0, AC, DT, NSW, NUW);

			// The GEP index scale ("Scale") scales C1*V+C2, yielding (C1*V+C2)*Scale.
			// This gives us an aggregate computation of (C1*Scale)*V + C2*Scale.

			// It can be the case that, even through C1*V+C2 does not overflow for
			// relevant values of V, (C2*Scale) can overflow. In that case, we cannot
			// decompose the expression in this way.
			//
			// FIXME: C1*Scale and the other operations in the decomposed
			// (C1*Scale)*V+C2*Scale can also overflow. We should check for this
			// possibility.
			APInt WideScaledOffset = IndexOffset.sextOrTrunc(MaxPointerSize*2) *
																 Scale.sext(MaxPointerSize*2);
			if (WideScaledOffset.getMinSignedBits() > MaxPointerSize) {
				Index = OrigIndex;
				IndexScale = 1;
				IndexOffset = 0;

				ZExtBits = SExtBits = 0;
				if (PointerSize > Width)
					SExtBits += PointerSize - Width;
			} else {
				Decomposed.OtherOffset += IndexOffset.sextOrTrunc(MaxPointerSize) * Scale;
				Scale *= IndexScale.sextOrTrunc(MaxPointerSize);
			}

			// If we already had an occurrence of this index variable, merge this
			// scale into it.	For example, we want to handle:
			//	 A[x][x] -> x*16 + x*4 -> x*20
			// This also ensures that 'x' only appears in the index list once.
			for (unsigned i = 0, e = Decomposed.VarIndices.size(); i != e; ++i) {
				if (Decomposed.VarIndices[i].V == Index &&
						Decomposed.VarIndices[i].ZExtBits == ZExtBits &&
						Decomposed.VarIndices[i].SExtBits == SExtBits) {
					Scale += Decomposed.VarIndices[i].Scale;
					Decomposed.VarIndices.erase(Decomposed.VarIndices.begin() + i);
					break;
				}
			}

			// Make sure that we have a scale that makes sense for this target's
			// pointer size.
			Scale = adjustToPointerSize(Scale, PointerSize);

			if (!!Scale) {
				VariableGEPIndex Entry = {Index, ZExtBits, SExtBits, Scale};
				Decomposed.VarIndices.push_back(Entry);
			}
		}

		// Take care of wrap-arounds
		if (GepHasConstantOffset) {
			Decomposed.StructOffset =
					adjustToPointerSize(Decomposed.StructOffset, PointerSize);
			Decomposed.OtherOffset =
					adjustToPointerSize(Decomposed.OtherOffset, PointerSize);
		}

		// Analyze the base pointer next.
		V = GEPOp->getOperand(0);
	} while (--MaxLookup);

	// If the chain of expressions is too deep, just return early.
	Decomposed.Base = V;
	//SearchLimitReached++;
	return true;
}

void PMRobustness::printMap(state_t * map) {
	for (state_t::iterator it = map->begin(); it != map->end(); it++) {
		ob_state_t *object_state = it->second;
		errs() << "loc: " << *it->first << "\n";
		object_state->print();
	}

//	errs() << "map size: " << map->size() << "\n";
//	for (unsigned i = 0; i < value_list.size(); i++) {
//		Value * val = (Value *)value_list[i];
//		ob_state_t *object_state = (*map)[val];
//		object_state->print();
//	}
	errs() << "\n**\n";
}

void PMRobustness::test() {
	typedef DenseMap<int *, int *> type_t;
	type_t * map = new type_t();

	int *x = new int(2);
	int *y = new int(3);

	(*map)[x] = x;
	(*map)[y] = y;

	errs() << "x: " << *x << "\n";
	delete map;
	errs() << "x: " << *x << "\n";
}

char PMRobustness::ID = 0;
static RegisterPass<PMRobustness> X("pmrobust", "Persistent Memory Robustness Analysis Pass");

// Automatically enable the pass.
static void registerPMRobustness(const PassManagerBuilder &,
							legacy::PassManagerBase &PM) {
	PM.add(createPromoteMemoryToRegisterPass());
	PM.add(createEarlyCSEPass(false));
	PM.add(new PMRobustness());
}
/* Enable the pass when opt level is greater than 0 */
static RegisterStandardPasses 
	RegisterMyPass1(PassManagerBuilder::EP_EarlyAsPossible,
registerPMRobustness);

/* Enable the pass when opt level is 0 */
// Don't turn this on when using EP_EarlyAsPossible. Otherwise, the pass will run twice
//static RegisterStandardPasses
//	RegisterMyPass2(PassManagerBuilder::EP_EnabledOnOptLevel0,
//registerPMRobustness);
