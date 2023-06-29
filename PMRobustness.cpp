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
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
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
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/ADT/None.h"
#include "PMRobustness.h"
#include "FunctionSummary.h"
//#include "andersen/include/AndersenAA.h"
//#include "llvm/Analysis/AliasAnalysis.h"

//#define PMROBUST_DEBUG
#define INTERPROCEDURAL
#define DEBUG_TYPE "PMROBUST_DEBUG"
#include <llvm/IR/DebugLoc.h>

namespace {
	struct PMRobustness : public ModulePass {
		PMRobustness() : ModulePass(ID) {}
		StringRef getPassName() const override;
		bool doInitialization(Module &M) override;
		bool runOnModule(Module &M) override;
		void getAnalysisUsage(AnalysisUsage &AU) const override;
		void analyzeFunction(Function &F, CallingContext *Context);

		static char ID;
		//AliasAnalysis *AA;
		//AndersenAAResult *AA;

	private:
		void copyState(state_t * src, state_t * dst);
		void copyArrayState(addr_set_t *src, addr_set_t *dst);
		bool copyStateCheckDiff(state_t * src, state_t * dst);

		void copyMergedState(state_map_t *AbsState, SmallPtrSetImpl<BasicBlock *> * src_list,
			state_t * dst, DenseMap<const BasicBlock *, bool> &visited_blocks);
		void copyMergedArrayState(DenseMap<BasicBlock *, addr_set_t *> *ArraySets,
			SmallPtrSetImpl<BasicBlock *> *src_list, addr_set_t *dst,
			DenseMap<const BasicBlock *, bool> &visited_blocks);

		void processInstruction(state_t * map, Instruction * I);
		bool processAtomic(state_t * map, Instruction * I);
		bool processMemIntrinsic(state_t * map, Instruction * I);
		bool processLoad(state_t * map, Instruction * I);
		bool processStore(state_t * map, Instruction * I);
		bool processPHI(state_t * map, Instruction * I);
		void processFlushWrapperFunction(state_t * map, Instruction * I);
		void processFlushParameterFunction(state_t * map, Instruction * I);
		void processNTSWrapperFunction(state_t * map, Instruction * I);
		void processCalls(state_t *map, Instruction *I, bool &non_dirty_escaped_before);
		void getAnnotatedParamaters(std::string attr, std::vector<StringRef> &annotations, Function *callee);

		//bool processParamAnnotationFunction(Instruction * I);

		bool skipFunction(Function &F);
		// TODO: Address check to be implemented
		bool isPMAddr(const Value * Addr, const DataLayout &DL);
		bool mayInHeap(const Value * Addr);
		ob_state_t * getObjectState(state_t *map, const Value *Addr, bool &updated);
		ob_state_t * getObjectState(state_t *map, const Value *Addr);

		void checkEndError(state_map_t *AbsState, Function &F);
		void checkEscapedObjError(state_t *map, Instruction *I, bool non_dirty_escaped_before);

		void decomposeAddress(DecomposedGEP &DecompGEP, Value *Addr, const DataLayout &DL);
		unsigned getMemoryAccessSize(Value *Addr, const DataLayout &DL);
		unsigned getFieldSize(Value *Addr, const DataLayout &DL);
		NVMOP whichNVMoperation(Instruction *I);
		NVMOP whichNVMoperation(StringRef flushtype);
		NVMOP analyzeFlushType(Function &F);
		//bool isParamAnnotationFunction(Instruction *I);
		bool isFlushWrapperFunction(Instruction *I);
		bool isFlushParameterFunction(Instruction *I);
		bool ignoreFunction(Instruction *I);
		bool isNTSWrapperFunction(Instruction *I);

		addr_set_t * getOrCreateUnflushedAddrSet(Function *F, BasicBlock *B);
		bool checkUnflushedAddress(Function *F, addr_set_t * AddrSet, Value * Addr, DecomposedGEP &DecompGEP);
		bool compareDecomposedGEP(DecomposedGEP &GEP1, DecomposedGEP &GEP2);

		CallingContext * computeContext(state_t *map, Instruction *I);
		void lookupFunctionResult(state_t *map, CallBase *CB, CallingContext *Context, bool &non_dirty_escaped_before);
		void computeInitialState(state_t *map, Function &F, CallingContext *Context);
		bool computeFinalState(state_map_t *AbsState, Function &F, CallingContext *Context);

		void makeParametersTOP(state_t *map, CallBase *CB);
		void modifyReturnState(state_t *map, CallBase *CB, OutputState *out_state);

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

		// object states at the end of each instruction for each function
		DenseMap<Function *, state_map_t *> AbstractStates;
		// object states at the end of each basic block for each function
		DenseMap<Function *, b_state_map_t *> BlockEndAbstractStates;
		DenseMap<Function *, DenseMap<BasicBlock *, addr_set_t *> *> UnflushedArrays;
		DenseMap<Function *, FunctionSummary *> FunctionSummaries;
		DenseMap<Function *, SmallPtrSet<Instruction *, 8> > FunctionRetInstMap;

		CallingContext *CurrentContext;

		// Arguments of the function being analyzed
		DenseMap<Value *, unsigned> FunctionArguments;

		// Map a Function to its call sites
		DenseMap<Function *, SmallDenseSet<std::pair<Function *, CallingContext *>> > FunctionCallerMap;

		std::list<std::pair<Function *, CallingContext *>> FunctionWorklist;

		// The set of items in FunctionWorklist; used to avoid insert duplicate items to FunctionWorklist
		DenseSet<std::pair<Function *, CallingContext *>> UniqueFunctionSet;

		std::vector<BasicBlock *> unprocessed_blocks;
		std::list<BasicBlock *> BlockWorklist;

		DenseMap<Function *, DenseSet<const Value *> *> FunctionEndErrorSets;
		DenseMap<Function *, DenseSet<const Instruction *> *> FunctionStmtErrorSets;
		DenseSet<const Instruction *> * StmtErrorSet;
		bool hasTwoEscapedDirtyParams;
		bool InstructionMarksEscDirObj;	// Instruction: current instruction
		bool FunctionMarksEscDirObj;	// Function: this function

		// Call: this call instruction marks any object as dirty and escaped when non parameters are dirty and escaped;
		bool CallMarksEscDirObj;
		bool hasError;

		unsigned MaxLookupSearchDepth = 100;
		std::set<std::string> MemAllocatingFunctions;

		// May consider having several DenseMaps for function with different parameter sizes: 4, 8, 12, 16, etc.
	};
}

StringRef PMRobustness::getPassName() const {
	return "PMRobustness";
}

/* annotations -> myflush:[addr|size|ignore] */
bool PMRobustness::doInitialization (Module &M) {
	MemAllocatingFunctions = {
		"operator new(unsigned long)", "operator new[](unsigned long)", "malloc",
		"calloc", "realloc"
	};

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

bool PMRobustness::runOnModule(Module &M) {
	for (Function &F : M) {
#ifdef INTERPROCEDURAL
		if (F.getName() == "main") {
			// Function has parameters
			CallingContext *Context = new CallingContext();
			for (unsigned i = 0; i < F.arg_size(); i++)
				Context->addAbsInput(ParamStateType::NON_PMEM);

			FunctionWorklist.emplace_back(&F, Context);
			UniqueFunctionSet.insert(std::make_pair(&F, Context));
		} else if (!F.isDeclaration()) {
			// Assume parameters have states TOP or do not push them to worklist?
			CallingContext *Context = new CallingContext();
			for (Function::arg_iterator it = F.arg_begin(); it != F.arg_end(); it++) {
				Argument *Arg = &*it;
				if (Arg->getType()->isPointerTy())
					Context->addAbsInput(ParamStateType::TOP);
				else
					Context->addAbsInput(ParamStateType::NON_PMEM);
			}

			FunctionWorklist.emplace_back(&F, Context);
			UniqueFunctionSet.insert(std::make_pair(&F, Context));
		} else {
			// F.isDeclaration
			//errs() << F.getName() << " isDeclaration ignored in Module " << M.getName() << "\n";

 #ifdef PMROBUST_DEBUG
			errs() << "{" << F.empty() << "," << !F.isMaterializable() << "}\n";
			errs() << F.getName() << " isDeclaration ignored\n";
 #endif
		}
#else
		if (!F.isDeclaration())
			FunctionWorklist.emplace_back(&F, new CallingContext());
#endif
	}

	while (!FunctionWorklist.empty()) {
		std::pair<Function *, CallingContext *> &pair = FunctionWorklist.front();
		FunctionWorklist.pop_front();
		UniqueFunctionSet.erase(pair);

		Function *F = pair.first;
		CallingContext *context = pair.second;

		if (skipFunction(*F)) {
			continue;
		}

		//AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
		//AA = &getAnalysis<AndersenAAWrapperPass>().getResult();

		//errs() << "processing " << F->getName() << "\n";
		analyzeFunction(*F, context);
		delete context;
	}

	return true;
}

void PMRobustness::analyzeFunction(Function &F, CallingContext *Context) {
	state_map_t *AbsState = AbstractStates[&F];
	if (AbsState == NULL) {
		AbsState = new state_map_t();
		AbstractStates[&F] = AbsState;
	}

	b_state_map_t *BlockAbsState = BlockEndAbstractStates[&F];
	if (BlockAbsState == NULL) {
		BlockAbsState = new b_state_map_t();
		BlockEndAbstractStates[&F] = BlockAbsState;
	}

	DenseMap<BasicBlock *, addr_set_t *> *ArraySets = UnflushedArrays[&F];
	if (ArraySets == NULL) {
		ArraySets = new DenseMap<BasicBlock *, addr_set_t *>();
		UnflushedArrays[&F] = ArraySets;
	}

#ifdef INTERPROCEDURAL
	CurrentContext = Context;
	FunctionArguments.clear();
	unsigned i = 0;
	for (Function::arg_iterator it = F.arg_begin(); it != F.arg_end(); it++) {
		FunctionArguments[&*it] = i++;
	}

	// Check if two or more parameters are escaped and dirty
	hasTwoEscapedDirtyParams = false;
	bool has_escaped_dirty_objs = false;
	for (unsigned i = 0; i < F.arg_size(); i++) {
		ParamState &PS = Context->getState(i);
		if (PS.isEscaped() && PS.isDirty()) {
			if (has_escaped_dirty_objs) {
				hasTwoEscapedDirtyParams = true;
				break;
			}

			has_escaped_dirty_objs = true;
		}
	}
#endif

	// Collect all return statements of Function F
	SmallPtrSet<Instruction *, 8> &RetSet = FunctionRetInstMap[&F];
	if (RetSet.empty()) {
		for (BasicBlock &BB : F) {
			if (isa<ReturnInst>(BB.getTerminator()))
				RetSet.insert(BB.getTerminator());
		}
	}

	StmtErrorSet = FunctionStmtErrorSets[&F];
	if (StmtErrorSet == NULL) {
		StmtErrorSet = new DenseSet<const Instruction *>();
		FunctionStmtErrorSets[&F] = StmtErrorSet;
	}

	FunctionMarksEscDirObj = false;
	hasError = false;

	// LLVM allows duplicate predecessors: https://stackoverflow.com/questions/65157239/llvmpredecessors-could-return-duplicate-basic-block-pointers
	DenseMap<const BasicBlock *, SmallPtrSet<BasicBlock *, 8> *> block_predecessors;
	DenseMap<const BasicBlock *, SmallPtrSet<BasicBlock *, 8> *> block_successors;
	DenseMap<const BasicBlock *, bool> visited_blocks;

	// Analyze F
	BlockWorklist.push_back(&F.getEntryBlock());
	while (!BlockWorklist.empty()) {
		BasicBlock *block = BlockWorklist.front();
		BlockWorklist.pop_front();

		bool predsNotAnalyzed = false;
		BasicBlock::iterator prev = block->begin();
		state_t *block_end_state = BlockAbsState->lookup(block);
		if (block_end_state == NULL) {
			block_end_state = new state_t();
			(*BlockAbsState)[block] = block_end_state;
		}

		for (BasicBlock::iterator it = block->begin(); it != block->end();) {
			state_t * state = AbsState->lookup(&*it);
			if (state == NULL) {
				state = new state_t();
				(*AbsState)[&*it] = state;
			}

			// Build state from predecessors' states
			if (it == block->begin()) {
				addr_set_t * array_addr_set = (*ArraySets)[block];
				if (array_addr_set == NULL) {
					array_addr_set = new addr_set_t();
					(*ArraySets)[block] = array_addr_set;
				}

				// First instruction; take the union of predecessors' states
				if (block == &F.getEntryBlock()) {
					// Entry block: has no precedessors
					// Prepare initial state based on parameters
#ifdef INTERPROCEDURAL
					state->clear();
					computeInitialState(state, F, Context);
#endif
				} else if (BasicBlock *pred = block->getUniquePredecessor()) {
					// Unique predecessor; copy the state of last instruction and unflushed arrays in the pred block
					state_t * prev_s = AbsState->lookup(pred->getTerminator());

					if (!visited_blocks[pred]) {
						BlockWorklist.push_back(pred);
						predsNotAnalyzed = true;
						break;
					} else {
						copyState(prev_s, state);
						copyArrayState((*ArraySets)[pred], array_addr_set);
					}
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

					bool visited_any = false;
					for (BasicBlock *pred : *pred_list) {
						if (visited_blocks[pred])
							visited_any = true;
						else
							BlockWorklist.push_back(pred);
					}

					if (!visited_any) {
						predsNotAnalyzed = true;
						break;
					}

					copyMergedState(AbsState, pred_list, state, visited_blocks);
					copyMergedArrayState(ArraySets, pred_list, array_addr_set, visited_blocks);
				}
			} else {
				// Copy the previous instruction's state
				copyState((*AbsState)[&*prev], state);
			}

			processInstruction(state, &*it);

			prev = it;
			it++;
		} // End of basic block instruction iteration

		if (predsNotAnalyzed)
			continue;

		// Copy block terminator's state and check if it has changed
		state_t * terminator_state = AbsState->lookup(block->getTerminator());
		bool block_state_changed = copyStateCheckDiff(terminator_state, block_end_state);

		// Push block successors to BlockWorklist if block state has changed
		// or it is the first time we visit this block
		if (block_state_changed || !visited_blocks[block]) {
			visited_blocks[block] = true;

			SmallPtrSet<BasicBlock *, 8> *succ_list = block_successors[block];
			if (succ_list == NULL) {
				succ_list = new SmallPtrSet<BasicBlock *, 8>();
				block_successors[block] = succ_list;
				for (BasicBlock *succ : successors(block)) {
					succ_list->insert(succ);
				}
			}

			for (BasicBlock *succ : *succ_list) {
				BlockWorklist.push_back(succ);
			}
		}
	}

#ifdef INTERPROCEDURAL
	// Only verify if output states have been changed
	bool state_changed = computeFinalState(AbsState, F, Context);
	if (state_changed) {
		// push callers with their contexts
		SmallDenseSet<std::pair<Function *, CallingContext *>> &Callers = FunctionCallerMap[&F];
		for (const std::pair<Function *, CallingContext *> &C : Callers) {
			if (UniqueFunctionSet.find(C) == UniqueFunctionSet.end()) {
				// Not found in FunctionWorklist
				Function *Function = C.first;
				CallingContext *CallerContext = new CallingContext(C.second);
				FunctionWorklist.emplace_back(Function, CallerContext);
				UniqueFunctionSet.insert(std::make_pair(Function, CallerContext));

				//errs() << "Function " << Function->getName() << " added to worklist in " << F.getName() << "\n";
			}
		}
	}
#endif
}

// DenseMap<Value *, ob_state_t *> state_t;
void PMRobustness::copyState(state_t * src, state_t * dst) {
	for (state_t::iterator it = src->begin(); it != src->end(); it++) {
		ob_state_t *object_state = dst->lookup(it->first);
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

void PMRobustness::copyArrayState(addr_set_t *src, addr_set_t *dst) {
	for (addr_set_t::iterator it = src->begin(); it != src->end(); it++) {
		ArrayInfo *info = (*dst)[it->first];
		if (info == NULL) {
			info = new ArrayInfo(it->second);
			(*dst)[it->first] = info;
		} else {
			info->copyFrom(it->second);
		}
	}
}

bool PMRobustness::copyStateCheckDiff(state_t * src, state_t * dst) {
	bool updated = false;

	// Mark each item in dst as `to delete`; they are unmarked if src contains them
	for (state_t::iterator it = dst->begin(); it != dst->end(); it++)
		it->second->markDelete();

	for (state_t::iterator it = src->begin(); it != src->end(); it++) {
		ob_state_t *object_state = dst->lookup(it->first);
		if (object_state == NULL) {
			// mark_delete is initialized to false;
			object_state = new ob_state_t(it->second);
			(*dst)[it->first] = object_state;
			updated = true;
		} else {
			updated |= object_state->copyFromCheckDiff(it->second);
			object_state->unmarkDelete();
		}
	}

	// Remove items not contained in src
	for (state_t::iterator it = dst->begin(); it != dst->end(); it++) {
		if (it->second->shouldDelete()) {
			dst->erase(it);
			updated = true;
		}
	}

	return updated;
}

void PMRobustness::copyMergedState(state_map_t *AbsState,
		SmallPtrSetImpl<BasicBlock *> * src_list, state_t * dst,
		DenseMap<const BasicBlock *, bool> &visited_blocks) {
	for (state_t::iterator it = dst->begin(); it != dst->end(); it++)
		it->second->setSize(0);

	for (BasicBlock *pred : *src_list) {
		state_t *s = AbsState->lookup(pred->getTerminator());
		if (!visited_blocks[pred]) {
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
					A->setSize(B->getSize());	// TODO: useless? 
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

void PMRobustness::copyMergedArrayState(DenseMap<BasicBlock *, addr_set_t *> *ArraySets,
		SmallPtrSetImpl<BasicBlock *> *src_list, addr_set_t *dst,
		DenseMap<const BasicBlock *, bool> &visited_blocks) {
//	for (addr_set_t::iterator it = dst->begin(); it != dst->end(); it++)
//		(*dst)[it->first]->setSize(0);

	for (BasicBlock *pred : *src_list) {
		addr_set_t *s = (*ArraySets)[pred];
		if (!visited_blocks[pred]) {
			continue;
		}

		//DenseMap<Value *, ArrayInfo *> state_t;
		for (addr_set_t::iterator it = s->begin(); it != s->end(); it++) {
			addr_set_t::iterator item = dst->find(it->first);
			if (item != dst->end()) {
				// (Loc, vector<VarState>) pair found
				ArrayInfo *A = item->second;
				ArrayInfo *B = it->second;
				A->mergeFrom(B);;
			} else {
				(*dst)[it->first] = new ArrayInfo(it->second);
			}
		}
	}
}

void PMRobustness::processInstruction(state_t * map, Instruction * I) {
	InstructionMarksEscDirObj = false;
	CallMarksEscDirObj = false;
	bool updated = false;
	bool check_error = false;
	bool non_dirty_escaped_before_call = true;

	if (I->isAtomic()) {
		updated |= processAtomic(map, I);
		check_error = true;
	} else if (isa<LoadInst>(I)) {
		updated |= processLoad(map, I);
		check_error = true;
	} else if (isa<StoreInst>(I)) {
		updated |= processStore(map, I);
		check_error = true;
	} else if (isa<PHINode>(I)) {
		updated |= processPHI(map, I);
	} else if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
		/* TODO: identify primitive functions that allocate memory
		CallBase *CallSite = cast<CallBase>(I);
		if (CallSite->getCalledFunction()) {
			if (MemAllocatingFunctions.count(llvm::demangle(
				CallSite->getCalledFunction()->getName().str()))) {}

		if (isParamAnnotationFunction(I))
			processParamAnnotationFunction(I);
		*/

		// Ignore Debug info Instrinsics
		if (isa<DbgInfoIntrinsic>(I)) {
			return;
		} else if (isa<MemIntrinsic>(I)) {
			updated |= processMemIntrinsic(map, I);
			check_error = true;
		} else if (ignoreFunction(I)) {
			return;
		} else if (isFlushWrapperFunction(I)) {
			updated |= true;
			processFlushWrapperFunction(map, I);
		} else if (isFlushParameterFunction(I)) {
			updated |= true;
			processFlushParameterFunction(map, I);
		} /*else if (isNTSWrapperFunction(I)) {
			//updated |= true;
			//processNTSWrapperFunction(map, I);
		}*/
		else {
			NVMOP op = whichNVMoperation(I);
			if (op == NVM_FENCE) {
				// TODO: fence operations
			} else if (op == NVM_CLWB || op == NVM_CLFLUSH) {
				// TODO: assembly flush operations. Are they used in real code?
				// updated |= processFlush(map, I, op);
			} else {
#ifdef INTERPROCEDURAL
				processCalls(map, I, non_dirty_escaped_before_call);
				updated |= true;
				check_error = true;
#endif
			}
		}
	}

	if (check_error) {
		// TODO: report bugs when a function marks an object as escaped, dirty
		checkEscapedObjError(map, I, non_dirty_escaped_before_call);
	}
/*
	if (updated) {
		errs() << "After " << *I << "\n";
		printMap(map);
	}
*/
}

bool PMRobustness::processAtomic(state_t * map, Instruction * I) {
	bool updated = false;
	//const DataLayout &DL = I->getModule()->getDataLayout();

	if (isa<StoreInst>(I)) {
		updated |= processStore(map, I);
	} else if (isa<LoadInst>(I)) {
		updated |= processLoad(map, I);
	} else if (AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(I)) {
		// Treat atomic RMWs as load + store
		updated |= processLoad(map, I);
		updated |= processStore(map, I);
		//errs() << "Atomic RMW processed\n";
	} else if (AtomicCmpXchgInst *CASI = dyn_cast<AtomicCmpXchgInst>(I)) {
#ifdef PMROBUST_DEBUG
		errs() << "CASI not implemented yet\n";
#endif
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

	if (dyn_cast<MemSetInst>(I)) {
		// TODO
		//errs() << "memset addr: " << M->getArgOperand(0) << "\t";
		//errs() << "const bit: " << *M->getArgOperand(1) << "\t";
		//errs() << "size: " << *M->getArgOperand(2) << "\n\n";
	} else if (dyn_cast<MemTransferInst>(I)) {
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

	DecomposedGEP DecompGEP;
	decomposeAddress(DecompGEP, Addr, DL);
	unsigned offset = DecompGEP.getOffsets();

	//printDecomposedGEP(DecompGEP);
	if (DecompGEP.isArray) {
		addr_set_t * unflushed_addr = getOrCreateUnflushedAddrSet(I->getFunction(), I->getParent());
		ArrayInfo * tmp = new ArrayInfo();
		tmp->copyGEP(&DecompGEP);
		(*unflushed_addr)[Addr] = tmp;

		// Use to record escaped/captured for base pointers of arrays
		getObjectState(map, DecompGEP.Base);
/*
		errs() << "Addr inserted: " << *Addr << "\n";
		errs() << *I << "\n  ";
		//I->getFunction()->dump();
		getPosition(I, IRB, true);
		errs() << "\n";
		*/
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
		// Rule 1: x.f = v => x.f becomes dirty
		if (isPMAddr(Addr, DL)) {
			unsigned TypeSize = getMemoryAccessSize(Addr, DL);
			ob_state_t *object_state = getObjectState(map, DecompGEP.Base, updated);
			updated |= object_state->setDirty(offset, TypeSize);

			// For reporting in-function error
			if (object_state->isEscaped()) {
				InstructionMarksEscDirObj = true;
				FunctionMarksEscDirObj = true;
			}
		}

		// Rule 2.1: *x = p => all fields of p escapes
		// TODO: Val(i.e. p) should be PM Addr
		if (Val && Val->getType()->isPointerTy() && isPMAddr(Val, DL)) {
			DecomposedGEP ValDecompGEP;
			decomposeAddress(ValDecompGEP, Val, DL);
			unsigned offset = ValDecompGEP.getOffsets();

			if (ValDecompGEP.isArray) {
				// e.g. struct { int A; int arr[10]; };
				// StructOffset is the beginning address of arr.
				// TODO: can we compute the size of the arr field?
				ob_state_t *object_state = getObjectState(map, ValDecompGEP.Base, updated);

				// Mark it as escaped
				updated |= object_state->setEscape();
			} else if (offset == UNKNOWNOFFSET || offset == VARIABLEOFFSET) {
				// TODO: start working here
				//assert(false && "Fix me");
			} else {
				ob_state_t *object_state = getObjectState(map, ValDecompGEP.Base, updated);

				// Note: if *x = &p->f, then *p is potentially reachabled; so mark it as escaped
				bool changed_to_escaped = object_state->setEscape();
				updated |= changed_to_escaped;

				// For reporting in-function error
				if (changed_to_escaped && object_state->isDirty()) {
					InstructionMarksEscDirObj = true;
					FunctionMarksEscDirObj = true;
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
		if (isPMAddr(Addr, DL)) {
			if (I->isAtomic() && isa<LoadInst>(I)) {
				// Mark the address as dirty to detect interthread robustness violation
				// For Atomic RMW, this is already done in processStore
				unsigned TypeSize = getMemoryAccessSize(Addr, DL);
				ob_state_t *object_state = getObjectState(map, DecompGEP.Base, updated);
				updated |= object_state->setDirty(offset, TypeSize);

				// For reporting in-function error
				if (object_state->isEscaped()) {
					InstructionMarksEscDirObj = true;
					FunctionMarksEscDirObj = true;
				}
			}
		}

		// Rule 2.2: p = *x => p escapes
		if (I->getType()->isPointerTy() && isPMAddr(I, DL)) {
			DecomposedGEP LIDecompGEP;
			decomposeAddress(LIDecompGEP, I, DL);
			unsigned offset = LIDecompGEP.getOffsets();

			if (LIDecompGEP.isArray) {
				assert(false && "Fix me");
			} else if (offset > 0) {
				assert(false && "fix me");
			}

			//unsigned TypeSize = getMemoryAccessSize(Addr, DL);
			ob_state_t *object_state = getObjectState(map, LIDecompGEP.Base, updated);
			bool changed_to_escaped = object_state->setEscape();
			updated |= changed_to_escaped;

			// For reporting in-function error
			if (changed_to_escaped && object_state->isDirty()) {
				InstructionMarksEscDirObj = true;
				FunctionMarksEscDirObj = true;
			}
		}
	}

	return updated;
}

bool PMRobustness::processPHI(state_t * map, Instruction * I) {
	bool updated = false;
	IRBuilder<> IRB(I);
	const DataLayout &DL = I->getModule()->getDataLayout();

	if (!I->getType()->isPointerTy())
		return false;

	ob_state_t *phi_state = NULL;
	bool first_state = true;
	for (User::op_iterator it = I->op_begin(); it != I->op_end(); it++) {
		Value *V = *it;

		DecomposedGEP DecompGEP;
		decomposeAddress(DecompGEP, V, DL);
		unsigned offset = DecompGEP.getOffsets();

		if (DecompGEP.isArray) {
			continue;
		} else if (offset == UNKNOWNOFFSET || offset == VARIABLEOFFSET) {
			continue;
		} else {
			ob_state_t *object_state = map->lookup(DecompGEP.Base);
			if (object_state == NULL)
				continue;

			if (first_state) {
				phi_state = getObjectState(map, I);
				phi_state->copyFrom(object_state);
				first_state = false;
			} else {
				// FIXME: only need to merge some field, not the entire object
				phi_state->mergeFrom(object_state);
			}
		}
	}

	return updated;
}

void PMRobustness::processFlushWrapperFunction(state_t * map, Instruction * I) {
	CallBase *callInst = cast<CallBase>(I);
	const DataLayout &DL = I->getModule()->getDataLayout();
	Function *callee = callInst->getCalledFunction();
	assert(callee);

	std::vector<StringRef> annotations;
	getAnnotatedParamaters("myflush", annotations, callee);

	Value *Addr = NULL;
	Value *FlushSize = NULL;
	for (unsigned i = 0; i < annotations.size(); i++) {
		StringRef &token = annotations[i];
		if (token == "addr") {
			Addr = callInst->getArgOperand(i);
		} else if (token == "size") {
			FlushSize = callInst->getArgOperand(i);
		} else if (token == "ignore") {
			// Ignore
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

/*
	errs() << "flush wrapper " << *Addr << "\n  for " << *FlushSize << "\n  ";
	IRBuilder<> IRB(I);
	getPosition(I, IRB, true);
	I->getFunction()->dump();
*/
	DecomposedGEP DecompGEP;
	decomposeAddress(DecompGEP, Addr, DL);
	unsigned offset = DecompGEP.getOffsets();

	if (DecompGEP.isArray) {
		//TODO: compare GEP address
		addr_set_t *AddrSet = getOrCreateUnflushedAddrSet(I->getFunction(), I->getParent());
		checkUnflushedAddress(I->getFunction(), AddrSet, Addr, DecompGEP);
	} else if (offset == UNKNOWNOFFSET || offset == VARIABLEOFFSET) {
		// TODO: treat it the same way as array
	} else {
		//unsigned TypeSize = getMemoryAccessSize(Addr, DL);
		ob_state_t *object_state = map->lookup(DecompGEP.Base);
		if (object_state == NULL) {
			// TODO: to be solve in interprocedural analysis
			// public method calls can modify the state of objects e.g. masstress.cpp:320
			errs() << "Flush an unknown address\n";
			//assert(false && "Flush an unknown address");
		} else {
			unsigned size;
			if (isa<ConstantInt>(FlushSize))
				size = cast<ConstantInt>(FlushSize)->getZExtValue();
			else {
				// FIXME: flush size can be a variable, such as `sizeof(leafvalue) + len`
				size = object_state->getSize();
			}

			//errs() << "flush " << *DecompGEP.Base << " from " << offset << " to " << size << "\n";
			if (FlushOp == NVM_CLFLUSH)
				object_state->setFlush(offset, size, true);
			else if (FlushOp == NVM_CLWB)
				object_state->setClwb(offset, size, true);
		}
	}
}

void PMRobustness::processFlushParameterFunction(state_t * map, Instruction * I) {
	CallBase *callInst = cast<CallBase>(I);
	const DataLayout &DL = I->getModule()->getDataLayout();
	Function *callee = callInst->getCalledFunction();
	assert(callee);

	std::vector<StringRef> annotations;
	getAnnotatedParamaters("flush_parameter", annotations, callee);

	Value *Addr = NULL;
	for (unsigned i = 0; i < annotations.size(); i++) {
		StringRef &token = annotations[i];
		if (token == "addr") {
			Addr = callInst->getArgOperand(i);
		} else if (token == "ignore") {
			// Ignore
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

/*
	errs() << "flush wrapper " << *Addr << "\n";
	IRBuilder<> IRB(I);
	getPosition(I, IRB, true);
	I->getFunction()->dump();
*/
	DecomposedGEP DecompGEP;
	decomposeAddress(DecompGEP, Addr, DL);
	unsigned offset = DecompGEP.getOffsets();

	if (DecompGEP.isArray) {
		assert(false);
	} else if (offset == UNKNOWNOFFSET || offset == VARIABLEOFFSET) {
		// TODO: treat it the same way as array
	} else {
		//unsigned TypeSize = getMemoryAccessSize(Addr, DL);
		ob_state_t *object_state = map->lookup(DecompGEP.Base);
		if (object_state == NULL) {
			// TODO: to be solve in interprocedural analysis
			// public method calls can modify the state of objects e.g. masstress.cpp:320
			errs() << "Flush an unknown address\n";
			//assert(false && "Flush an unknown address");
		} else {
			unsigned size = object_state->getSize();
			if (FlushOp == NVM_CLFLUSH)
				object_state->setFlush(offset, size, true);
			else if (FlushOp == NVM_CLWB)
				object_state->setClwb(offset, size, true);
		}
	}
}


// Not using this function for now
void PMRobustness::processNTSWrapperFunction(state_t * map, Instruction * I) {
	const DataLayout &DL = I->getModule()->getDataLayout();
	CallBase *callInst = cast<CallBase>(I);
	Function *callee = callInst->getCalledFunction();
	assert(callee);

	std::vector<StringRef> annotations;
	getAnnotatedParamaters("nts", annotations, callee);

	Value *Addr = NULL;
	Value *V = NULL;
	for (unsigned i = 0; i < annotations.size(); i++) {
		StringRef &token = annotations[i];
		if (token == "addr") {
			Addr = callInst->getArgOperand(i);
		} else if (token == "value") {
			// Ignore
			V = callInst->getArgOperand(i);
		} else if (token == "ignore") {
			// Ignore
		} else {
			assert(false && "bad annotation");
		}
	}

/*
	errs() << "Non-Temporal Store wrapper " << *Addr << "\n for " << *V << "\n  ";
	IRBuilder<> IRB(I);
	getPosition(I, IRB, true);
	//I->getFunction()->dump();
*/

	DecomposedGEP DecompGEP;
	decomposeAddress(DecompGEP, Addr, DL);
	unsigned offset = DecompGEP.getOffsets();

	if (DecompGEP.isArray) {
		// FIXME: N4.cpp:28
		// assert(false);
	} else if (offset == UNKNOWNOFFSET || offset == VARIABLEOFFSET) {
		// TODO: treat it the same way as array
		assert(false);
	} else {
		// Size is likely to be either 32, 64, or 128 bits
		unsigned TypeSize = getMemoryAccessSize(Addr, DL);
		ob_state_t *object_state = map->lookup(DecompGEP.Base);
		if (object_state == NULL) {
			object_state = new ob_state_t();
			(*map)[DecompGEP.Base] = object_state;
		}

		object_state->setClwb(offset, TypeSize);
		//(object_state of V)->setEscape(); //TODO & FIXME: if V is a reference, it will escape
	}
}

void PMRobustness::processCalls(state_t *map, Instruction *I, bool &non_dirty_escaped_before) {
	CallBase *CB = cast<CallBase>(I);
	if (CallInst *CI = dyn_cast<CallInst>(CB)) {
		if (CI->isInlineAsm())
			return;
	}

	Function *F = CB->getCalledFunction();
	if (F == NULL) {
		// TODO: why this happens?
		//assert(false);
		return;
	}

	if (F->isVarArg()) {
#ifdef PMROBUST_DEBUG
		errs() << "Cannot handle variable argument functions for " << F->getName() << "\n";
#endif
		return;
	} else if (F->isDeclaration()) {
		// TODO: think about how to approximate calls to functions with no body
#ifdef PMROBUST_DEBUG
		errs() << "Cannot handle functions with no function body: " << F->getName() << "\n";
#endif
		return;
	}

	// Update FunctionCallerMap
	SmallDenseSet<std::pair<Function *, CallingContext *>> &Callers = FunctionCallerMap[F];
	if (Callers.find(std::make_pair(CB->getCaller(), CurrentContext)) == Callers.end()) {
		// If Caller w/ Context is not found
		CallingContext *ContextCopy = new CallingContext(CurrentContext);
		Callers.insert(std::make_pair(CB->getCaller(), ContextCopy));
	}

	CallingContext *context = computeContext(map, I);
	lookupFunctionResult(map, CB, context, non_dirty_escaped_before);
}

void PMRobustness::getAnnotatedParamaters(std::string attr, std::vector<StringRef> &annotations, Function *callee) {
	StringRef AttrValue = callee->getFnAttribute(attr).getValueAsString();
	std::pair<StringRef, StringRef> Split = AttrValue.split("|");
	annotations.push_back(Split.first);

	while (!Split.second.empty()) {
		Split = Split.second.split("|");
		annotations.push_back(Split.first);
	}

	assert(callee->arg_size() == annotations.size() &&
		"annotations should match the number of paramaters");
}

/*
bool PMRobustness::processParamAnnotationFunction(Instruction * I) {
	CallInst *callInst = cast<CallInst>(I);
	const DataLayout &DL = I->getModule()->getDataLayout();

	Value *str = callInst->getArgOperand(0)->stripPointerCasts();
	Constant *c = cast<GlobalVariable>(str)->getInitializer();
	errs() << cast<ConstantDataArray>(c)->getAsCString() << "\n";

	return false;
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
	if (!OrigTy->isSized()) {
		if (OrigTy->isFunctionTy())
			return -1;
		else
			assert("false && OrigTy is not sized");
	}
	unsigned TypeSize = DL.getTypeStoreSizeInBits(OrigTy);

	return TypeSize / 8;
}


void PMRobustness::getAnalysisUsage(AnalysisUsage &AU) const {
	AU.addRequiredTransitive<AAResultsWrapperPass>();
	AU.addRequiredTransitive<MemoryDependenceWrapperPass>();
	//AU.addRequired<AAResultsWrapperPass>();
	//AU.addRequired<AndersenAAWrapperPass>();
	AU.setPreservesAll();
}

bool PMRobustness::skipFunction(Function &F) {
	std::vector<StringRef> nameList = { "myflush", "flush_parameter", "ignore", "suppress" };
	for (StringRef &s : nameList) {
		if (F.hasFnAttribute(s)) {
			errs() << "Function " << F.getName() << " is ignored\n";
			return true;
		}
	}

	return false;
}

bool PMRobustness::isPMAddr(const Value * Addr, const DataLayout &DL) {
	const Value *Origin = GetUnderlyingObject(Addr, DL);
	for (auto &u : Origin->uses()) {
		if (isa<AllocaInst>(u)) {
			//errs() << *Addr << " is non pmem\n";
			return false;
		}
	}

	//errs() << "*** " << *Addr << " is PMEM\n";
	return true;
}

/** Simple may-analysis for checking if an address is in the heap
 *  TODO: may need more sophisticated checks
 **/
 /*
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

	// TODO: if pointer comes from function parameters; check the caller
	// Attach metadata to each uses of function parameters
	//if (I->getMetadata(FUNC_PARAM_USE)) {}

	// Address may be in the heap. We don't know for sure.
	return true;
}*/

ob_state_t * PMRobustness::getObjectState(state_t *map, const Value *Addr, bool &updated) {
	ob_state_t *object_state = map->lookup(Addr);
	if (object_state == NULL) {
		object_state = new ob_state_t();
		(*map)[Addr] = object_state;
		updated |= true;
	}

	return object_state;
}

ob_state_t * PMRobustness::getObjectState(state_t *map, const Value *Addr) {
	ob_state_t *object_state = map->lookup(Addr);
	if (object_state == NULL) {
		object_state = new ob_state_t();
		(*map)[Addr] = object_state;
	}

	return object_state;
}

void PMRobustness::checkEndError(state_map_t *AbsState, Function &F) {
	SmallPtrSet<Instruction *, 8> &RetSet = FunctionRetInstMap[&F];
	DenseMap<BasicBlock *, addr_set_t *> *ArraySets = UnflushedArrays[&F];
	DenseSet<const Value *> * ErrorSet = FunctionEndErrorSets[&F];
	if (ErrorSet == NULL) {
		ErrorSet = new DenseSet<const Value *>();
		FunctionEndErrorSets[&F] = ErrorSet;
	}

	SmallPtrSet<Value *, 8> RetValSet;
	for (Instruction *I : RetSet) {
		Value *Ret = cast<ReturnInst>(I)->getReturnValue();
		RetValSet.insert(Ret);
	}

	for (Instruction *I : RetSet) {
		// 1) Check objects other than parameters/return values that are dirty and escaped
		// Get the state at each return statement
		state_t *state = AbsState->lookup(I);
		for (state_t::iterator it = state->begin(); it != state->end(); it++) {
			if (FunctionArguments.find(it->first) != FunctionArguments.end())
				continue;
			else if (RetValSet.find(it->first) != RetValSet.end())
				continue;

			Value *Ret = cast<ReturnInst>(I)->getReturnValue();
			if (Ret == it->first)
				continue;

			ob_state_t *object_state = it->second;
			if (object_state->isDirty() && object_state->isEscaped()) {
				if (ErrorSet->find(it->first) == ErrorSet->end()) {
					hasError = true;
					ErrorSet->insert(it->first);

					errs() << "Error!!!!!!! at return statement: ";
					IRBuilder<> IRB(I);
					Value *pos = NULL;
					if (isa<Instruction>(it->first) && !isa<PHINode>(it->first)) {
						const Instruction *inst = cast<Instruction>(it->first);
						pos = getPosition(inst, IRB, true);

						if (pos == NULL)
							getPosition(I, IRB, true);
					} else {
						getPosition(I, IRB, true);
					}

					errs() << "@@ Instruction " << *it->first << "\n";
				}
			}
		}

		// 2) Check unflushed arrays at the exit blocks
		BasicBlock *exit_block = I->getParent();
		addr_set_t *array_addr_set = (*ArraySets)[exit_block];
		for (addr_set_t::iterator it = array_addr_set->begin(); it != array_addr_set->end(); it++) {
			ob_state_t *object_state = state->lookup(it->second->Base);
			assert(object_state);

			if (object_state->isEscaped()) {
				if (ErrorSet->find(it->first) == ErrorSet->end()) {
					hasError = true;
					ErrorSet->insert(it->first);

					errs() << "Error: Unflushed array address: ";
					IRBuilder<> IRB(I);
					if (isa<Instruction>(it->first)) {
						const Instruction *inst = cast<Instruction>(it->first);
						getPosition(inst, IRB, true);
					} else
						getPosition(I, IRB, true);

					errs() << "@@ Instruction " << *it->first << "\n";
				}
			}
		}
	}
}

void PMRobustness::checkEscapedObjError(state_t *map, Instruction *I, bool non_dirty_escaped_before) {
	unsigned escaped_dirty_objs_count = 0;
	bool check_and_report = false;
	IRBuilder<> IRB(I);

	if (!InstructionMarksEscDirObj)
		return;

	for (state_t::iterator it = map->begin(); it != map->end(); it++) {
		ob_state_t *object_state = it->second;
		if (object_state->isEscaped() && object_state->isDirty()) {
			escaped_dirty_objs_count++;

			// There is already one or more escaped dirty objects
			if (escaped_dirty_objs_count == 2) {
				check_and_report = true;
				break;
			}

			// TODO: check if two fields in an object is dirty
		}
	}

	if (escaped_dirty_objs_count == 1 && CallMarksEscDirObj) {
		// If escaped_dirty_objs_count was 0 before processCall updated states,
		// then it is not a bug
		// Some of the cases where two fields of an object are dirty are covered here
		if (!non_dirty_escaped_before)
			check_and_report = true;
	}

	if (check_and_report && StmtErrorSet->find(I) == StmtErrorSet->end()) {
		hasError = true;
		StmtErrorSet->insert(I);
		errs() << "Reporting errors for function: " << I->getFunction()->getName() << "\n";
		errs() << "Error: More than two objects are escaped and dirty at: ";
		getPosition(I, IRB, true);
		errs() << "@@ Instruction " << *I << "\n";
		if (hasTwoEscapedDirtyParams) {
			errs() << "Two Parameters are already escaped dirty, this error may not be real\n";
		}
	}
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
		if (isFlushWrapperFunction(&*I)) {
			CallBase *callInst = cast<CallBase>(&*I);
			Function *callee = callInst->getCalledFunction();
			op = analyzeFlushType(*callee);
		} else {
			op = whichNVMoperation(&*I);
		}

		if (op == NVM_CLWB) {
			F.addFnAttr("flush_type", "clflushopt");
			break;
		} else if (op == NVM_CLFLUSH) {
			F.addFnAttr("flush_type", "clflush");
			break;
		} else if (op == NVM_FENCE) {
			continue;
		}
	}

	return op;
}

/*
bool PMRobustness::isParamAnnotationFunction(Instruction *I) {
	if (CallInst *CI = dyn_cast<CallInst>(I)) {
		if (Function *callee = CI->getCalledFunction()) {
			if (callee->hasFnAttribute("magic_function"))
				return true;
		}
	}

	return false;
}*/

bool PMRobustness::isFlushWrapperFunction(Instruction *I) {
	if (CallBase *CB = dyn_cast<CallBase>(I)) {
		if (Function *callee = CB->getCalledFunction()) {
			if (callee->hasFnAttribute("myflush"))
				return true;
		}
	}

	return false;
}

bool PMRobustness::isFlushParameterFunction(Instruction *I) {
	if (CallBase *CB = dyn_cast<CallBase>(I)) {
		if (Function *callee = CB->getCalledFunction()) {
			if (callee->hasFnAttribute("flush_parameter"))
				return true;
		}
	}

	return false;
}

bool PMRobustness::ignoreFunction(Instruction *I) {
	if (CallBase *CB = dyn_cast<CallBase>(I)) {
		if (Function *callee = CB->getCalledFunction()) {
			if (callee->hasFnAttribute("ignore") || callee->hasFnAttribute("suppress"))
				return true;
		}
	}

	return false;
}

/*
// Non-temporal store
bool PMRobustness::isNTSWrapperFunction(Instruction *I) {
	if (CallBase *CB = dyn_cast<CallBase>(I)) {
		if (Function *callee = CB->getCalledFunction()) {
			if (callee->hasFnAttribute("nts"))
				return true;
		}
	}

	return false;
}
*/

addr_set_t * PMRobustness::getOrCreateUnflushedAddrSet(Function *F, BasicBlock *B) {
	DenseMap<BasicBlock *, addr_set_t *> * ArraySets = UnflushedArrays[F];
	if (ArraySets == NULL) {
		ArraySets = new DenseMap<BasicBlock *, addr_set_t *>();
		UnflushedArrays[F] = ArraySets;
	}

	addr_set_t * set = (*ArraySets)[B];
	if (set == NULL) {
		set = new addr_set_t();
		(*ArraySets)[B] = set;
	}

	return set;
}

bool PMRobustness::checkUnflushedAddress(Function *F, addr_set_t * AddrSet, Value * Addr, DecomposedGEP &DecompGEP) {
	addr_set_t::iterator it = AddrSet->find(Addr);
	if (it != AddrSet->end()) {
		AddrSet->erase(it);
		errs() << "Addr is flushed: " << *Addr << "\n";
		return true;
	}

	MemoryDependenceResults &MDA = getAnalysis<MemoryDependenceWrapperPass>(*F).getMemDep();

	Value *V = Addr;
	Value *VBase = NULL;
	while (true) {
		Operator *Op = dyn_cast<Operator>(V);
		if (Op->getOpcode() == Instruction::BitCast ||
			Op->getOpcode() == Instruction::AddrSpaceCast) {
			V = Op->getOperand(0);
			continue;
		}

		GEPOperator *GEPOp = dyn_cast<GEPOperator>(Op);
		if (!GEPOp) {
			errs() << "not GEP\n";
			break;
		}

		// Don't attempt to analyze GEPs over unsized objects.
		if (!GEPOp->getSourceElementType()->isSized()) {
			assert(false && "Cannot compare unsized objects");
			return false;
		}

		VBase = GEPOp->getOperand(0);
		if (Instruction *Load = dyn_cast<Instruction>(VBase)) {
			MemDepResult Res = MDA.getDependency(cast<Instruction>(Load));
			if (!Res.isNonLocal()) {
				if (Res.getInst()) {
					errs() << "\nDepends on: " << *Res.getInst() << "\n";
					if (isa<StoreInst>(Res.getInst())) {
						VBase = Res.getInst()->getOperand(0);
					}
				}
			} else {
				errs() << "\nNonLocal Inst: " << "\n";
				//Load->getFunction()->dump();
			}
		} else {
			errs() << "GEP Base not a load: " << *VBase << "\n";
		}

		break;
	}

	errs() << "V: " << *V << "\n";
	errs() << "VBase: " << *VBase << "\n";
	// Iterate over the set of unflushed addresses and compare the decomposed GEP
	for (addr_set_t::iterator it = AddrSet->begin(); it != AddrSet->end(); it++) {
		Value *targetAddr = it->first;
		Value *targetBase = NULL;
		DecomposedGEP *targetAddrDecom = it->second;

		//printDecomposedGEP(*targetAddrDecom);

		// Compare offsets and variable indices in decomposed GEP
		if (compareDecomposedGEP(*targetAddrDecom, DecompGEP)) {
			AddrSet->erase(it);
			errs() << "match 1\n";
			return true;
		}

		while (true) {
			Operator *Op = dyn_cast<Operator>(targetAddr);
			if (Op->getOpcode() == Instruction::BitCast ||
					Op->getOpcode() == Instruction::AddrSpaceCast) {
				targetAddr = Op->getOperand(0);
				continue;
			}

			GEPOperator *GEPOp = dyn_cast<GEPOperator>(Op);
			if (!GEPOp) {
				errs() << "target not GEP\n";
				break;
			}

			// Don't attempt to analyze GEPs over unsized objects.
			if (!GEPOp->getSourceElementType()->isSized()) {
				continue;
			}

			targetBase = GEPOp->getOperand(0);
			break;
		}

		errs() << "targetAddr: " << *targetAddr << "\n";
		if (targetBase)
			errs() << "targetBase: " << *targetBase << "\n";
		else
			errs() << "targetBase: NULL\n";

		if (VBase == NULL && targetBase == NULL) {
			// Both addresses are not GEP instructions
			if (V == targetAddr) {
				AddrSet->erase(it);
				errs() << "match 2\n";
				return true;
			}
		} else if (VBase != NULL && targetBase != NULL) {
			// Compare indices in GEP instructions
			if (VBase == targetBase) {
				GEPOperator *VGEPOp = dyn_cast<GEPOperator>(V);
				GEPOperator *targetGEPOp = dyn_cast<GEPOperator>(targetAddr);

				if (VGEPOp->getNumIndices() != targetGEPOp->getNumIndices())
					continue;

				bool diff = false;
				for (unsigned i = 1; i < VGEPOp->getNumIndices(); i++) {
					if (VGEPOp->getOperand(i) != targetGEPOp->getOperand(i)) {
						diff = true;
						break;
					}
				}

				if (!diff) {
					errs() << "match 3\n";
					AddrSet->erase(it);
					return true;
				}
			}
		}
	}

	errs() << "flush an array address not seen before\n";
	//assert(false && "flush an array address not seen before");
	return false;
}

bool PMRobustness::compareDecomposedGEP(DecomposedGEP &GEP1, DecomposedGEP &GEP2) {
	if (GEP1.OtherOffset != GEP2.OtherOffset ||
		GEP1.StructOffset != GEP2.StructOffset) {
		return false;
	}

	if (GEP1.VarIndices.size() != GEP2.VarIndices.size())
		return false;

	for (unsigned i = 0; i < GEP1.VarIndices.size(); i++) {
		if (GEP1.VarIndices[i] != GEP2.VarIndices[i]) {
			return false;
		}
	}

	return true;
}

CallingContext * PMRobustness::computeContext(state_t *map, Instruction *I) {
	CallBase *CB = cast<CallBase>(I);
	const DataLayout &DL = I->getModule()->getDataLayout();
	//if (CB->arg_size() == 0)
	//	return NULL;

	CallingContext *context = new CallingContext();

	//Function *F = CB->getCalledFunction();
	//F->dump();

	unsigned i = 0;
	for (User::op_iterator it = CB->arg_begin(); it != CB->arg_end(); it++) {
		Value *op = *it;

		if (op->getType()->isPointerTy() && isPMAddr(op, DL)) {
			DecomposedGEP DecompGEP;
			decomposeAddress(DecompGEP, op, DL);
			unsigned offset = DecompGEP.getOffsets();

			if (DecompGEP.isArray || offset == UNKNOWNOFFSET || offset == VARIABLEOFFSET) {
				// TODO: We have information about arrays escaping or not.
				// Dirty array slots are stored in UnflushedArrays

				// FIXME
				context->addAbsInput(ParamStateType::TOP);
			} else {
				ob_state_t *object_state = map->lookup(DecompGEP.Base);
				ParamStateType absState = ParamStateType::TOP;
				if (object_state != NULL) {
					unsigned TypeSize = getFieldSize(op, DL);
					//errs() << "Base: " << *DecompGEP.Base << " of instruction " << *I <<"\n";
					//errs() << "offset: " << offset << "; field size: " << TypeSize << "\n";
					//errs() << "checking state of op: " << *op << "\n";

					if (object_state->getSize() == 0 && FunctionArguments.find(op) != FunctionArguments.end()) {
						// The parent function's parameter is passed to call site without any modification
						absState = CurrentContext->getStateType(FunctionArguments.lookup(op));
					} else {
						absState = object_state->checkState(offset, TypeSize);
					}
				}

				context->addAbsInput(absState);
			}
		} else {
			context->addAbsInput(ParamStateType::NON_PMEM);
		}

		i++;
	}

	return context;
}

void PMRobustness::lookupFunctionResult(state_t *map, CallBase *CB, CallingContext *Context, bool &non_dirty_escaped_before) {
	Function *F = CB->getCalledFunction();
	FunctionSummary *FS = FunctionSummaries[F];
	const DataLayout &DL = CB->getModule()->getDataLayout();
	bool use_higher_results = false;

	//errs() << "lookup results for function: " << F->getName() << "\n";
	// Function has not been analyzed before
	if (FS == NULL) {
		if (UniqueFunctionSet.find(std::make_pair(F, Context)) == UniqueFunctionSet.end()) {
			FunctionWorklist.emplace_back(F, Context);
			UniqueFunctionSet.insert(std::make_pair(F, Context));
			//errs() << "Function " << F->getName() << " added to worklist in " << CB->getFunction()->getName();
		} else {
			delete Context;
		}

		// Mark all parameters as TOP (clean and captured)
		makeParametersTOP(map, CB);
		return;
	}

	OutputState *out_state = FS->getResult(Context);
	// Function has not been analyzed with the context
	// TODO: If there are multiple contexts higher than the current context, use merged results
	if (out_state == NULL) {
		if (UniqueFunctionSet.find(std::make_pair(F, Context)) == UniqueFunctionSet.end()) {
			FunctionWorklist.emplace_back(F, Context);
			UniqueFunctionSet.insert(std::make_pair(F, Context));
			//errs() << "Function " << F->getName() << " added to worklist in " << CB->getFunction()->getName() << "\n";
		} else {
			delete Context;
		}

		out_state = FS->getLeastUpperResult(Context);
		if (out_state == NULL) {
			// No least upper context exists
			// Mark all parameters as TOP (clean and captured)
			makeParametersTOP(map, CB);
			return;
		} else
			use_higher_results = true;
	}

	//errs() << "Function cache found\n";

	// For reporting in-function error
	if (out_state->marksEscDirObj) {
		InstructionMarksEscDirObj = true;
		FunctionMarksEscDirObj = true;
	}

	if (!use_higher_results && out_state->marksEscDirObjConditional) {
		CallMarksEscDirObj = true;
	}

	if (CallMarksEscDirObj) {
		for (state_t::iterator it = map->begin(); it != map->end(); it++) {
			ob_state_t *object_state = it->second;
			if (object_state->isEscaped() && object_state->isDirty()) {
				non_dirty_escaped_before = false;
				break;
			}
		}

	}

	// Use cached result to modify parameter states
	unsigned i = 0;
	for (User::op_iterator it = CB->arg_begin(); it != CB->arg_end(); it++) {
		Value *op = *it;

		if (Context->getStateType(i) == ParamStateType::NON_PMEM) {
			// Ignore NON_PMEM input parameters
			assert(out_state->getStateType(i) == ParamStateType::NON_PMEM);
			i++;
			continue;
		} else if (out_state->getStateType(i) == ParamStateType::NON_PMEM) {
			// It has happened that when the input is top, the output is NON_PMEM
			i++;
			continue;
		}

		DecomposedGEP DecompGEP;
		decomposeAddress(DecompGEP, op, DL);
		unsigned offset = DecompGEP.getOffsets();

		if (DecompGEP.isArray || offset == UNKNOWNOFFSET || offset == VARIABLEOFFSET) {
			// TODO: We have information about arrays escaping or not.
			// Dirty array slots are stored in UnflushedArrays
			ob_state_t *object_state = getObjectState(map, DecompGEP.Base);
			ParamStateType param_state = out_state->getStateType(i);

			addr_set_t *unflushed_addr = getOrCreateUnflushedAddrSet(CB->getFunction(), CB->getParent());
			ArrayInfo *info = unflushed_addr->lookup(op);
			ArrayInfo *tmp = new ArrayInfo();
			tmp->copyGEP(&DecompGEP);

			if (param_state == ParamStateType::DIRTY_CAPTURED) {
				if (info == NULL)
					(*unflushed_addr)[op] = tmp;
			} else if (param_state == ParamStateType::DIRTY_ESCAPED) {
				if (info == NULL)
					(*unflushed_addr)[op] = tmp;
				object_state->setEscape();
			} else if (param_state == ParamStateType::CLWB_CAPTURED) {
				// FIXME
				if (info == NULL)
					(*unflushed_addr)[op] = tmp;
			} else if (param_state == ParamStateType::CLWB_ESCAPED) {
				// FIXME
				if (info == NULL)
					(*unflushed_addr)[op] = tmp;
				object_state->setEscape();
			} else if (param_state == ParamStateType::CLEAN_CAPTURED) {
				if (info != NULL)
					checkUnflushedAddress(CB->getFunction(), unflushed_addr, op, DecompGEP);
				delete tmp;
			} else if (param_state == ParamStateType::CLEAN_ESCAPED) {
				if (info != NULL)
					checkUnflushedAddress(CB->getFunction(), unflushed_addr, op, DecompGEP);
				object_state->setEscape();
				delete tmp;
			} else if (param_state == ParamStateType::TOP) {
				checkUnflushedAddress(CB->getFunction(), unflushed_addr, op, DecompGEP);
				object_state->setCaptured();
				delete tmp;
			} else {
				assert(false && "other cases");
			}
		} else {
			unsigned TypeSize = getFieldSize(op, DL);
			if (TypeSize == (unsigned)-1) {
				i++;
				continue;
			}

			ob_state_t *object_state = getObjectState(map, DecompGEP.Base);
			ParamStateType param_state = out_state->getStateType(i);
			if (param_state == ParamStateType::DIRTY_CAPTURED) {
				// Approximate dirty
				if (Context->getState(i).isClean()) {
					// Note: We don't recapture escaped objects
					// If input state is clean, then use DirtyBytesInfo to get dirty bytes
					DirtyBytesInfo *info = out_state->getDirtyBytesInfo(i);
					std::vector<std::pair<int, int>> *lst = info->getDirtyBytes();

					assert(offset != UNKNOWNOFFSET && offset != VARIABLEOFFSET);
					for (unsigned i = 0; i < lst->size(); i++) {
						std::pair<int, int> &elem = (*lst)[i];
						object_state->setDirty(offset + elem.first, elem.second - elem.first);
					}
				} else if (!out_state->isUntouched(i)) {
					object_state->setDirty(offset, TypeSize);
				}
			} else if (param_state == ParamStateType::DIRTY_ESCAPED) {
				// Approximate dirty
				if (Context->getState(i).isClean()) {
					// Note: We don't recapture escaped objects
					// If input state is clean, then use DirtyBytesInfo to get dirty bytes
					DirtyBytesInfo *info = out_state->getDirtyBytesInfo(i);
					std::vector<std::pair<int, int>> *lst = info->getDirtyBytes();

					assert(offset != UNKNOWNOFFSET && offset != VARIABLEOFFSET);
					for (unsigned i = 0; i < lst->size(); i++) {
						std::pair<int, int> &elem = (*lst)[i];
						object_state->setDirty(offset + elem.first, elem.second - elem.first);
					}
				} else if (!out_state->isUntouched(i)) {
					object_state->setDirty(offset, TypeSize);
				}

				object_state->setEscape();
			} else if (param_state == ParamStateType::CLWB_CAPTURED) {
				object_state->setClwb(offset, TypeSize);
			} else if (param_state == ParamStateType::CLWB_ESCAPED) {
				object_state->setClwb(offset, TypeSize);
				object_state->setEscape();
			} else if (param_state == ParamStateType::CLEAN_CAPTURED) {
				object_state->setFlush(offset, TypeSize);
			} else if (param_state == ParamStateType::CLEAN_ESCAPED) {
				object_state->setFlush(offset, TypeSize);
				object_state->setEscape();
			} else if (param_state == ParamStateType::TOP) {
				object_state->setFlush(offset, TypeSize);
				object_state->setCaptured();
			} else {
				assert(false && "other cases");
			}
		}

		i++;
	}

	if (out_state->hasRetVal) {
		modifyReturnState(map, CB, out_state);
	}
}

// Get initial states of parameters from Context
void PMRobustness::computeInitialState(state_t *map, Function &F, CallingContext *Context) {
	assert(F.arg_size() == Context->AbstractInputState.size());
	const DataLayout &DL = F.getParent()->getDataLayout();

	unsigned i = 0;
	for (Function::arg_iterator it = F.arg_begin(); it != F.arg_end(); it++) {
		ParamStateType PS = Context->getStateType(i);
		i++;

		Argument *Arg = &*it;
		ob_state_t *object_state = getObjectState(map, Arg);

		if (PS == ParamStateType::NON_PMEM) {
			object_state->setNonPmem();
			continue;
		}

		unsigned TypeSize = getFieldSize(Arg, DL);
		if (TypeSize == (unsigned)-1) {
			continue;
		}

		if (PS == ParamStateType::DIRTY_CAPTURED) {
			// TODO: how to better approximate dirty
			object_state->setDirty(0, TypeSize);
		} else if (PS == ParamStateType::DIRTY_ESCAPED) {
			// TODO: how to better approximate dirty
			object_state->setDirty(0, TypeSize);
			object_state->setEscape();
		} else if (PS == ParamStateType::CLWB_CAPTURED) {
			object_state->setClwb(0, TypeSize);
		} else if (PS == ParamStateType::CLWB_ESCAPED) {
			object_state->setClwb(0, TypeSize);
			object_state->setEscape();
		} else if (PS == ParamStateType::CLEAN_CAPTURED) {
			object_state->setFlush(0, TypeSize);
		} else if (PS == ParamStateType::CLEAN_ESCAPED) {
			object_state->setFlush(0, TypeSize);
			object_state->setEscape();
		} else if (PS == ParamStateType::TOP) {
			// TOP: clean and captured
			object_state->setFlush(0, TypeSize);
		} else {
			// BOTTOM, etc.
			// Not sure what to do
		}
	}
}

// Get final states of parameters and return value from Function F
bool PMRobustness::computeFinalState(state_map_t *AbsState, Function &F, CallingContext *Context) {
	const DataLayout &DL = F.getParent()->getDataLayout();
	SmallPtrSet<Instruction *, 8> &RetSet = FunctionRetInstMap[&F];

	FunctionSummary *FS = FunctionSummaries[&F];
	if (FS == NULL) {
		FS = new FunctionSummary();
		FunctionSummaries[&F] = FS;
	}
	OutputState *Output = FS->getOrCreateResult(Context);

	state_t final_state;
	for (Instruction *I : RetSet) {
		// Get the state at each return statements
		state_t *s = AbsState->lookup(I);

		// Merge the state of each function paremeter. This is a conservative approximation.
		for (Function::arg_iterator it = F.arg_begin(); it != F.arg_end(); it++) {
			Argument *Arg = &*it;
			ob_state_t *A = final_state.lookup(Arg);
			ob_state_t *B = s->lookup(Arg);

			if (B == NULL)
				continue;

			if (A == NULL)
				final_state[Arg] = new ob_state_t(B);
			else {
				A->mergeFrom(B);
			}
		}
	}

	bool updated = false;
	unsigned i = 0;
	Output->AbstractOutputState.resize(F.arg_size());
	for (Function::arg_iterator it = F.arg_begin(); it != F.arg_end(); it++) {
		Argument *Arg = &*it;

		ob_state_t *object_state = final_state.lookup(Arg);
		if (object_state == NULL || object_state->isNonPmem()) {
			if (Output->getStateType(i) != ParamStateType::NON_PMEM) {
				Output->AbstractOutputState[i].setState(ParamStateType::NON_PMEM);
				updated = true;
			}

			i++;
			continue;
		}

		ParamStateType absState;
		if (object_state->getSize() == 0) {
			// No store to the parameter; TODO: how about loads?
			absState = Context->getStateType(i);
		} else {
			unsigned TypeSize = getFieldSize(Arg, DL);
			absState = object_state->checkState(0, TypeSize);

			// If the input state is clean, add dirty btyes to list
			ParamState &input_state = Context->getState(i);
			if (input_state.isClean()) {
				if (absState == ParamStateType::DIRTY_CAPTURED || absState == ParamStateType::DIRTY_ESCAPED) {
					DirtyBytesInfo *info = Output->getOrCreateDirtyBytesInfo(i);
					object_state->computeDirtyBytes(info);
				} // TODO: else if (CLWB_CAPTURED/CLWB_ESCAPED)
			}
		}

		if (Output->getStateType(i) != absState) {
			Output->AbstractOutputState[i].setState(absState);
			updated = true;
		}

		i++;
	}

	// Cache return Type
	Type *RetType = F.getReturnType();
	if (RetType->isVoidTy()) {
		Output->hasRetVal = false;
	} else if (RetType->isPointerTy()) {
		Output->hasRetVal = true;
		//Output->retVal = object_state->checkState();
		ParamState PS(ParamStateType::TOP);
		for (Instruction *I : RetSet) {
			state_t *s = AbsState->lookup(I);
			Value *Ret = cast<ReturnInst>(I)->getReturnValue();
			ob_state_t *RetState = s->lookup(Ret);

			if (RetState != NULL) {
				ParamStateType tmp_state = RetState->checkState();
				ParamState tmp = ParamState(tmp_state);
				if (tmp.isLowerThan(PS)) {
					//errs() << tmp.print() << " < " << PS.print() << "\n";
					PS.setState(tmp_state);
				}
			}
		}

		Output->retVal.setState(PS.get_state());
	} else {
		//errs() << "RetVal non PMEM\n";
		Output->hasRetVal = true;
		Output->retVal.setState(ParamStateType::NON_PMEM);
	}

	if (FunctionMarksEscDirObj) {
		Output->marksEscDirObj = true;

		// all parameters are not dirty escaped
		bool non_dirty_escaped = true;
		for (unsigned i = 0; i < Context->AbstractInputState.size(); i++) {
			ParamState &state = Context->getState(i);
			if (state.isDirtyEscaped()) {
				non_dirty_escaped = false;
				break;
			}
		}

		if (non_dirty_escaped) {
			Output->marksEscDirObjConditional = true;
		}
	}

	if (!Output->checkUntouched) {
		unsigned i = 0;
		for (Function::arg_iterator it = F.arg_begin(); it != F.arg_end(); it++) {
			Argument *Arg = &*it;
			if (Arg->getNumUses() == 0)
				Output->setUntouched(i);

			i++;
		}
	}

	// Free memory in temporarily allocated final_state object
	for (state_t::iterator it = final_state.begin(); it != final_state.end(); it++) {
		delete it->second;
	}

	checkEndError(AbsState, F);

	if (hasError) {
		F.dump();
		CurrentContext->dump();
	}

	return updated;
}

// Mark all parameters as TOP (clean and captured)
void PMRobustness::makeParametersTOP(state_t *map, CallBase *CB) {
	const DataLayout &DL = CB->getModule()->getDataLayout();
	for (User::op_iterator it = CB->arg_begin(); it != CB->arg_end(); it++) {
		Value *op = *it;

		if (op->getType()->isPointerTy() && isPMAddr(op, DL)) {
			DecomposedGEP DecompGEP;
			decomposeAddress(DecompGEP, op, DL);
			unsigned offset = DecompGEP.getOffsets();

			if (DecompGEP.isArray || offset == UNKNOWNOFFSET || offset == VARIABLEOFFSET) {
				// TODO: What do we do here?
			} else {
				ob_state_t *object_state = getObjectState(map, DecompGEP.Base);
				unsigned TypeSize = getFieldSize(op, DL);
				//errs() << "TypeSize: " << TypeSize << "\n";
				//errs() << *op << "\n";

				if (TypeSize == (unsigned)-1)
					object_state->setFlush(offset, TypeSize, true);
				else
					object_state->setFlush(offset, TypeSize);

				object_state->setCaptured();
			}
		} // Else case: op is NON_PMEM, so don't do anything
	}
}

void PMRobustness::modifyReturnState(state_t *map, CallBase *CB, OutputState *out_state) {
	const DataLayout &DL = CB->getModule()->getDataLayout();
	ParamStateType return_state = out_state->retVal.get_state();
	if (return_state == ParamStateType::NON_PMEM)
		return;

	DecomposedGEP DecompGEP;
	decomposeAddress(DecompGEP, CB, DL);
	unsigned offset = DecompGEP.getOffsets();

	if (DecompGEP.isArray || offset == UNKNOWNOFFSET || offset == VARIABLEOFFSET) {
		assert(false);
	} else {
		unsigned TypeSize = getFieldSize(CB, DL);
		if (TypeSize == (unsigned)-1) {
			errs() << "CB " << *CB << " has field size -1; so it is a function type\n";
			//errs() << *CB->getFunction() << "\n";
			//assert(false);
			return;
		}

		ob_state_t *object_state = getObjectState(map, DecompGEP.Base);

		if (return_state == ParamStateType::DIRTY_CAPTURED) {
			// Approximate dirty
			object_state->setDirty(offset, TypeSize);
		} else if (return_state == ParamStateType::DIRTY_ESCAPED) {
			// TODO: How to approximate dirty?
			//assert(false);
			object_state->setEscape();
			object_state->setDirty(offset, TypeSize);
		} else if (return_state == ParamStateType::CLWB_CAPTURED) {
			object_state->setClwb(offset, TypeSize);
		} else if (return_state == ParamStateType::CLWB_ESCAPED) {
			object_state->setClwb(offset, TypeSize);
			object_state->setEscape();
		} else if (return_state == ParamStateType::CLEAN_CAPTURED) {
			object_state->setFlush(offset, TypeSize);
		} else if (return_state == ParamStateType::CLEAN_ESCAPED) {
			object_state->setFlush(offset, TypeSize);
			object_state->setEscape();
		} else if (return_state == ParamStateType::TOP) {
			object_state->setFlush(offset, TypeSize);
			object_state->setCaptured();
		} else {
			assert(false && "other cases");
		}
	}
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
	errs() << "### Printing Map\n";
	if (map == NULL) {
		errs() << "NULL Map\n";
		return;
	}

	if (map->empty()) {
		errs() << "Empty Map\n";
		return;
	}

	for (state_t::iterator it = map->begin(); it != map->end(); it++) {
		ob_state_t *object_state = it->second;
		errs() << "#loc: " << *it->first << "\n";
		object_state->dump();
	}

	errs() << "\n***\n";
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
	PM.add(createCFGSimplificationPass());
	PM.add(new PMRobustness());
}
/* Enable the pass when opt level is greater than 0 */
// Module pass cannot be scheduled with EP_EarlyAsPossible
static RegisterStandardPasses 
	RegisterMyPass1(PassManagerBuilder::EP_ModuleOptimizerEarly,
registerPMRobustness);

/* Enable the pass when opt level is 0 */
// Don't turn this on when using EP_EarlyAsPossible. Otherwise, the pass will run twice
static RegisterStandardPasses
	RegisterMyPass2(PassManagerBuilder::EP_EnabledOnOptLevel0,
registerPMRobustness);
