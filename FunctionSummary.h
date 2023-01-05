#ifndef __FUNCTION_SUMMARY__
#define __FUNCTION_SUMMARY__ 

using namespace llvm;

enum class ParamState {
	// TODO: Do we need to model the partial order?
	EMPTY_KEY,      // for hash table key
	TOMBSTONE_KEY,  // for hash table key

	BOTTOM,
	NON_PMEM,
	DIRTY_CAPTURED,
	DIRTY_ESCAPED,	// 0x5
	CLWB_CAPTURED,
	CLWB_ESCAPED,
	CLEAN_CAPTURED,
	CLEAN_ESCAPED,
	TOP				// 0xa
};

struct OutputState {
	SmallVector<ParamState, 8> AbstractOutputState;

	bool hasRetVal;
	ParamState retVal;

	void dump() {
		errs() << "Abstract output state: ";
		for (ParamState &I : AbstractOutputState) {
			errs() << (int)I << "\t";
		}
		errs() << "\n";

		if (hasRetVal)
			errs() << "Abstract return state: " << (int)retVal << "\n";

		errs() << "\n";
	}

};

struct CallingContext {
	SmallVector<Value *, 8> parameters;
	SmallVector<ParamState, 8> AbstractInputState;

	void addAbsInput(ParamState s) {
		AbstractInputState.push_back(s);
	}

	void dump() {
		/*
		errs() << "Original parameters: ";
		for (Value *V : parameters) {
			errs() << *V << "\t";
		}
		errs() << "\n";
		*/

		errs() << "Abstract input state:  ";
		for (ParamState &I : AbstractInputState) {
			errs() << (int)I << "\t";
		}
		errs() << "\n";
	}
};

class FunctionSummary {
public:
	struct SummaryDenseMapInfo;

	using iterator = DenseMap<SmallVector<ParamState, 8>, OutputState *, SummaryDenseMapInfo>::iterator;
	typedef DenseMap<SmallVector<ParamState, 8>, OutputState *, SummaryDenseMapInfo> result_map_t;

	struct SummaryDenseMapInfo {
		static SmallVector<ParamState, 8> getEmptyKey() {
			return {ParamState::EMPTY_KEY};
		}

		static SmallVector<ParamState, 8> getTombstoneKey() {
			return {ParamState::TOMBSTONE_KEY};
		}

		static unsigned getHashValue(const SmallVector<ParamState, 8> &V) {
			return static_cast<unsigned>(hash_combine_range(V.begin(), V.end()));
		}

		static bool isEqual(const SmallVector<ParamState, 8> &LHS,
							const SmallVector<ParamState, 8> &RHS) {
			return LHS == RHS;
		}
	};

	OutputState * getResult(CallingContext *Context) {
		return ResultMap.lookup(Context->AbstractInputState);
	}

	OutputState * getOrCreateResult(CallingContext *Context) {
		OutputState *state = ResultMap.lookup(Context->AbstractInputState);
		if (state == NULL) {
			state = new OutputState();
			ResultMap[Context->AbstractInputState] = state;
		}

		return state;
	}

	result_map_t * getResultMap() {
		return &ResultMap;
	}

private:
	result_map_t ResultMap;
	unsigned ArgSize = 0;
};

#endif
