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
	DIRTY_ESCAPED,
	CLWB_CAPTURED,
	CLWB_ESCAPED,
	CLEAN_CAPTURED,
	CLEAN_ESCAPED,
	TOP
};

struct OutputState {
	SmallVector<ParamState, 8> AbstrastOuputState;

	bool hasRetVal;
	ParamState retVal;
};

struct CallingContext {
	SmallVector<Value *, 8> parameters;
	SmallVector<ParamState, 8> AbstrastInputState;

	void addAbsInput(ParamState s) {
		AbstrastInputState.push_back(s);
	}

	void dump() {
		errs() << "Priginal parameters: ";
		for (Value *V : parameters) {
			errs() << *V << "\n";
		}

		errs() << "\nAbstract input state: ";
		for (ParamState &I : AbstrastInputState) {
			errs() << (int)I << "\n";
		}
		errs() << "\n";
	}
};


class FunctionSummary {
public:
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

	DenseMap<SmallVector<ParamState, 8>, OutputState *, SummaryDenseMapInfo> ResultMap;
	unsigned ArgSize = 0;
};

#endif
