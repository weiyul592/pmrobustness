#ifndef __FUNCTION_SUMMARY__
#define __FUNCTION_SUMMARY__ 

using namespace llvm;

enum class InputState {
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

enum class OuputState {
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

struct CallingContext {
	SmallVector<Value *, 10> parameters;
	SmallVector<InputState, 10> AbstrastInputState;

	void dump() {
		errs() << "Priginal parameters: ";
		for (Value *V : parameters) {
			errs() << *V << "\n";
		}

		errs() << "\nAbstract input state: ";
		for (InputState &I : AbstrastInputState) {
			errs() << (int)I << "\n";
		}
		errs() << "\n";
	}
};


class FunctionSummary {
public:
	struct SummaryDenseMapInfo {
		static SmallVector<InputState, 10> getEmptyKey() {
			return {InputState::EMPTY_KEY};
		}

		static SmallVector<InputState, 10> getTombstoneKey() {
			return {InputState::TOMBSTONE_KEY};
		}

		static unsigned getHashValue(const SmallVector<InputState, 10> &V) {
			return static_cast<unsigned>(hash_combine_range(V.begin(), V.end()));
		}

		static bool isEqual(const SmallVector<InputState, 10> &LHS,
							const SmallVector<InputState, 10> &RHS) {
			return LHS == RHS;
		}
	};

	DenseMap<SmallVector<InputState, 10>, SmallVector<OuputState, 11>, SummaryDenseMapInfo> ResultMap;
	unsigned ArgSize = 0;
};

#endif
