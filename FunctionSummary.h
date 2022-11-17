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
	SmallVector<Value *, 8> parameters;
	SmallVector<InputState, 8> AbstrastInputState;

	void addAbsInput(InputState s) {
		AbstrastInputState.push_back(s);
	}

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
		static SmallVector<InputState, 8> getEmptyKey() {
			return {InputState::EMPTY_KEY};
		}

		static SmallVector<InputState, 8> getTombstoneKey() {
			return {InputState::TOMBSTONE_KEY};
		}

		static unsigned getHashValue(const SmallVector<InputState, 8> &V) {
			return static_cast<unsigned>(hash_combine_range(V.begin(), V.end()));
		}

		static bool isEqual(const SmallVector<InputState, 8> &LHS,
							const SmallVector<InputState, 8> &RHS) {
			return LHS == RHS;
		}
	};

	DenseMap<SmallVector<InputState, 8>, SmallVector<OuputState, 9>, SummaryDenseMapInfo> ResultMap;
	unsigned ArgSize = 0;
};

#endif
