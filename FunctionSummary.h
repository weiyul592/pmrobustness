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

struct DirtyBytesInfo {
	// isComplex: when there are too many stores that are not consecutive in byte positions.
	// simply approximate the object as dirty in all btyes
	bool isComplex;
	std::vector<std::pair<int, int>> *lst;
	std::vector<std::pair<int, int>> *tmp_lst;

	void push(int i, int j) {
		if (tmp_lst == NULL)
			tmp_lst = new std::vector<std::pair<int, int>>();

		//errs() << "[" << i << ", " << j << ") is pushed\n";
		tmp_lst->emplace_back(i, j);
	}

	DirtyBytesInfo() {
		isComplex = false;
		lst = NULL;
		tmp_lst = NULL;
	}

	std::vector<std::pair<int, int>> * getDirtyBtyes() {
		return lst;
	}

	void finalize() {
		if (lst != NULL) {
			// FIXME: Compare elements in lst and tmp_list (is this needed? We are static)
			bool should_assert = false;
			if (lst->size() != tmp_lst->size())
				should_assert = true;
			else {
				for (unsigned i = 0; i < lst->size(); i++) {
					std::pair<int, int> &A = (*lst)[i];
					std::pair<int, int> &B = (*tmp_lst)[i];

					if (A.first != B.first || A.second != B.second) {
						should_assert = true;
						break;
					}
				}
			}

			if (should_assert)
				assert(false && "dirty btyes are different");
		}

		lst = tmp_lst;
		tmp_lst = NULL;
	}
};

struct OutputState {
	SmallVector<ParamState, 8> AbstractOutputState;

	// So far it is only accurate when the input state is clean
	// Because when the input state is dirty, all bytes are approximated as dirty.
	std::vector<DirtyBytesInfo *> *DirtyBytesList;

	bool hasRetVal;
	ParamState retVal;

	DirtyBytesInfo * getOrCreateDirtyBtyesInfo(unsigned i) {
		if (DirtyBytesList == NULL)
			DirtyBytesList = new std::vector<DirtyBytesInfo *>();

		if (DirtyBytesList->size() < i + 1)
			DirtyBytesList->resize(i + 1);

		DirtyBytesInfo *info = (*DirtyBytesList)[i];
		if (info == NULL) {
			info = new DirtyBytesInfo();
			(*DirtyBytesList)[i] = info;
		}

		return info;
	}

	DirtyBytesInfo * getDirtyBtyesInfo(unsigned i) {
		return (*DirtyBytesList)[i];
	}

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

	CallingContext() {}

	CallingContext(CallingContext *other) {
		parameters = other->parameters;
		AbstractInputState = other->AbstractInputState;
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

namespace llvm {
template<> struct DenseMapInfo<CallingContext> {
	static CallingContext * getEmptyKey() {
		return new CallingContext();
	}

	static CallingContext * getTombstoneKey() {
		return new CallingContext();
	}

	static unsigned getHashValue(const CallingContext *V) {
		return static_cast<unsigned>(hash_combine_range(V->AbstractInputState.begin(), V->AbstractInputState.end()));
	}

	static bool isEqual(const CallingContext *LHS,
						const CallingContext *RHS) {
		return LHS->AbstractInputState == RHS->AbstractInputState;
	}
};
}

#endif
