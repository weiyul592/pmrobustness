#ifndef __FUNCTION_SUMMARY__
#define __FUNCTION_SUMMARY__ 

using namespace llvm;

enum class ParamStateType {
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

class ParamState {
public:
	ParamState() : state(ParamStateType::TOP) {}

	ParamState(ParamStateType state) : state(state) {}

	ParamStateType get_state() const {
		return state;
	}

	void setState(ParamStateType s) {
		state = s;
	}

	std::string print() {
		switch (state) {
			case ParamStateType::EMPTY_KEY:
				return "EMPTY_KEY";
			case ParamStateType::TOMBSTONE_KEY:
				return "TOMBSTONE_KEY";
			case ParamStateType::BOTTOM:
				return "BOTTOM";
			case ParamStateType::NON_PMEM:
				return "NON_PMEM";
			case ParamStateType::DIRTY_CAPTURED:
				return "DIRTY_CAPTURED";
			case ParamStateType::DIRTY_ESCAPED:
				return "DIRTY_ESCAPED";
			case ParamStateType::CLWB_CAPTURED:
				return "CLWB_CAPTURED";
			case ParamStateType::CLWB_ESCAPED:
				return "CLWB_ESCAPED";
			case ParamStateType::CLEAN_CAPTURED:
				return "CLEAN_CAPTURED";
			case ParamStateType::CLEAN_ESCAPED:
				return "CLEAN_ESCAPED";
			case ParamStateType::TOP:
				return "TOP";
			default:
				return "UNKNOWN";
		}
	}

	bool isDirty() {
		return state == ParamStateType::DIRTY_CAPTURED || state == ParamStateType::DIRTY_ESCAPED;
	}

	bool isClwb() {
		return state == ParamStateType::CLWB_CAPTURED || state == ParamStateType::CLWB_ESCAPED;
	}

	bool isClean() {
		return state == ParamStateType::TOP ||
			state == ParamStateType::CLEAN_CAPTURED || state == ParamStateType::CLEAN_ESCAPED;
	}

	bool isCaptured() {
		return state == ParamStateType::TOP || state == ParamStateType::DIRTY_CAPTURED ||
			state == ParamStateType::CLWB_CAPTURED || state == ParamStateType::CLEAN_CAPTURED;
	}

	bool isEscaped() {
		return state == ParamStateType::DIRTY_ESCAPED || state == ParamStateType::CLWB_ESCAPED ||
			state == ParamStateType::CLEAN_ESCAPED;
	}

	inline bool isLowerThan(const ParamState& other) const {
		return isLowerThan(other.get_state());
	}

	// Check if `this` is strictly lower than `other` in lattice
	// Caution: returning false could also mean that two states are uncomparable
	inline bool isLowerThan(const ParamStateType other_state) const {
		if (state == ParamStateType::EMPTY_KEY || state == ParamStateType::TOMBSTONE_KEY ||
			state == ParamStateType::NON_PMEM || other_state == ParamStateType::EMPTY_KEY ||
			other_state == ParamStateType::TOMBSTONE_KEY || other_state == ParamStateType::NON_PMEM) {
			return false;
		}

		switch (state) {
			case ParamStateType::BOTTOM:
				if (other_state != ParamStateType::BOTTOM)
					return true;
				break;
			case ParamStateType::DIRTY_ESCAPED:
				if (other_state != ParamStateType::BOTTOM &&
					other_state != ParamStateType::DIRTY_ESCAPED)
					return true;
				break;
			case ParamStateType::DIRTY_CAPTURED:
				if (other_state == ParamStateType::CLWB_CAPTURED ||
					other_state == ParamStateType::CLEAN_CAPTURED ||
					other_state == ParamStateType::TOP)
					return true;
				break;
			case ParamStateType::CLWB_ESCAPED:
				if (other_state == ParamStateType::CLWB_CAPTURED ||
					other_state == ParamStateType::CLEAN_ESCAPED ||
					other_state == ParamStateType::CLEAN_CAPTURED ||
					other_state == ParamStateType::TOP)
					return true;
				break;
			case ParamStateType::CLWB_CAPTURED:
			case ParamStateType::CLEAN_ESCAPED:
				if (other_state == ParamStateType::CLEAN_CAPTURED ||
					other_state == ParamStateType::TOP)
					return true;
				break;
			case ParamStateType::CLEAN_CAPTURED:
				if (other_state == ParamStateType::TOP)
					return true;
				break;
			case ParamStateType::TOP:
				break;
			default:
				break;
		}

		return false;
	}

	inline bool operator ==(const ParamState& other) const {
		return state == other.get_state();
	}

	inline bool operator !=(const ParamState& other) const {
		return state != other.get_state();
	}

	private:
		ParamStateType state;
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

	std::vector<std::pair<int, int>> * getDirtyBytes() {
		return lst;
	}

	void finalize() {
/*
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
*/
		if (lst != NULL)
			delete lst;

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
	bool marksEscDirObj;	// Whether this function ever marks anything as escaped and dirty

	DirtyBytesInfo * getOrCreateDirtyBytesInfo(unsigned i) {
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

	DirtyBytesInfo * getDirtyBytesInfo(unsigned i) {
		return (*DirtyBytesList)[i];
	}

	ParamStateType getStateType(unsigned i) {
		return AbstractOutputState[i].get_state();
	}

	ParamState& getState(unsigned i) {
		return AbstractOutputState[i];
	}

	void dump() {
		errs() << "Abstract output state: ";
		for (ParamState &I : AbstractOutputState) {
			errs() << I.print() << "\t";
		}
		errs() << "\n";

		if (hasRetVal)
			errs() << "Abstract return state: " << retVal.print() << "\n";

		errs() << "\n";
	}

};

struct CallingContext {
	SmallVector<Value *, 8> parameters;
	SmallVector<ParamState, 8> AbstractInputState;

	CallingContext() {}

	CallingContext(CallingContext *other) {
		parameters = other->parameters;
		AbstractInputState = other->AbstractInputState;
	}

	void addAbsInput(ParamStateType s) {
		AbstractInputState.push_back(ParamState(s));
	}

	ParamStateType getStateType(unsigned i) {
		return AbstractInputState[i].get_state();
	}

	ParamState& getState(unsigned i) {
		return AbstractInputState[i];
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
		for (ParamState&I : AbstractInputState) {
			errs() << I.print() << "\t";
		}
		errs() << "\n";
	}
};

hash_code hash_value(const ParamState &value) {
	return ::llvm::hashing::detail::hash_integer_value(
		static_cast<uint64_t>(value.get_state()));
}

// lhs >= rhs ?
bool isStateVectorHigherOrEqual(SmallVector<ParamState, 8> &lhs, SmallVector<ParamState, 8> &rhs) {
	assert(lhs.size() == rhs.size());

	for (unsigned i = 0; i < lhs.size(); i++) {
		if (rhs[i].isLowerThan(lhs[i]) != true &&
			lhs[i] != rhs[i]) {

			return false;
		}
	}

	return true;
}

class FunctionSummary {
public:
	struct SummaryDenseMapInfo;

	//using iterator = DenseMap<SmallVector<ParamState, 8>, OutputState *, SummaryDenseMapInfo>::iterator;
	typedef DenseMap<SmallVector<ParamState, 8>, OutputState *, SummaryDenseMapInfo> result_map_t;

	struct SummaryDenseMapInfo {
		static SmallVector<ParamState, 8> getEmptyKey() {
			return {ParamState(ParamStateType::EMPTY_KEY)};
		}

		static SmallVector<ParamState, 8> getTombstoneKey() {
			return {ParamState(ParamStateType::TOMBSTONE_KEY)};
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
			state->DirtyBytesList = NULL;
			state->hasRetVal = false;
			state->retVal.setState(ParamStateType::BOTTOM);
			state->marksEscDirObj = false;
			ResultMap[Context->AbstractInputState] = state;
		}

		return state;
	}

	OutputState * getLeastUpperResult(CallingContext *Context) {
		SmallVector<ParamState, 8> *leastUpperContext = NULL;
		for (result_map_t::iterator it = ResultMap.begin(); it != ResultMap.end(); it++) {
			SmallVector<ParamState, 8> *cur = &it->first;
			if (isStateVectorHigherOrEqual(*cur, Context->AbstractInputState)) {
				if (leastUpperContext == NULL)
					leastUpperContext = cur;
				else if (isStateVectorHigherOrEqual(*leastUpperContext, *cur))
					leastUpperContext = cur;
				// TODO: Multiple higher contexts
			}
		}

		if (leastUpperContext == NULL)
			return NULL;

/*
		errs() << "Using least upper context: \n";
		errs() << "Abstract input state: ";
		for (ParamState& I : *leastUpperContext) {
			errs() << I.print() << "\t";
		}
		errs() << "\n";
*/
		return ResultMap.lookup(*leastUpperContext);
	}

	result_map_t * getResultMap() {
		return &ResultMap;
	}

private:
	result_map_t ResultMap;
	unsigned ArgSize = 0;
};

namespace llvm {
template<> struct DenseMapInfo<CallingContext *> {
	static inline CallingContext * getEmptyKey() {
		uintptr_t Val = static_cast<uintptr_t>(-1);
		Val <<= PointerLikeTypeTraits<CallingContext *>::NumLowBitsAvailable;
		return reinterpret_cast<CallingContext *>(Val);
	}

	static inline CallingContext * getTombstoneKey() {
		uintptr_t Val = static_cast<uintptr_t>(~1U);
		Val <<= PointerLikeTypeTraits<CallingContext *>::NumLowBitsAvailable;
		return reinterpret_cast<CallingContext *>(Val);
	}

	static unsigned getHashValue(const CallingContext *V) {
		return static_cast<unsigned>(hash_combine_range(V->AbstractInputState.begin(), V->AbstractInputState.end()));
	}

	static bool isEqual(const CallingContext *LHS,
						const CallingContext *RHS) {
		if (LHS == getEmptyKey() || RHS == getEmptyKey() ||
			LHS == getTombstoneKey() || RHS == getTombstoneKey())
			return LHS == RHS;

		return LHS->AbstractInputState == RHS->AbstractInputState;
	}
};
}

#endif
