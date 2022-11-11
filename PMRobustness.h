#include "llvm/IR/IRBuilder.h"
#include "llvm/ADT/BitVector.h"
#include <cassert>

using namespace llvm;

#define UNKNOWNOFFSET 0xffffffff
#define VARIABLEOFFSET 0xfffffffe
#define FUNC_PARAM_USE 100

struct ob_state_t;
struct DecomposedGEP;

typedef DenseMap<const Value *, ob_state_t *> state_t;
typedef DenseMap<const Instruction *, state_t *> state_map_t;
typedef DenseMap<Value *, DecomposedGEP *> addr_set_t;

enum NVMOP {
	NVM_CLWB,
	NVM_CLFLUSH,
	NVM_FENCE,
	NVM_UNKNOWN
};

enum class AbsInputState {
	// TODO: Do we need to model the partial order?
	EMPTY_KEY,		// for hash table key
	TOMBSTONE_KEY,	// for hash table key

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

enum class AbsOuputState {
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


struct VariableGEPIndex {
	// An opaque Value - we can't decompose this further.
	const Value *V;

	// We need to track what extensions we've done as we consider the same Value
	// with different extensions as different variables in a GEP's linear
	// expression;
	// e.g.: if V == -1, then sext(x) != zext(x).
	unsigned ZExtBits;
	unsigned SExtBits;

	APInt Scale;

	bool operator==(const VariableGEPIndex &Other) const {
		return V == Other.V && ZExtBits == Other.ZExtBits &&
					 SExtBits == Other.SExtBits && Scale == Other.Scale;
	}

	bool operator!=(const VariableGEPIndex &Other) const {
		return !operator==(Other);
	}
};

struct DecomposedGEP {
	// Base pointer of the GEP
	const Value *Base;
	// Total constant offset w.r.t the base from indexing into structs
	APInt StructOffset;
	// Total constant offset w.r.t the base from indexing through
	// pointers/arrays/vectors
	APInt OtherOffset;
	// Scaled variable (non-constant) indices.
	SmallVector<VariableGEPIndex, 4> VarIndices;

	bool isArray;

	uint64_t getOffsets() {
		if (VarIndices.size() == 0) {
			assert(StructOffset.getSExtValue() >= 0);
			//assert(OtherOffset.getSExtValue() >= 0);
			if (OtherOffset.getSExtValue() < 0) {
#ifdef PMROBUST_DEBUG
				errs() << "strange offset: " << StructOffset.getSExtValue() << ", " << OtherOffset.getSExtValue() << "\n";
#endif
				return UNKNOWNOFFSET;
			}

			return StructOffset.getZExtValue() + OtherOffset.getZExtValue();
		} else
			return VARIABLEOFFSET;
	}

	uint64_t getStructOffset() {
		assert(StructOffset.getSExtValue() >= 0);
		return StructOffset.getZExtValue();
	}

	void copyFrom(DecomposedGEP &other) {
		Base = other.Base;
		StructOffset = other.StructOffset;
		OtherOffset = other.OtherOffset;
		VarIndices = other.VarIndices;
		isArray = other.isArray;
	}
};

/**
 * TODO: may need to track the size of fields
 **/
// <flushed_bit, clwb_bit> is either <1, 0>, <0, 1>, or <0, 0>
struct ob_state_t {
	unsigned size;
	BitVector flushed_bits;
	BitVector clwb_bits;
	BitVector escaped_bits;
	bool escaped;

	ob_state_t() :
		size(0),
		flushed_bits(),
		clwb_bits(),
		escaped_bits(),
		escaped(false)
	{}

	ob_state_t(unsigned s) :
		size(s),
		flushed_bits(s),
		clwb_bits(s),
		escaped_bits(s),
		escaped(false)
	{ assert(s <= (1 << 12)); }

	ob_state_t(ob_state_t * other) :
		size(other->size),
		flushed_bits(other->flushed_bits),
		clwb_bits(other->clwb_bits),
		escaped_bits(other->escaped_bits),
		escaped(other->escaped)
	{ assert(size <= (1 << 12)); }

	void mergeFrom(ob_state_t * other) {
		//assert(size == other->size);

		//BitVector clwb_bits = clwb_bits & other->clwb_bits |
		//	((clwb_bits ^ other->clwb_bits) & (flushed_bits ^ other->flushed_bits));
		BitVector tmp1(clwb_bits);
		BitVector tmp2(flushed_bits);
		tmp1 ^= other->clwb_bits;
		tmp2 ^= other->flushed_bits;
		tmp1 &= tmp2;

		clwb_bits &= other->clwb_bits;
		clwb_bits |= tmp1;

		flushed_bits &= other->flushed_bits;
		escaped_bits |= other->escaped_bits;
		escaped |= other->escaped;
	}

	void copyFrom(ob_state_t * src) {
		//assert(size == src->size);

		size = src->size;
		flushed_bits = src->flushed_bits;
		clwb_bits = src->clwb_bits;
		escaped_bits = src->escaped_bits;
		escaped = src->escaped;
	}

	void resize(unsigned s) {
		assert(s <= (1 << 12));
		if (size < s) {
			size = s;
			flushed_bits.resize(size);
			clwb_bits.resize(size);
			escaped_bits.resize(size, escaped);
		}
	}

	// return true: modified; return else: unchanged
	bool setDirty(unsigned start, unsigned len) {
		unsigned end = start + len;
		if (end > size) {
			resize(end);
		}

		int index1 = flushed_bits.find_first_in(start, end);
		int index2 = clwb_bits.find_first_in(start, end);

		if (index1 == -1 && index2 == -1)
			return false;
		else {
			flushed_bits.reset(start, end);
			clwb_bits.reset(start, end);
			return true;
		}
	}

	// TODO: start + len and size?
	void setFlush(unsigned start, unsigned len) {
		unsigned end;
		if (len == (unsigned)-1) {
			assert("false");
		} else if (start + len > size){
			end = size;
		} else {
			end = start + len;
		}

		if (start >= size) {
			errs() << "FIXME: Flush unknown bits\n";
			return;
			//assert(false && "Flush unknown bits");
		}

		flushed_bits.set(start, end);
		clwb_bits.reset(start, end);
	}

	// TODO: start + len and size?
	void setClwb(unsigned start, unsigned len) {
		unsigned end;
		if (len == (unsigned)-1) {
			assert("false");
		} else if (start + len > size){
			end = size;
		} else {
			end = start + len;
		}

		if (start >= size) {
			assert(false && "Clwb unknown bits");
		}

		BitVector tmp(flushed_bits);
		tmp.flip();

		clwb_bits.set(start, end);
		clwb_bits &= tmp;
	}

	// return true: modified; return else: unchanged
	bool setEscape(unsigned start, unsigned len, bool objectEscape = false) {
		escaped = objectEscape;
		unsigned end = start + len;
		if (end > size) {
			resize(end);
		}

		int index = escaped_bits.find_first_unset_in(start, end);
		if (index == -1)
			return false;
		else {
			escaped_bits.set(start, end);
			return true;
		}
	}

	unsigned getSize() {
		return size;
	}

	void setSize(unsigned s) {
		size = s;
	}

	void print() {
		errs() << "bit vector size: " << size << "\n";
		for (unsigned i = 0; i < size; i++) {
			errs() << flushed_bits[i];
		}
		errs() << "\n";
		for (unsigned i = 0; i < size; i++) {
			errs() << clwb_bits[i];
		}
		errs() << "\n";
		for (unsigned i = 0; i < size; i++) {
			errs() << escaped_bits[i];
		}
		errs() << "\n";
	}
};

class FunctionSummary {
	struct SummaryDenseMapInfo {
		static SmallVector<AbsInputState, 10> getEmptyKey() {
			return {AbsInputState::EMPTY_KEY};
		}

		static SmallVector<AbsInputState, 10> getTombstoneKey() {
			return {AbsInputState::TOMBSTONE_KEY};
		}

		static unsigned getHashValue(const SmallVector<AbsInputState, 10> &V) {
			return static_cast<unsigned>(hash_combine_range(V.begin(), V.end()));
		}

		static bool isEqual(const SmallVector<AbsInputState, 10> &LHS,
							const SmallVector<AbsInputState, 10> &RHS) {
			return LHS == RHS;
		}
	};

	DenseMap<SmallVector<AbsInputState, 10>, SmallVector<AbsOuputState, 11>, SummaryDenseMapInfo> ResultMap;
	unsigned ArgSize = 0;
};

void printDecomposedGEP(DecomposedGEP &Decom) {
	errs() << "Store Base: " << *Decom.Base << "\t";
	errs() << "Struct Offset: " << Decom.StructOffset << "\t";
	errs() << "Other Offset: " << Decom.OtherOffset << "\t";
	errs() << "Has VarIndices: " << Decom.VarIndices.size() << "\n";
}

static inline Value *getPosition(Instruction * I, IRBuilder <> IRB, bool print = false)
{
	const DebugLoc & debug_location = I->getDebugLoc ();
	std::string position_string;
	{
		llvm::raw_string_ostream position_stream (position_string);
		debug_location . print (position_stream);
	}

	if (print) {
		errs() << position_string << "\n";
	}

	return IRB.CreateGlobalStringPtr (position_string);
}

/// To ensure a pointer offset fits in an integer of size PointerSize
/// (in bits) when that size is smaller than the maximum pointer size. This is
/// an issue, for example, in particular for 32b pointers with negative indices
/// that rely on two's complement wrap-arounds for precise alias information
/// where the maximum pointer size is 64b.
static APInt adjustToPointerSize(APInt Offset, unsigned PointerSize) {
    assert(PointerSize <= Offset.getBitWidth() && "Invalid PointerSize!");
    unsigned ShiftBits = Offset.getBitWidth() - PointerSize;
    return (Offset << ShiftBits).ashr(ShiftBits);
}

static unsigned getMaxPointerSize(const DataLayout &DL) {
    unsigned MaxPointerSize = DL.getMaxPointerSizeInBits();
    //if (MaxPointerSize < 64 && ForceAtLeast64Bits) MaxPointerSize = 64;
    //if (DoubleCalcBits) MaxPointerSize *= 2;

    return MaxPointerSize;
}
