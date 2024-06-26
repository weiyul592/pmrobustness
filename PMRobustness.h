#include "llvm/IR/IRBuilder.h"
#include "llvm/ADT/BitVector.h"
#include "FunctionSummary.h"
#include <cassert>

using namespace llvm;

#define UNKNOWNOFFSET 0xffffffff
#define VARIABLEOFFSET 0xfffffffe
//#define FUNC_PARAM_USE 100

struct ob_state_t;
struct ArrayInfo;

typedef DenseMap<const Value *, ob_state_t *> state_t;
typedef DenseMap<const Instruction *, state_t *> state_map_t;
typedef DenseMap<const BasicBlock *, state_t *> b_state_map_t;
typedef DenseMap<Value *, ArrayInfo *> addr_set_t;

typedef DenseSet<Value *> value_set_t;
typedef DenseMap<const Value *, value_set_t *> alias_set_t;

enum NVMOP {
	NVM_CLWB,
	NVM_CLFLUSH,
	NVM_FENCE,
	NVM_UNKNOWN
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
};

struct ArrayInfo : public DecomposedGEP {
	ParamStateType state;

	ArrayInfo(ArrayInfo *other) :
		state(other->state)
	{
		Base = other->Base;
		StructOffset = other->StructOffset;
		OtherOffset = other->OtherOffset;
		VarIndices = other->VarIndices;
		isArray = other->isArray;
	}

	ArrayInfo() {}

	void copyGEP(DecomposedGEP *other) {
		Base = other->Base;
		StructOffset = other->StructOffset;
		OtherOffset = other->OtherOffset;
		VarIndices = other->VarIndices;
		isArray = other->isArray;
	}

	void copyFrom(ArrayInfo *other) {
		Base = other->Base;
		StructOffset = other->StructOffset;
		OtherOffset = other->OtherOffset;
		VarIndices = other->VarIndices;
		isArray = other->isArray;
		state = other->state;
	}

	void mergeFrom(ArrayInfo *other) {
		assert(Base == other->Base);
		assert(StructOffset == other->StructOffset);
		assert(OtherOffset == other->OtherOffset);
		assert(VarIndices == other->VarIndices);

		if (state == ParamStateType::DIRTY_ESCAPED) {
			// TODO
		}
	}
};

static inline std::string getPosition(const Instruction * I, bool print = false)
{
	const DebugLoc & debug_location = I->getDebugLoc ();
	std::string position_string;
	{
		llvm::raw_string_ostream position_stream (position_string);
		debug_location . print (position_stream);
	}

	// Phi instructions do not have positions
	// TODO: some instructions have position:0

	if (print) {
		errs() << position_string << "\n";
	}

	return position_string;
}

bool checkPosition(Instruction * I, IRBuilder <> IRB, std::string sub)
{
	const DebugLoc & debug_location = I->getDebugLoc ();
	std::string position_string;
	{
		llvm::raw_string_ostream position_stream (position_string);
		debug_location . print (position_stream);
	}

	std::size_t found = position_string.find(sub);
	if (found!=std::string::npos)
		return true;

	return false;
}

static void printBitVectorAsIntervals(BitVector BV) {
	int IntervalStart = -1; //-1 if not in a strip of ones
	for (unsigned i = 0; i < BV.size(); i++) {
		if (BV[i] && IntervalStart == -1)
			IntervalStart = i;
		if ((!BV[i] || i+1 == BV.size())  && IntervalStart != -1) {
			errs() << IntervalStart << " - " << i << ", ";
			IntervalStart =  -1;
		}
	}

}

/**
 * TODO: may need to track the size of fields
 **/
// <dirty_byte, clwb_byte> is either <0, 0>, <1, 0>, or <1, 1>
class ob_state_t {
private:
	unsigned size;
	BitVector dirty_bytes;
	BitVector clwb_bytes;
	bool escaped;
	bool nonpmem;
	// the position where the state most recently
	// changes to dirty/escaped
	// empty from not dirty/escaped
	const Instruction* dirty_pos = nullptr;
	const Instruction* escaped_pos = nullptr; 
	bool mark_delete;

	void resize(unsigned s) {
		//if (s > (1 << 12))
		//	errs() << "oversize s: " << s << "\n";

		//assert(s <= (1 << 12));
		if (size < s) {
			size = s;
			dirty_bytes.resize(s);
			clwb_bytes.resize(s);
		}
	}

public:
	ob_state_t() :
		size(0),
		dirty_bytes(),
		clwb_bytes(),
		escaped(false),
		nonpmem(false),
		mark_delete(false)
	{}

	ob_state_t(unsigned s) :
		size(s),
		dirty_bytes(s),
		clwb_bytes(s),
		escaped(false),
		nonpmem(false),
		mark_delete(false)
	{ /*assert(s <= (1 << 12));*/ }

	ob_state_t(ob_state_t * other) :
		size(other->size),
		dirty_bytes(other->dirty_bytes),
		clwb_bytes(other->clwb_bytes),
		escaped(other->escaped),
		nonpmem(other->nonpmem),
		dirty_pos(other->dirty_pos),
		escaped_pos(other->escaped_pos),
		mark_delete(false)
	{ /*assert(size <= (1 << 12));*/ }

	// For merging positions, give priority to positions from this mergeFrom
	void mergeFrom(ob_state_t * other) {
		//assert(size == other->size);

		// <dirty_byte, clwb_byte> is either <0, 0>, <1, 0>, or <1, 1>
		// clwb_bytes = (clwb_bytes | other->clwb_bytes) &
		// ~((dirty_bytes ^ clwb_bytes) | (other->dirty_bytes ^ other->clwb_bytes));
		// dirty_bytes = dirty_bytes | other->dirty_bytes

		// <dirty_byte, clwb_byte> is either <0, 0>, <1, 0>, or <0, 1>
		// dirty_byte = dirty_byte | other->dirty_byte
		// clwb_byte = (clwb_byte | other->clwb_byte) & ~dirty_byte [result from last line]
		if(!isDirty()) dirty_pos = other->dirty_pos;
		dirty_bytes |= other->dirty_bytes;
		BitVector tmp(dirty_bytes);
		tmp.flip();

		clwb_bytes |= other->clwb_bytes;
		clwb_bytes &= tmp;

		if(!escaped) escaped_pos = other->escaped_pos; 
		escaped |= other->escaped;
		nonpmem = other->nonpmem;
	}

	void copyFrom(ob_state_t * src) {
		//assert(size == src->size);

		size = src->size;
		dirty_bytes = src->dirty_bytes;
		dirty_pos = src->dirty_pos;
		clwb_bytes = src->clwb_bytes;
		escaped = src->escaped;
		escaped_pos = src->escaped_pos;

//		if (nonpmem != src->nonpmem) {
//			assert(false);
//		}

		// nonpmem comes from Calling Contexts, which can be different
		// Since only the initial state is reset each time analyzing functions,
		// we propagate nonpmem by copying
		nonpmem = src->nonpmem;
	}

	bool copyFromCheckDiff(ob_state_t * src) {
		bool updated = false;
		updated |= (size != src->size);
		updated |= (dirty_bytes != src->dirty_bytes);
		updated |= (clwb_bytes != src->clwb_bytes);
		updated |= (escaped != src->escaped);

		size = src->size;
		dirty_bytes = src->dirty_bytes;
		clwb_bytes = src->clwb_bytes;
		escaped = src->escaped;

		// nonpmem comes from Calling Contexts, and is not considered as a change in states
		nonpmem = src->nonpmem;

		return updated;
	}

	// return true: modified; return else: unchanged
	bool setDirty(unsigned start, unsigned len, const Instruction *new_dirty_pos) {
		if (nonpmem)
			return false;

		unsigned end = start + len;
		if (end > size) {
			resize(end);
		}

		//errs() << "start: " << start << "; len: " << len << "; end:" << end << "\n";
		//errs() << "actual size: " << size << "\n";
		int index1 = dirty_bytes.find_first_unset_in(start, end);
		int index2 = clwb_bytes.find_first_in(start, end);

		// dirty_byte are all 1 and clwb_bytes are all 0, then no change
		if (index1 == -1 && index2 == -1)
			return false;

		if(!isDirty()) 
			dirty_pos = new_dirty_pos;
		dirty_bytes.set(start, end);
		clwb_bytes.reset(start, end);	

		return true;
	}

	// Check if there are other dirty bytes other than the ones in [start, end)
	bool hasDirtyBytesNotIn(unsigned start, unsigned len) {
		if (nonpmem)
			return false;

		unsigned end = start + len;
		if (end > size) {
			resize(end);
		}

		int setBitBeforeStart = dirty_bytes.find_first_in(0, start);
		int setBitAfterEnd = dirty_bytes.find_first_in(end, dirty_bytes.size());

		if (setBitBeforeStart == -1 && setBitAfterEnd == -1)
			return false;

		return true;
	}

	bool MultipleDirtyFieldsBadApproximation() {
		if (nonpmem)
			return false;

		unsigned j = dirty_bytes.size() / 8;
		if (dirty_bytes.size() % 8 != 0)
			j++;

		int dirty_count = 0;
		for (unsigned i = 0; i < j; i++) {
			unsigned start = i * 8;
			unsigned end = (i + 1) * 8;
			if (end >= dirty_bytes.size())
				end = dirty_bytes.size();
/*
			if (dirty_bytes.find_first_in(start, end) != -1) {
				dirty_count++;

				if (dirty_count >= 2) {
					return true;
				}
			}
*/
		}

		return false;
	}

	// TODO: start + len and size?
	// Flush wrapper function may flush cache lines exceeding the size of this object
	bool setFlush(unsigned start, unsigned len, bool onlyFlushWrittenBytes = false) {
		if (nonpmem)
			return false;

		if (start > size && onlyFlushWrittenBytes) {
			errs() << "FIXME: Flush unknown bytes\n";
			return false;
			//assert(false && "Flush unknown bytes");
		}

		unsigned end = start + len;
		if (len == (unsigned)-1 && onlyFlushWrittenBytes) {
			// start + len may overflow
			end = size;
		} else if (end > size && onlyFlushWrittenBytes) {
			end = size;
		} else if (end > size)
			resize(end);

		int index1 = dirty_bytes.find_first_in(start, end);
		int index2 = clwb_bytes.find_first_in(start, end);

		// dirty_byte and clwb_bytes are all 0, then no change
		if (index1 == -1 && index2 == -1)
			return false;

		dirty_bytes.reset(start, end);
		clwb_bytes.reset(start, end);

		return true;
	}

	// TODO: start + len and size?
	// Flush wrapper function may flush cache lines exceeding the size of this object
	bool setClwb(unsigned start, unsigned len, bool onlyFlushWrittenBytes = false) {
		if (nonpmem)
			return false;

		if (start > size && onlyFlushWrittenBytes) {
			errs() << "FIXME: Clwb unknown bytes\n";
			return false;
			//assert(false && "Clwb unknown bytes");
		}

		unsigned end = start + len;
		if (end > size && onlyFlushWrittenBytes) {
			end = size;
		} else if (end > size)
			resize(end);

		// set clwb_bytes for bytes in dirty_bytes
		BitVector tmp(dirty_bytes);
		tmp.reset(0, start);
		tmp.reset(end, tmp.size());

		BitVector old_clwb_bytes(clwb_bytes);
		clwb_bytes |= tmp;

		if (old_clwb_bytes == clwb_bytes)
			return false;	// No change

		return true;
	}

	// return true: modified; return else: unchanged
	bool setEscape(const Instruction *new_escaped_pos) {
		if (escaped == false) {
			escaped_pos = new_escaped_pos;
			escaped = true;
			return true;
		}

		return false;
	}

	bool setCaptured() {
		if (escaped == true) {
			escaped = false;
			return true;
		}

		return false;
	}

	bool isEscaped() {
		return escaped;
	}

	void setNonPmem() {
		nonpmem = true;
	}

	bool isNonPmem() {
		return nonpmem;
	}

	void markDelete() {
		mark_delete = true;
	}

	void unmarkDelete() {
		mark_delete = false;
	}

	bool shouldDelete() {
		return mark_delete;
	}

	unsigned getSize() {
		return size;
	}

	void setSize(unsigned s) {
		size = s;
	}

	ParamStateType checkState() {
		return checkState(0, size);
	}

	ParamStateType checkState(unsigned startByte, unsigned len) {
		unsigned endByte = startByte + len;
		//errs() << "range: " << startByte << " - " << startByte + len << "; size: " << size << "\n";

		if (size == 0) {
			if (escaped)
				return ParamStateType::CLEAN_ESCAPED;
			else
				return ParamStateType::TOP;
		}

		if (startByte >= size) {
			errs() << "Checking state out of range\n";
			errs() << "range: " << startByte << " - " << startByte + len << "; size: " << size << "\n";
			return ParamStateType::TOP;
			//assert(false);
		}

		if (endByte > size)
			endByte = size;

		BitVector tmp(dirty_bytes);
		tmp &= clwb_bytes;

		if (escaped) {
			if (dirty_bytes.find_first_in(startByte, endByte) == -1) {
				// dirty_bytes are all 0
				return ParamStateType::CLEAN_ESCAPED;
			} else if (dirty_bytes == tmp) {
				// all set dirty_bytes are clwbed;
				return ParamStateType::CLWB_ESCAPED;
			} else {
				// Some set dirty_bytes are not clwbed
				return ParamStateType::DIRTY_ESCAPED;
			}
		} else {
			if (dirty_bytes.find_first_in(startByte, endByte) == -1) {
				// dirty_bytes are all 0
				return ParamStateType::CLEAN_CAPTURED;
			} else if (dirty_bytes == tmp) {
				// all set dirty_bytes are clwbed;
				return ParamStateType::CLWB_CAPTURED;
			} else {
				// Some set dirty_bytes are not clwbed
				return ParamStateType::DIRTY_CAPTURED;
			}
		}
	}

	bool isDirty() {
		for (auto Itr = dirty_bytes.set_bits_begin(); Itr != dirty_bytes.set_bits_end(); Itr++)
			if(!clwb_bytes.test(*Itr))
				return true;
		return false; 

		//this bitvector constructor causes memory errors in certain cases, possibly a llvm bug.
		/*BitVector tmp(dirty_bytes);
		tmp ^= clwb_bytes;

		// fence are not implemented yet
//		if (tmp.any())
//			return true;

		if (dirty_bytes.any())
			return true;

		return false;*/
	}

	void computeDirtyBytes(DirtyBytesInfo *info) {
		//dump();

		BitVector only_dirty_bytes(dirty_bytes);
		only_dirty_bytes ^= clwb_bytes;

		int i = only_dirty_bytes.find_first();
		assert(i != -1);

		while (i != -1) {
			// Store [i, j)
			int j = only_dirty_bytes.find_next_unset(i);

			if (j == -1) {
				j = only_dirty_bytes.size();
				info->push(i, j);
				break;
			}

			info->push(i, j);
			assert(j >= 1);
			i = only_dirty_bytes.find_next(j - 1);
		}

		info->finalize();
	}

	const Instruction *getDirtyPos() {
		return dirty_pos;
	}

	const Instruction *getEscapedPos() {
		return escaped_pos;
	}

	std::string getDirtyEscapedPos() {
		return " dirty at " + (dirty_pos ? getPosition(dirty_pos): "") + ", escaped at " + (escaped_pos ? getPosition(escaped_pos): "");
	}

	void dump() {
		errs() << "bit vector size: " << size << "\n";
		if (size != 0) {
			if (dirty_bytes.any()) {
				errs() << "dirty bytes: ";
				printBitVectorAsIntervals(dirty_bytes);
				errs() << "\nfirst dirty at " << *dirty_pos << "\n";

				errs() << "clwb bytes: ";
				printBitVectorAsIntervals(clwb_bytes);
				errs() << "\n";
			}
		}

		if (escaped)
			errs() << "escaped at " << *escaped_pos;
		else
			errs() << "captured";

		if (nonpmem)
			errs() << "; nonpmem";
		else
			errs() << "; pmem";
		errs() << "\n";
	}
};

void printDecomposedGEP(DecomposedGEP &Decom) {
	errs() << "Store Base: " << *Decom.Base << "\t";
	errs() << "Struct Offset: " << Decom.StructOffset << "\t";
	errs() << "Other Offset: " << Decom.OtherOffset << "\t";
	errs() << "Has VarIndices: " << Decom.VarIndices.size() << "\n";
	/*
	for (unsigned i = 0 ; i < Decom.VarIndices.size(); i++) {
		VariableGEPIndex &VI = Decom.VarIndices[i];
		errs() << *VI.V << "\n";
		errs() << "(" << VI.ZExtBits << ", " << VI.SExtBits << ")\t";
		errs() << "Scale: " << VI.Scale << "\n";
	}*/
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
