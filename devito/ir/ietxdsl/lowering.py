from dataclasses import dataclass, field
from xdsl.ir import Block, SSAValue, Operation
from xdsl.dialects import builtin, scf, arith, func, llvm, memref

from devito.ir.ietxdsl import iet_ssa

from xdsl.pattern_rewriter import (RewritePattern, PatternRewriter, PatternRewriteWalker,
                                   GreedyRewritePatternApplier, op_type_rewrite_pattern)


class ConvertScfForArgsToIndex(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.For, rewriter: PatternRewriter, /):
        # TODO: properly figure out why we enter an infinite loop with recursive
        # rewrites here:
        if isinstance(op.lb.type, builtin.IndexType):
            return
        for val in (op.lb, op.ub, op.step):
            cast = arith.IndexCastOp(val, builtin.IndexType())
            rewriter.insert_op_before_matched_op(cast)
            op.operands[op.operands.index(val)] = cast.result


class ConvertScfParallelArgsToIndex(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ParallelOp, rewriter: PatternRewriter,
                          /):
        # TODO: properly figure out why we enter an infinite loop with recursive
        # rewrites here:
        if isinstance(op.lowerBound[0].type, builtin.IndexType):
            return
        # increment upper bound
        cst1 = arith.Constant.from_int_and_width(1, builtin.i64)
        rewriter.insert_op_before_matched_op(cst1)
        for ub in op.upperBound:
            op.replace_operand(
                op.operands.index(ub),
                ub,
            )

        for val in (*op.lowerBound, *op.upperBound, *op.step):
            if isinstance(val.type, builtin.IndexType):
                continue
            cast = arith.IndexCastOp(val, builtin.IndexType())
            rewriter.insert_op_before_matched_op(cast)
            op.operands[op.operands.index(val)] = cast.result


class ConvertForLoopVarToIndex(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if not isinstance(op, (scf.For, scf.ParallelOp)):
            return

        block: Block = op.regions[0].blocks[0]
        # change all loop vars to index and immediately cast to i64
        for loop_var in block.args:
            if isinstance(loop_var.type, builtin.IndexType):
                continue
            loop_var.type = builtin.IndexType()
            # insert a cast from index to i64 at the start of the loop

            rewriter.insert_op_at_start(
                i64_val := arith.IndexCastOp(loop_var, builtin.i64),
                block,
            )

            loop_var.replace_by(i64_val.result)
            i64_val.operands[0] = loop_var

@dataclass
class LowerIetPointerCastAndDataObject(RewritePattern):
    dimensions: list[SSAValue] = field(default_factory=list)
    """
    This pass converts the pointer cast into an `getelementptr` operation.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: iet_ssa.PointerCast,
                          rewriter: PatternRewriter, /):
        # TODO: this can be fixed, just requires some effort.
        # if you hit this ping anton lydike :D (I should know what to do)
        assert len(self.dimensions) == 0, "can only convert one pointer cast!"

        # grab the type the pointercast returns (memref type)
        memref_typ = op.result.type
        assert isinstance(memref_typ, memref.MemRefType)

        # u_vec_size_ptr_addr : llvm.ptr<llvm.ptr<i64>>
        u_vec_size_ptr_addr = llvm.GEPOp.get(
            op.arg,
            indices=[0, 1],
            result_type=llvm.LLVMPointerType.typed(
                llvm.LLVMPointerType.typed(builtin.i64)),
        )
        # dereference the ptr ptr to get a normal pointer
        # u_vec_size_ptr : llvm.ptr<i64>
        u_vec_size_ptr = llvm.LoadOp.get(u_vec_size_ptr_addr)

        new_ops = [u_vec_size_ptr_addr, u_vec_size_ptr]

        for i in range(len(memref_typ.shape)):
            # calculate ptr = (u_vec_size_ptr + i) : llvm.ptr<i64>
            # TODO: since the actual type of `u_vec->size` is `ptr<i64>` we skip every
            #       second entry and only use the lower 32 bits of the size.
            #       this is okay as long as we don't work on data with more than
            #       2^32 cells, if each cell is 32bit long that is ~128GB of data.
            ptr_to_size_i = llvm.GEPOp.get(
                u_vec_size_ptr,
                indices=[i * 2],  # get the 2-i-th element
                result_type=llvm.LLVMPointerType.typed(builtin.i64),
            )
            # dereference
            # val = *(ptr) : i64
            load = llvm.LoadOp.get(ptr_to_size_i)
            new_ops += [ptr_to_size_i, load]
            self.dimensions.append(load.dereferenced_value)

        # insert ops to load dimensions
        rewriter.insert_op_before_matched_op(new_ops)

        # now lower the actual pointer cast
        rewriter.replace_matched_op(
            [
                # this is (u_vec + 0)
                # the type is llvm.ptr<llvm.ptr<encapsulated type>>
                u_vec_data_ptr_addr := llvm.GEPOp.get(
                    op.arg,
                    indices=[0, 0],
                    result_type=llvm.LLVMPointerType.typed(
                        llvm.LLVMPointerType.typed(memref_typ.element_type)),
                ),
                # we dereference u_vec_data_ptr_addr to get
                # a pointer to the data
                u_vec_data_ptr := llvm.LoadOp.get(u_vec_data_ptr_addr),
            ],
            [u_vec_data_ptr.dereferenced_value],
        )


def recalc_func_type(op: func.FuncOp):
    # Only if blocks exist
    if op.body.blocks:
        op.attributes['function_type'] = builtin.FunctionType.from_lists(
            [arg.type for arg in op.body.blocks[0].args],
            op.function_type.outputs.data,
        )


@dataclass
class LowerMemrefLoadToLLvmPointer(RewritePattern):
    """
    This converts a memref load %ref at %x, %y, %z, ...
    to an llvm pointer based load

    We have the dimensions in variables o1, o2, o3

    %linear_offset = x(o1 * o2) + y(o2) + z
    %elem_ptr = getelementptr %ptr, %linear_offset
    ptr.load %elem_ptr

    Why we need to convert from memref.load to llvm.load?

    Memerefs are much more complex than a pointer to contiguous memory.
    Since we are just given a pointer to contigous memory, we think it's
    easiest to manually compute these linear_offsets for accesses.
    """
    ptr_lowering: LowerIetPointerCastAndDataObject

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Load, rewriter: PatternRewriter, /):
        idx_calc_ops, idx = calc_index(op.indices,
                                       self.ptr_lowering.dimensions)

        rewriter.replace_matched_op(
            [
                *idx_calc_ops,
                gep := llvm.GEPOp.get(op.memref,
                                      indices=[llvm.GEP_USE_SSA_VAL],
                                      ssa_indices=[idx],
                                      result_type=llvm.LLVMPointerType.typed(
                                          op.memref.memref.type.element_type)),
                load := llvm.LoadOp.get(gep),
            ],
            [load.dereferenced_value],
        )


@dataclass
class LowerMemrefStoreToLLvmPointer(RewritePattern):
    ptr_lowering: LowerIetPointerCastAndDataObject

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Store, rewriter: PatternRewriter,
                          /):
        idx_calc_ops, idx = calc_index(op.indices,
                                       self.ptr_lowering.dimensions)

        rewriter.replace_matched_op(
            [
                *idx_calc_ops,
                # -2147483648 is INT_MIN, a magic value used in the implementation of GEP
                gep := llvm.GEPOp.get(
                    op.memref,
                    indices=[llvm.GEP_USE_SSA_VAL],
                    ssa_indices=[idx],
                    result_type=llvm.LLVMPointerType.typed(op.memref.memref.element_type)
                ),
                llvm.StoreOp(op.value, gep),
            ],
            [],
        )


def calc_index(indices: list[SSAValue],
               offsets: list[SSAValue]) -> list[Operation, SSAValue]:
    assert len(indices) == len(offsets)
    results = []
    # we have to convert the indices in the format
    # prt[x][y][z] with the offsets s0, s1, s2c into the format:
    # ptr[
    #   (x * s1 * s2) +
    #   (y * s2) +
    #   (z)
    # ]
    # we re-arrange this to minimize multiplications
    # ((x * s1) + y) * s2) + z
    # z + (s2 * (y + (s1 * x)))
    carry: SSAValue = indices[0]  # x

    for offset, index in zip(offsets[1:], indices[1:]):
        mul = arith.Muli.get(carry, offset)
        add = arith.Addi.get(mul, index)
        results += [mul, add]
        carry = add.result

    return results, carry


def iet_to_standard_mlir(module: builtin.ModuleOp):

    walk = PatternRewriteWalker(
        GreedyRewritePatternApplier([
            LowerIetForToScfParallel(),
            LowerIetForToScfFor(),
            ConvertScfForArgsToIndex(),
            ConvertScfParallelArgsToIndex(),
            ptr_lower := LowerIetPointerCastAndDataObject(),
            LowerMemrefLoadToLLvmPointer(ptr_lower),
            LowerMemrefStoreToLLvmPointer(ptr_lower),
        ]))

    walk.rewrite_module(module)
    PatternRewriteWalker(ConvertForLoopVarToIndex()).rewrite_module(module)
