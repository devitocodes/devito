from dataclasses import dataclass, field
from xdsl.ir import Block, SSAValue, Operation, OpResult
from xdsl.dialects import builtin, scf, arith, func, llvm, memref

from devito.ir.ietxdsl import iet_ssa

from xdsl.pattern_rewriter import (RewritePattern, PatternRewriter, PatternRewriteWalker,
                                   GreedyRewritePatternApplier, op_type_rewrite_pattern)


def _generate_subindices(subindices: int, block: Block,
                         rewriter: PatternRewriter):
    # keep track of the what argument we should replace with what
    arg_changes: list[tuple[SSAValue, SSAValue]] = []

    # keep track of the ops we want to insert
    modulo = arith.Constant.from_int_and_width(subindices, builtin.i64)
    new_ops = [modulo]

    # generate the new indices
    for i in range(subindices):
        offset = arith.Constant.from_int_and_width(i, builtin.i64)
        index_off = arith.Addi(block.args[0], offset)
        index = arith.RemSI(index_off, modulo)

        new_ops += [
            offset,
            index_off,
            index,
        ]
        # replace block.args[i+1] with (arg0 + i) % n
        arg_changes.append((block.args[i + 1], index.result))

    rewriter.insert_op_at_start(new_ops, block)

    for old, new in arg_changes:
        old.replace_by(new)
        block.erase_arg(old)


class ConvertScfForArgsToIndex(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.For, rewriter: PatternRewriter, /):
        # TODO: properly figure out why we enter an infinite loop with recursive rewrites here:
        if isinstance(op.lb.type, builtin.IndexType):
            return
        for val in (op.lb, op.ub, op.step):
            cast = arith.IndexCastOp.get(val, builtin.IndexType())
            rewriter.insert_op_before_matched_op(cast)
            op.replace_operand(op.operands.index(val), cast.result)


class ConvertScfParallelArgsToIndex(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ParallelOp, rewriter: PatternRewriter,
                          /):
        # TODO: properly figure out why we enter an infinite loop with recursive rewrites here:
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
            cast = arith.IndexCastOp.get(val, builtin.IndexType())
            rewriter.insert_op_before_matched_op(cast)
            op.replace_operand(op.operands.index(val), cast.result)


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
                i64_val := arith.IndexCastOp.get(loop_var, builtin.i64),
                block,
            )

            loop_var.replace_by(i64_val.result)
            i64_val.replace_operand(0, loop_var)


class LowerIetForToScfFor(RewritePattern):
    """
    This lowers ALL `iet.for` loops it finds to *sequential* scf.for loops
    regardless of whether the loop is declared "parallel".
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: iet_ssa.For, rewriter: PatternRewriter, /):
        body = op.body.detach_block(0)

        field_name = op.attributes['index_owner'].data

        subindice_vals = [
            iet_ssa.LoadSymbolic.get(
                f"{field_name}_{i}",
                body.args[i+1].type,
            ) for i in range(op.subindices.data)
        ]
        rewriter.insert_op_before_matched_op(subindice_vals)

        subindice_vals = list(reversed(subindice_vals))
        subindice_vals.append(subindice_vals.pop(0))

        rewriter.replace_matched_op([
            cst1    := arith.Constant.from_int_and_width(1, builtin.IndexType()),
            new_ub  := arith.Addi(op.ub, cst1),
            scf_for := scf.For.get(op.lb, new_ub.result, op.step, subindice_vals, body),
        ], [scf_for.results[0]])

        for use in scf_for.results[0].uses:
            if isinstance(use.operation, func.Return):
                assert isinstance(use.operation.parent_op(), func.FuncOp)
                use.operation.parent_op().update_function_type()


class LowerIetForToScfParallel(RewritePattern):
    """
    This converts all loops with a "parallel" property to `scf.parallel`

    It does not currently fuse together multiple nested parallel runs
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: iet_ssa.For, rewriter: PatternRewriter, /):
        if op.parallelism_property != 'parallel':
            return

        body: Block = op.body.detach_block(0)

        assert op.subindices.data == 0

        lb, ub, step, new_body = self.recurse_scf_parallel([op.lb], [op.ub],
                                                           [op.step], body,
                                                           rewriter)

        # insert scf.yield at the bacl
        rewriter.insert_op_at_pos(scf.Yield(), new_body, len(new_body.ops))

        rewriter.replace_matched_op(
            par_op := scf.ParallelOp.get(lb, ub, step, [new_body])
        )

        cst1 = arith.Constant.from_int_and_width(1, builtin.IndexType())
        rewriter.insert_op_before_matched_op(cst1)
        for ub_val in ub:
            rewriter.insert_op_before_matched_op(
                new_ub := arith.Addi.get(ub_val, cst1)
            )
            par_op.replace_operand(
                par_op.operands.index(ub_val),
                new_ub.result
            )

    def recurse_scf_parallel(
        self,
        lbs: list[SSAValue],
        ubs: list[SSAValue],
        steps: list[SSAValue],
        body: Block,
        rewriter: PatternRewriter,
    ):
        if not (len(body.ops) == 2 and isinstance(body.ops[1], iet_ssa.For)
                and body.ops[1].parallelism_property == 'parallel'):
            return lbs, ubs, steps, body
        inner_for = body.ops[1]
        # re-use step
        step = inner_for.step
        if isinstance(inner_for.step, OpResult) and isinstance(
                inner_for.step.op, arith.Constant):
            if inner_for.step.op.value.value.data == steps[
                    0].op.value.value.data:
                step = steps[0]
                if len(inner_for.step.uses) > 1:
                    assert False, "TODO: move op out of loop"

        steps.append(step)
        lbs.append(inner_for.lb)
        ubs.append(inner_for.ub)

        new_body = inner_for.body.detach_block(0)
        for arg in body.args:
            new_body.insert_arg(arg.type, arg.index)
            arg.replace_by(new_body.args[arg.index])

        body = new_body

        lbs, ubs, steps, body = self.recurse_scf_parallel(
            lbs, ubs, steps, body, rewriter)

        return lbs, ubs, steps, body


class DropIetComments(RewritePattern):
    """
    This drops all iet.comment operations

    TODO: convert iet.comment ops that have timer info into their own nodes
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: iet_ssa.Statement,
                          rewriter: PatternRewriter, /):
        rewriter.erase_matched_op()


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


class CleanupDanglingIetDatatypes(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /):
        for i, arg_typ in enumerate(op.function_type.inputs.data):
            if isinstance(arg_typ, iet_ssa.Dataobj):
                op.body.blocks[0].args[i].type = llvm.LLVMPointerType.typed(
                    iet_ssa.Dataobj.get_llvm_struct_type(), )
            elif isinstance(arg_typ, iet_ssa.Profiler):
                op.body.blocks[0].args[i].type = llvm.LLVMPointerType.opaque()

        recalc_func_type(op)


def recalc_func_type(op: func.FuncOp):
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
                store := llvm.StoreOp(op.value, gep),
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
            DropIetComments(),
            CleanupDanglingIetDatatypes(),
            ptr_lower := LowerIetPointerCastAndDataObject(),
            LowerMemrefLoadToLLvmPointer(ptr_lower),
            LowerMemrefStoreToLLvmPointer(ptr_lower),
        ]))

    walk.rewrite_module(module)
    PatternRewriteWalker(ConvertForLoopVarToIndex()).rewrite_module(module)
