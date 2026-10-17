#include "gac/VortexTcu.h"
#include "gac/DeviceLowering.h"
#include "gac/VortexMatmul.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <array>

using namespace mlir;
namespace gac {
namespace {
constexpr int64_t lanes = 4, tileM = 8, tileN = 4, tileK = 4;

// Inverse register permutations from vx_tensor.h::wmma_context::mma_sync,
// dense Pattern A (b_sub_blocks == 1), wmma_config_t<4, fp32, fp32>.
// A physical f10..f17; B f28..f31; C/D f0..f7. These are NOT row-major
// arrays in the register file: every register contains one element per lane.
constexpr std::array<unsigned, 8> aPermutation{4, 5, 0, 1, 6, 7, 2, 3};
constexpr std::array<unsigned, 4> bPermutation{3, 0, 2, 1};
constexpr std::array<unsigned, 8> cPermutation{4, 5, 1, 0, 6, 7, 3, 2};

struct Emitter {
  OpBuilder &b;
  Location loc;
  ModuleOp module;
  Value i(int64_t value) {
    return b.create<arith::ConstantIntOp>(loc, value, 32);
  }
  Value zero() {
    return b.create<arith::ConstantFloatOp>(loc, APFloat(0.0f), b.getF32Type());
  }
  Value add(Value a, Value c) { return b.create<arith::AddIOp>(loc, a, c); }
  Value mul(Value a, Value c) { return b.create<arith::MulIOp>(loc, a, c); }
  Value div(Value a, int64_t d) {
    return b.create<arith::DivUIOp>(loc, a, i(d));
  }
  Value rem(Value a, int64_t d) {
    return b.create<arith::RemUIOp>(loc, a, i(d));
  }
  Value inside(Value row, int64_t rows, Value col, int64_t cols) {
    auto a = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, row, i(rows));
    auto c = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, col, i(cols));
    return b.create<arith::AndIOp>(loc, a, c);
  }
  Value address(Value pointer, Value offset) {
    return b.create<LLVM::GEPOp>(loc, pointer.getType(), b.getF32Type(),
                                pointer, ValueRange{offset});
  }
  Value load(Value pointer, Value offset) {
    return b.create<LLVM::LoadOp>(loc, b.getF32Type(), address(pointer, offset));
  }
  Value guardedLoad(Value pointer, Value offset, Value valid) {
    auto branch = b.create<scf::IfOp>(loc, TypeRange{b.getF32Type()}, valid, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(branch.thenBlock());
      b.create<scf::YieldOp>(loc, load(pointer, offset));
      b.setInsertionPointToStart(branch.elseBlock());
      b.create<scf::YieldOp>(loc, zero());
    }
    return branch.getResult(0);
  }
  Value target(StringRef name) {
    OperationState state(loc, name);
    state.addTypes(b.getI32Type());
    return b.create(state)->getResult(0);
  }
  // Batch broadcasting is resolved from the static tensor shapes; no host
  // tensor packing or separately generated C++ device code is involved.
  Value batchIndex(Value batch, ArrayRef<int64_t> output,
                   ArrayRef<int64_t> input) {
    Value result = i(0);
    int64_t stride = 1;
    for (int64_t dim : output)
      stride *= dim;
    for (size_t d = 0; d < output.size(); ++d) {
      stride /= output[d];
      if (d + input.size() < output.size())
        continue;
      int64_t size = input[d + input.size() - output.size()];
      Value coordinate = size == 1 ? i(0) : rem(div(batch, stride), output[d]);
      result = add(mul(result, i(size)), coordinate);
    }
    return result;
  }
  Value biasIndex(Value outputIndex, const VortexMatmulPlan &plan) {
    return batchIndex(outputIndex, plan.outShape, plan.biasShape);
  }
  // Vortex TF32 is a compact 19-bit format, NOT an FP32 word with low bits
  // cleared: sign=18, exponent=17:10, mantissa=9:0 (tensor_cfg/VX_tcu_pkg).
  // Bitcasts keep that packed word intact while LLVM places it in an FP reg.
  Value packTf32(Value value) {
    Value bits = b.create<arith::BitcastOp>(loc, b.getI32Type(), value);
    Value packed = b.create<arith::ShRUIOp>(loc, bits, i(13));
    return b.create<arith::BitcastOp>(loc, b.getF32Type(), packed);
  }
  void splitTf32(ArrayRef<Value> values, SmallVectorImpl<Value> &hi,
                 SmallVectorImpl<Value> &lo) {
    for (Value value : values) {
      Value bits = b.create<arith::BitcastOp>(loc, b.getI32Type(), value);
      Value highBits = b.create<arith::AndIOp>(loc, bits, i(-8192));
      Value high = b.create<arith::BitcastOp>(loc, b.getF32Type(), highBits);
      Value residual = b.create<arith::SubFOp>(loc, value, high);
      hi.push_back(packTf32(high));
      lo.push_back(packTf32(residual));
    }
  }
  Value repairNonFinite(Value accumulated, Value a, Value c, Value aOffset,
                        Value bOffset, Value row, Value col,
                        const VortexMatmulPlan &plan) {
    Value bits = b.create<arith::BitcastOp>(loc, b.getI32Type(), accumulated);
    Value exponent = b.create<arith::AndIOp>(loc, bits, i(0x7f800000));
    Value exceptional = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, exponent, i(0x7f800000));
    auto branch = b.create<scf::IfOp>(loc, TypeRange{b.getF32Type()}, exceptional, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(branch.thenBlock());
      // This runs after the last collective MMA, so per-output divergence is
      // safe. No Host fallback: repeat the original dot product on device FPU.
      auto exact = b.create<scf::ForOp>(
          loc, i(0), i(plan.k), i(1), ValueRange{zero()},
          [&](OpBuilder &, Location, Value k, ValueRange sum) {
            Value left = load(a, add(aOffset, add(mul(row, i(plan.k)), k)));
            Value right = load(c, add(bOffset, add(mul(k, i(plan.n)), col)));
            Value product = b.create<arith::MulFOp>(loc, left, right);
            Value next = b.create<arith::AddFOp>(loc, sum[0], product);
            b.create<scf::YieldOp>(loc, next);
          });
      b.create<scf::YieldOp>(loc, exact.getResult(0));
      b.setInsertionPointToStart(branch.elseBlock());
      b.create<scf::YieldOp>(loc, accumulated);
    }
    return branch.getResult(0);
  }
  Value instruction(Type result, ValueRange operands) {
    OperationState state(loc, "llvm.inline_asm");
    state.addTypes(result);
    state.addOperands(operands);
    // custom-0 / funct3=0 / funct7=2; TF32 input id=1, FP32 output id=0.
    // Every lane participates, including zero-filled out-of-bounds lanes.
    state.addAttribute("asm_string",
                       b.getStringAttr(".insn r 0x0b, 0, 2, x0, x1, x0"));
    state.addAttribute("constraints", b.getStringAttr(
        "={f0},={f1},={f2},={f3},={f4},={f5},={f6},={f7},"
        "0,1,2,3,4,5,6,7,"
        "{f10},{f11},{f12},{f13},{f14},{f15},{f16},{f17},"
        "{f28},{f29},{f30},{f31}"));
    state.addAttribute("has_side_effects", b.getUnitAttr());
    return b.create(state)->getResult(0);
  }
  SmallVector<Value> unpack(Value packed) {
    SmallVector<Value> result(8);
    for (unsigned physical = 0; physical != 8; ++physical)
      result[cPermutation[physical]] = b.create<LLVM::ExtractValueOp>(
          loc, packed, ArrayRef<int64_t>{int64_t(physical)});
    return result;
  }
  SmallVector<Value> mma(ArrayRef<Value> a, ArrayRef<Value> c,
                         ValueRange accumulator) {
    // The instruction has implicit fixed FP-register operands. Explicit LLVM
    // constraints expose EVERY read/write to register allocation; an unadorned
    // .insn would silently clobber live values and is not a valid lowering.
    SmallVector<Value> operands;
    for (unsigned index : cPermutation)
      operands.push_back(accumulator[index]);
    for (unsigned index : aPermutation)
      operands.push_back(a[index]);
    for (unsigned index : bPermutation)
      operands.push_back(c[index]);
    constexpr StringLiteral name = "__gac_tcu_wmma_tf32_nt4";
    auto helper = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
    if (!helper) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(module.getBody());
      SmallVector<Type> fields(8, b.getF32Type()), args(20, b.getF32Type());
      Type result = LLVM::LLVMStructType::getLiteral(b.getContext(), fields);
      helper = b.create<LLVM::LLVMFuncOp>(
          loc, name, LLVM::LLVMFunctionType::get(result, args),
          LLVM::Linkage::Internal);
      // LLVM 20's InlineAsmOp cannot express convergence. Keep the instruction
      // inside a noinline/optnone helper and mark the call site convergent, so
      // optimizers cannot move it through divergent branches or duplicate it
      // for subsets of a warp. Only this tiny instruction wrapper is optnone.
      helper->setAttr("passthrough", b.getArrayAttr({
          b.getStringAttr("convergent"), b.getStringAttr("noinline"),
          b.getStringAttr("optnone")}));
      Block *entry = helper.addEntryBlock(b);
      b.setInsertionPointToStart(entry);
      b.create<LLVM::ReturnOp>(loc, instruction(result, entry->getArguments()));
    }
    auto call = b.create<LLVM::CallOp>(loc, helper, operands);
    call->setAttr("convergent", b.getUnitAttr());
    return unpack(call.getResult());
  }
  SmallVector<Value> mmaCombined(ArrayRef<Value> aHi, ArrayRef<Value> aLo,
                                 ArrayRef<Value> bHi, ArrayRef<Value> bLo,
                                 ValueRange accumulator) {
    SmallVector<Value> operands;
    for (unsigned index : cPermutation)
      operands.push_back(accumulator[index]);
    for (auto fragment : {aHi, aLo})
      for (unsigned index : aPermutation)
        operands.push_back(fragment[index]);
    for (auto fragment : {bHi, bLo})
      for (unsigned index : bPermutation)
        operands.push_back(fragment[index]);
    constexpr StringLiteral name = "__gac_tcu_wmma_tf32x3_nt4";
    auto helper = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
    if (!helper) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(module.getBody());
      SmallVector<Type> fields(8, b.getF32Type()), args(32, b.getF32Type());
      Type result = LLVM::LLVMStructType::getLiteral(b.getContext(), fields);
      helper = b.create<LLVM::LLVMFuncOp>(
          loc, name, LLVM::LLVMFunctionType::get(result, args),
          LLVM::Linkage::Internal);
      helper->setAttr("passthrough", b.getArrayAttr({
          b.getStringAttr("convergent"), b.getStringAttr("noinline"),
          b.getStringAttr("optnone")}));
      Block *entry = helper.addEntryBlock(b);
      b.setInsertionPointToStart(entry);
      ValueRange arguments = entry->getArguments();
      SmallVector<Value> carry(arguments.take_front(8));
      // Keep all three collective instructions inside one convergence boundary.
      // The intermediate C fragments are SSA values inside this helper, so they
      // cross the function return ABI once rather than after each instruction.
      // This is an inline-assembly wrapper experiment, not a formal intrinsic.
      for (unsigned step = 0; step != 3; ++step) {
        SmallVector<Value> values(carry);
        auto a = arguments.slice(step == 1 ? 16 : 8, 8);
        auto c = arguments.slice(step == 0 ? 28 : 24, 4);
        values.append(a.begin(), a.end());
        values.append(c.begin(), c.end());
        Value packed = instruction(result, values);
        if (step == 2) {
          b.create<LLVM::ReturnOp>(loc, packed);
        } else {
          carry.clear();
          for (int64_t physical = 0; physical != 8; ++physical)
            carry.push_back(b.create<LLVM::ExtractValueOp>(
                loc, packed, ArrayRef<int64_t>{physical}));
        }
      }
    }
    auto call = b.create<LLVM::CallOp>(loc, helper, operands);
    call->setAttr("convergent", b.getUnitAttr());
    return unpack(call.getResult());
  }
};
} // namespace

bool configureVortexTcuMatmul(const DeviceOptions &options,
                             VortexMatmulPlan &plan,
                             std::string &fallbackReason) {
  if (options.vortexLanes != lanes) {
    fallbackReason = "TF32x3 WMMA currently requires exactly four hardware lanes";
    return false;
  }
  plan.schedule = "tcu";
  plan.tcuCombined = options.vortexTcuCombined;
  plan.tileM = tileM;
  plan.tileN = tileN;
  plan.tileK = tileK;
  plan.lanes = plan.blockSize = lanes;
  int64_t rows = (plan.m + tileM - 1) / tileM;
  int64_t cols = (plan.n + tileN - 1) / tileN;
  if (plan.batches <= 0 || rows <= 0 || cols <= 0 ||
      rows > INT32_MAX / cols || plan.batches > INT32_MAX / (rows * cols)) {
    fallbackReason = "WMMA grid exceeds signed 32-bit launch limit";
    return false;
  }
  plan.gridSize = plan.batches * rows * cols;
  plan.lmemBytes = 0;
  plan.stages = 0;
  return true;
}

LogicalResult emitVortexTcuMatmul(OpBuilder &builder, Operation *op,
                                 ArrayRef<Value> pointers,
                                 const VortexMatmulPlan &plan) {
  if (plan.lanes != lanes || plan.blockSize != lanes ||
      plan.tileM != tileM || plan.tileN != tileN || plan.tileK != tileK ||
      pointers.size() != op->getNumOperands() + 1u)
    return op->emitError("invalid four-lane TF32x3 WMMA plan or pointer ABI");
  // scf::ForOp builds its callback before attaching the new operation. Resolve
  // the module from the original, already-attached tensor op, not the builder's
  // temporary loop block when the WMMA helper is first materialized.
  Emitter e{builder, op->getLoc(), op->getParentOfType<ModuleOp>()};
  auto &b = builder;
  Value lane = e.target("vortex.thread_id");
  Value block = e.target("vortex.block_id");
  int64_t mTiles = (plan.m + tileM - 1) / tileM;
  int64_t nTiles = (plan.n + tileN - 1) / tileN;
  Value batch = e.div(block, mTiles * nTiles);
  Value rowBase = e.mul(e.rem(e.div(block, nTiles), mTiles), e.i(tileM));
  Value colBase = e.mul(e.rem(block, nTiles), e.i(tileN));
  Value aBatch = e.i(0), bBatch = e.i(0);
  if (!plan.flattenedProjection) {
    auto outputBatch = ArrayRef<int64_t>(plan.outShape).drop_back(2);
    aBatch = e.batchIndex(batch, outputBatch,
                          ArrayRef<int64_t>(plan.aShape).drop_back(2));
    bBatch = e.batchIndex(batch, outputBatch,
                          ArrayRef<int64_t>(plan.bShape).drop_back(2));
  }
  Value aOffset = e.mul(aBatch, e.i(plan.m * plan.k));
  Value bOffset = e.mul(bBatch, e.i(plan.k * plan.n));
  Value laneRow = e.div(lane, 2), laneCol = e.rem(lane, 2);
  SmallVector<Value> initial(8, e.zero());
  auto reduction = b.create<scf::ForOp>(
      e.loc, e.i(0), e.i(plan.k), e.i(tileK), initial,
      [&](OpBuilder &, Location, Value kBase, ValueRange accumulator) {
        SmallVector<Value> a, c;
        for (unsigned r = 0; r != 8; ++r) {
          Value row = e.add(rowBase, e.add(laneRow, e.i((r / 2) * 2)));
          Value k = e.add(kBase, e.add(laneCol, e.i((r % 2) * 2)));
          Value index = e.add(aOffset, e.add(e.mul(row, e.i(plan.k)), k));
          a.push_back(e.guardedLoad(pointers[0], index,
                                    e.inside(row, plan.m, k, plan.k)));
        }
        for (unsigned r = 0; r != 4; ++r) {
          Value k = e.add(kBase, e.add(laneCol, e.i((r / 2) * 2)));
          Value col = e.add(colBase, e.add(laneRow, e.i((r % 2) * 2)));
          Value index = e.add(bOffset, e.add(e.mul(k, e.i(plan.n)), col));
          c.push_back(e.guardedLoad(pointers[1], index,
                                    e.inside(k, plan.k, col, plan.n)));
        }
        // Actual TCU instructions, with explicit precision reduction. Compute
        // small cross terms before the high product; no host-side arithmetic.
        SmallVector<Value> aHi, aLo, bHi, bLo;
        e.splitTf32(a, aHi, aLo);
        e.splitTf32(c, bHi, bLo);
        if (plan.tcuCombined) {
          b.create<scf::YieldOp>(e.loc,
                               e.mmaCombined(aHi, aLo, bHi, bLo, accumulator));
        } else {
          auto cross = e.mma(aHi, bLo, accumulator);
          cross = e.mma(aLo, bHi, cross);
          b.create<scf::YieldOp>(e.loc, e.mma(aHi, bHi, cross));
        }
      });
  for (unsigned r = 0; r != 8; ++r) {
    Value row = e.add(rowBase, e.add(laneRow, e.i((r / 2) * 2)));
    Value col = e.add(colBase, e.add(laneCol, e.i((r % 2) * 2)));
    Value index = e.add(e.mul(e.add(e.mul(batch, e.i(plan.m)), row),
                             e.i(plan.n)), col);
    b.create<scf::IfOp>(e.loc, e.inside(row, plan.m, col, plan.n),
                        [&](OpBuilder &, Location) {
      Value result = e.repairNonFinite(reduction.getResult(r), pointers[0],
                                      pointers[1], aOffset, bOffset, row, col,
                                      plan);
      if (op->getNumOperands() == 3) {
        Value bias = e.load(pointers[2], e.biasIndex(index, plan));
        result = plan.biasFirst
                     ? b.create<arith::AddFOp>(e.loc, bias, result)
                     : b.create<arith::AddFOp>(e.loc, result, bias);
      }
      if (plan.relu) {
        Value zero = e.zero();
        Value negative = b.create<arith::CmpFOp>(
            e.loc, arith::CmpFPredicate::OLT, result, zero);
        result = b.create<arith::SelectOp>(e.loc, negative, zero, result);
      }
      b.create<LLVM::StoreOp>(e.loc, result, e.address(pointers.back(), index));
      b.create<scf::YieldOp>(e.loc);
    });
  }
  return success();
}
} // namespace gac
