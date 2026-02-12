"""
Optimized MatMul Sketch for shape (2048, 1024) * (1024, 1536) on Ascend 910B2

============================== OPTIMIZATION ANALYSIS ==============================

Parent Solution (0.96x, 84.42us vs 81.42us baseline):
  - BLOCK_M=128, BLOCK_N=128, BLOCK_K=128
  - Diagonal scheduling with threshold=8 (complex GCD/LCM in kernel)
  - Single autotune config
  - Conditional branching for large/small shapes

Key Optimizations Applied:
  1. Swizzle2D block reordering: Replaces complex diagonal scheduling with
     proven swizzle2d grouping for better L2 cache locality. GROUP_SIZE
     controls how many consecutive M-blocks share B-tile data in L2.
  2. Larger BLOCK_K=256: Fully utilizes L0A (128*256*2=64KB) and L0B
     (256*128*2=64KB), reduces K-loop from 8 to 4 iterations.
  3. Multiple autotune configs: Search across tile sizes and GROUP_SIZE values.
  4. Removed conditional branching: Shape (2048,1536) is always "large".
  5. DIRECTION=0 (row-first): Since M(2048) > N(1536), group by rows to
     maximize B-tile reuse in L2 cache.

Hardware Constraints Verification (Ascend 910B2, fp16):
  Config A: BLOCK_M=128, BLOCK_K=256, BLOCK_N=128
    L0A: 128*256*2 = 64KB  <= 64KB ✓
    L0B: 256*128*2 = 64KB  <= 64KB ✓
    L0C: 128*128*4 = 64KB  <= 128KB ✓  (fp32 accumulator)
    Total blocks: 16*12=192, per core=8, K-iters=4

  Config B: BLOCK_M=128, BLOCK_K=128, BLOCK_N=256
    L0A: 128*128*2 = 32KB  <= 64KB ✓
    L0B: 128*256*2 = 64KB  <= 64KB ✓
    L0C: 128*256*4 = 128KB <= 128KB ✓
    Total blocks: 16*6=96, per core=4, K-iters=8

  Config C: BLOCK_M=256, BLOCK_K=128, BLOCK_N=128
    L0A: 256*128*2 = 64KB  <= 64KB ✓
    L0B: 128*128*2 = 32KB  <= 64KB ✓
    L0C: 256*128*4 = 128KB <= 128KB ✓
    Total blocks: 8*12=96, per core=4, K-iters=8

  Config D: BLOCK_M=128, BLOCK_K=128, BLOCK_N=128  (parent baseline)
    L0A: 128*128*2 = 32KB  <= 64KB ✓
    L0B: 128*128*2 = 32KB  <= 64KB ✓
    L0C: 128*128*4 = 64KB  <= 128KB ✓
    Total blocks: 16*12=192, per core=8, K-iters=8

====================================================================================
"""

# ========================== SKETCH ==========================

SKETCH = """
sketch matmul {
  symbols: M, N, K;
  tensors: A[M, K]: f16; B[K, N]: f16; C[M, N]: f16;
  constexpr: BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE;

  NUM_BLOCKS_M = ceil(M, BLOCK_M);
  NUM_BLOCKS_N = ceil(N, BLOCK_N);
  NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N;

  # M=2048 > N=1536, 选择 DIRECTION=0 (行优先分组)
  # 行优先意味着相邻核心处理相同行不同列的块，共享A行数据，B列数据在L2中被复用
  DIRECTION = 0;

  # 24核固定并行，Swizzle2D块分组重排
  @llm_hint("parallel", "coreidx")
  for core_id in range(24):
    for block_idx in range(core_id, NUM_BLOCKS, 24):
      block_m = block_idx // NUM_BLOCKS_N;
      block_n = block_idx % NUM_BLOCKS_N;

      # Swizzle2D: 行优先分组，GROUP_SIZE个连续M-block组成一组
      # 组内块在N方向连续排列，共享同一组A行数据，提升L2缓存命中率
      task_m_idx, task_n_idx = swizzle2d(
        block_m, block_n,
        NUM_BLOCKS_M, NUM_BLOCKS_N,
        GROUP_SIZE
      );

      m_start = task_m_idx * BLOCK_M;
      n_start = task_n_idx * BLOCK_N;

      # L0C累加器 (fp32精度累加)
      c_tile = alloc([BLOCK_M, BLOCK_N], llm_hint=["fastest", "accumulator", "init_zero"]);

      @llm_hint("pipeline")
      for k_start in range(0, K, BLOCK_K):
        # L0A ← A子块 (通过L1缓存)
        a_tile = alloc([BLOCK_M, BLOCK_K], llm_hint=["fast", "input_cache"]);
        load(A[m_start:m_start+BLOCK_M, k_start:k_start+BLOCK_K] -> a_tile);

        # L0B ← B子块 (通过L1缓存)
        b_tile = alloc([BLOCK_K, BLOCK_N], llm_hint=["fast", "input_cache"]);
        load(B[k_start:k_start+BLOCK_K, n_start:n_start+BLOCK_N] -> b_tile);

        # CUBE GEMM: c_tile += a_tile @ b_tile
        gemm(a_tile, b_tile, dst=c_tile);

      # FixP: L0C→GM 并 FP32→FP16
      store(c_tile -> C[m_start:m_start+BLOCK_M, n_start:n_start+BLOCK_N]);
}

@llm_hint("available_tiling")
available_tiling:
  BLOCK_M=128, BLOCK_K=256, BLOCK_N=128, GROUP_SIZE=4
  BLOCK_M=128, BLOCK_K=256, BLOCK_N=128, GROUP_SIZE=2
  BLOCK_M=128, BLOCK_K=256, BLOCK_N=128, GROUP_SIZE=8
  BLOCK_M=128, BLOCK_K=128, BLOCK_N=256, GROUP_SIZE=4
  BLOCK_M=128, BLOCK_K=128, BLOCK_N=256, GROUP_SIZE=2
  BLOCK_M=256, BLOCK_K=128, BLOCK_N=128, GROUP_SIZE=4
  BLOCK_M=256, BLOCK_K=128, BLOCK_N=128, GROUP_SIZE=2
  BLOCK_M=128, BLOCK_K=128, BLOCK_N=128, GROUP_SIZE=4
  BLOCK_M=128, BLOCK_K=128, BLOCK_N=128, GROUP_SIZE=2
"""


# ========================== IMPLEMENTATION ==========================

IMPLEMENTATION = """
import torch
import torch_npu
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Primary config: larger K-tile for fewer iterations
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_SIZE": 4}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_SIZE": 2}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_SIZE": 8}),
        # Wider N-tile: fewer total blocks, larger per-block compute
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE": 4}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE": 2}),
        # Taller M-tile
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE": 4}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE": 2}),
        # Baseline tile size with swizzle (replaces diagonal)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE": 4}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE": 2}),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel_swizzle2d(
        A, B, C,
        M, N, K,
        num_cores,
        NUM_BLOCKS_M,
        NUM_BLOCKS_N,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N

        # Swizzle2D: row-first grouping (M >= N)
        # Groups GROUP_SIZE consecutive M-blocks, interleaving N within each group
        # This improves L2 cache locality for B-tile reuse
        task_m_idx, task_n_idx = tl.swizzle2d(
            block_m, block_n,
            NUM_BLOCKS_M, NUM_BLOCKS_N,
            GROUP_SIZE
        )

        m_start = task_m_idx * BLOCK_M
        n_start = task_n_idx * BLOCK_N

        # L0C accumulator initialized to zero (fp32 precision)
        c_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            # Load A sub-block: GM -> L1 -> L0A
            a_offs = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + \\
                     (k_start + tl.arange(0, BLOCK_K))[None, :]
            a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & \\
                     ((k_start + tl.arange(0, BLOCK_K)) < K)[None, :]
            a_tile = tl.load(A + a_offs, mask=a_mask, other=0.0)
            tl.compile_hint(a_tile, "dot_pad_only_k")

            # Load B sub-block: GM -> L1 -> L0B
            b_offs = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + \\
                     (n_start + tl.arange(0, BLOCK_N))[None, :]
            b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & \\
                     ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
            b_tile = tl.load(B + b_offs, mask=b_mask, other=0.0)
            tl.compile_hint(b_tile, "dot_pad_only_k")

            # CUBE GEMM: c_tile += a_tile @ b_tile
            c_tile = tl.dot(a_tile, b_tile, c_tile)

        # Store result: L0C -> GM with FP32 -> FP16 cast
        c_offs = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + \\
                 (n_start + tl.arange(0, BLOCK_N))[None, :]
        c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & \\
                 ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
        tl.store(C + c_offs, c_tile.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # x: (2048, 1024), y: (1024, 1536)
        M, K = x.shape
        K2, N = y.shape
        assert K == K2, "matmul shape mismatch"

        c = torch.empty((M, N), dtype=x.dtype, device=x.device)
        num_cores = 24  # Ascend 910B2 has 24 AI Cores

        # Pre-compute block counts on host
        # Use max possible block size for cdiv (autotune will pick actual values)
        NUM_BLOCKS_M = triton.cdiv(M, 128)  # will be adjusted by autotune
        NUM_BLOCKS_N = triton.cdiv(N, 128)

        matmul_kernel_swizzle2d[(num_cores,)](
            x, y, c,
            M, N, K,
            num_cores,
            NUM_BLOCKS_M,
            NUM_BLOCKS_N,
        )
        return c
"""


# ========================== CORRECTED IMPLEMENTATION ==========================
# Note: NUM_BLOCKS_M/N must be computed based on actual BLOCK_M/N from autotune

IMPLEMENTATION_V2 = """
import torch
import torch_npu
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Primary: larger K-tile reduces K-loop from 8 to 4 iterations
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_SIZE": 4}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_SIZE": 2}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_SIZE": 8}),
        # Wider N-tile
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE": 4}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE": 2}),
        # Taller M-tile
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE": 4}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE": 2}),
        # Baseline with swizzle
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE": 4}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE": 2}),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel_swizzle2d(
        A, B, C,
        M, N, K,
        num_cores,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N

        # Swizzle2D row-first grouping (M >= N)
        task_m_idx, task_n_idx = tl.swizzle2d(
            block_m, block_n,
            NUM_BLOCKS_M, NUM_BLOCKS_N,
            GROUP_SIZE
        )

        m_start = task_m_idx * BLOCK_M
        n_start = task_n_idx * BLOCK_N

        # L0C accumulator (fp32)
        c_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            # Load A[m_start:m_start+BLOCK_M, k_start:k_start+BLOCK_K]
            a_offs = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + \\
                     (k_start + tl.arange(0, BLOCK_K))[None, :]
            a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & \\
                     ((k_start + tl.arange(0, BLOCK_K)) < K)[None, :]
            a_tile = tl.load(A + a_offs, mask=a_mask, other=0.0)
            tl.compile_hint(a_tile, "dot_pad_only_k")

            # Load B[k_start:k_start+BLOCK_K, n_start:n_start+BLOCK_N]
            b_offs = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + \\
                     (n_start + tl.arange(0, BLOCK_N))[None, :]
            b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & \\
                     ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
            b_tile = tl.load(B + b_offs, mask=b_mask, other=0.0)
            tl.compile_hint(b_tile, "dot_pad_only_k")

            # CUBE GEMM
            c_tile = tl.dot(a_tile, b_tile, c_tile)

        # Store with FP32 -> FP16 cast
        c_offs = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + \\
                 (n_start + tl.arange(0, BLOCK_N))[None, :]
        c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & \\
                 ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
        tl.store(C + c_offs, c_tile.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # x: (2048, 1024), y: (1024, 1536)
        M, K = x.shape
        K2, N = y.shape
        assert K == K2, "matmul shape mismatch"

        c = torch.empty((M, N), dtype=x.dtype, device=x.device)
        num_cores = 24  # Ascend 910B2: 24 AI Cores

        matmul_kernel_swizzle2d[(num_cores,)](
            x, y, c,
            M, N, K,
            num_cores,
        )
        return c
"""

print("Sketch and implementation defined successfully.")
print("See SKETCH variable for the optimized algorithm sketch.")
print("See IMPLEMENTATION_V2 variable for the corrected Triton implementation.")
