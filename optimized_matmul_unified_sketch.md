# Ascend 910B2 MatMul 优化草图（UnifiedSketch）

## 1) 针对父代方案的可优化点分析

父代方案（84.4us，慢于基线）已经有两个正确方向：固定 core 数启动、核内循环处理多个 block。  
但主要问题在于：

1. **切分过于保守**：仅 `128x128x128`，没有充分利用 910B2 的 L0C（128KB）与 L0B/L0A（64KB）容量边界。
2. **调度映射开销偏高**：手写复杂“对角线映射”分支，整数运算与条件较重，且可读性和稳定性较差。
3. **缺少 shape 自适应分组复用**：没有根据 `M/N` 比例切换行优先或列优先分组。
4. **K 轴流水粒度单一**：只有单层 K 循环，缺少额外分块/展开以帮助搬运计算重叠。

## 2) 优化策略摘要

- **固定 24 AI Core 启动**：`grid=(24,)`，每核步长循环遍历任务。
- **Swizzle2D 分组重排**：根据 `M/N` 自适应方向，提升 A 或 B 的局部复用率。
- **面向 910B2 容量约束的 tile 候选**：
  - `128x256x128`（L0C 128KB 打满）
  - `128x128x256`（L0A/L0B 64KB 打满）
  - `256x128x128`（适合 M 方向较大时提高并行块粒度）
- **K 轴两级循环（K_PIPE）**：增加核内流水层次，隐藏搬运开销。
- **FP32 累加 + 可配置输出精度**：兼顾吞吐和精度。

## 3) 优化后的 UnifiedSketch

```python
sketch matmul_ascend_swizzle {
  symbols: M, N, K;
  tensors: A[M, K]: f16; B[K, N]: f16; C[M, N]: f16;
  constexpr: NUM_CORES=24, BLOCK_M=128, BLOCK_N=256, BLOCK_K=128, GROUP_SIZE=4, K_PIPE=2;

  NUM_BLOCKS_M = ceil(M, BLOCK_M);
  NUM_BLOCKS_N = ceil(N, BLOCK_N);
  NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N;
  DIRECTION = 1 if M < N else 0;    # 0: 行优先分组；1: 列优先分组

  @llm_hint("parallel", "coreidx")
  for core_id in range(NUM_CORES):
    for block_id in range(core_id, NUM_BLOCKS, NUM_CORES):
      block_m = block_id // NUM_BLOCKS_N;
      block_n = block_id % NUM_BLOCKS_N;

      # shape-adaptive swizzle 映射
      if DIRECTION == 0:
        task_m, task_n = swizzle2d(block_m, block_n, NUM_BLOCKS_M, NUM_BLOCKS_N, GROUP_SIZE);
      else:
        group_span = GROUP_SIZE * NUM_BLOCKS_M;
        group_id = block_id // group_span;
        n_group_start = group_id * GROUP_SIZE;
        n_group_size = min(GROUP_SIZE, NUM_BLOCKS_N - n_group_start);
        local_idx = block_id % group_span;
        task_m = local_idx // n_group_size;
        task_n = n_group_start + (local_idx % n_group_size);

      m_start = task_m * BLOCK_M;
      n_start = task_n * BLOCK_N;

      c_tile = alloc([BLOCK_M, BLOCK_N], llm_hint=["fastest", "accumulator", "init_zero"]);

      # 两级K循环，增强搬运/计算流水
      @llm_hint("pipeline")
      for k_outer in range(0, ceil(K, BLOCK_K * K_PIPE)):
        @llm_hint("unroll")
        for k_inner in range(0, K_PIPE):
          k_start = k_outer * BLOCK_K * K_PIPE + k_inner * BLOCK_K;

          a_tile = alloc([BLOCK_M, BLOCK_K], llm_hint=["fast", "input_cache"]);
          b_tile = alloc([BLOCK_K, BLOCK_N], llm_hint=["fast", "input_cache"]);

          load(A[m_start:m_start+BLOCK_M, k_start:k_start+BLOCK_K] -> a_tile);
          load(B[k_start:k_start+BLOCK_K, n_start:n_start+BLOCK_N] -> b_tile);

          gemm(a_tile, b_tile, dst=c_tile);   # FP32 accumulate

      store(c_tile -> C[m_start:m_start+BLOCK_M, n_start:n_start+BLOCK_N]);

  @llm_hint("available_tiling")
}
```

available_tiling:

- BLOCK_M=128, BLOCK_N=256, BLOCK_K=128, GROUP_SIZE=4, K_PIPE=2, PARALLEL_NUM=24
- BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_SIZE=4, K_PIPE=2, PARALLEL_NUM=24
- BLOCK_M=256, BLOCK_N=128, BLOCK_K=128, GROUP_SIZE=4, K_PIPE=2, PARALLEL_NUM=24
- BLOCK_M=128, BLOCK_N=256, BLOCK_K=64,  GROUP_SIZE=8, K_PIPE=4, PARALLEL_NUM=24
- BLOCK_M=64,  BLOCK_N=256, BLOCK_K=128, GROUP_SIZE=8, K_PIPE=2, PARALLEL_NUM=24

## 4) 推荐默认配置（针对 2048x1024 x 1024x1536）

- 默认优先：`BLOCK_M=128, BLOCK_N=256, BLOCK_K=128, GROUP_SIZE=4, K_PIPE=2`
- 原因：
  - `M=2048, N=1536`，M 维更大，行优先分组可提升 A 的复用；
  - `BLOCK_M x BLOCK_N = 128x256`，L0C 恰好 128KB（FP32 累加）；
  - `BLOCK_K=128` 时 L0B 恰好 64KB，L0A 为 32KB，搬运与计算更平衡。
