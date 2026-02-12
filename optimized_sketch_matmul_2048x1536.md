# Optimized UnifiedSketch: MatMul (2048,1024) x (1024,1536) on Ascend 910B2

目标：为固定 shape `A[2048,1024] @ B[1024,1536] -> C[2048,1536]` 生成高效 matmul。

核心优化点（相对父代方案）：
- **固定 24 核启动**：每核循环处理多个输出块，避免 `grid=(NUM_BLOCKS_M*NUM_BLOCKS_N)` 过度启动。
- **Swizzle2D 分组重排**：按 `GROUP_SIZE` 对 (M,N) block 做组内重排，提升 L2/L1 局部性与负载均衡（本任务 `M>=N`，行优先分组更有利于复用 A 行块）。
- **更优 K 切分**：`BLOCK_K=256`（K=1024 -> 4 次迭代），在满足 L0A/L0B 约束下减少循环次数，提升 CUBE 利用率。
- **FP32 累加**：c_tile 作为 `f32` 累加器，最后 FixP/转换写回 `f16`，在保持精度的同时不显著影响吞吐。
- **核内流水 + 双缓冲意图**：GM->L1/UB 预取与 CUBE 计算重叠（由 coder 映射为 MTE 双缓冲/多级流水）。

---

```python
sketch matmul_2048x1536 {
  symbols: M, N, K;
  tensors: A[M, K]: f16; B[K, N]: f16; C[M, N]: f16;
  constexpr: BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE, NUM_CORES;

  # static shape for this task
  M, N, K = 2048, 1536, 1024
  NUM_CORES = 24

  # default tiling (tuned): fits L0A/L0B constraints for fp16
  BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 256
  GROUP_SIZE = 4

  NUM_BLOCKS_M = ceil(M, BLOCK_M)     # 2048/128=16
  NUM_BLOCKS_N = ceil(N, BLOCK_N)     # 1536/128=12
  NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

  # 24 AI Core parallel
  @llm_hint("parallel", "coreidx")
  for core_id in range(NUM_CORES):

    # each core handles multiple blocks: core_id, core_id+24, ...
    for block_linear in range(core_id, NUM_BLOCKS, NUM_CORES):
      block_m = block_linear // NUM_BLOCKS_N
      block_n = block_linear %  NUM_BLOCKS_N

      # swizzle2d grouping improves locality & balance (M>=N -> row-major grouping)
      @llm_hint("available_mapping", "swizzle2d_row_major")
      task_m, task_n = swizzle2d(block_m, block_n, NUM_BLOCKS_M, NUM_BLOCKS_N, GROUP_SIZE)

      m_start = task_m * BLOCK_M
      n_start = task_n * BLOCK_N

      # accumulator in fastest storage, FP32, init zero
      c_tile = alloc([BLOCK_M, BLOCK_N], llm_hint=["fastest", "accumulator", "init_zero", "f32"])

      # optional: double-buffered input tiles (coder maps to UB/L1 ping-pong + MTE pipeline)
      a_buf0 = alloc([BLOCK_M, BLOCK_K], llm_hint=["fast", "input_cache", "double_buffer"])
      a_buf1 = alloc([BLOCK_M, BLOCK_K], llm_hint=["fast", "input_cache", "double_buffer"])
      b_buf0 = alloc([BLOCK_K, BLOCK_N], llm_hint=["fast", "input_cache", "double_buffer"])
      b_buf1 = alloc([BLOCK_K, BLOCK_N], llm_hint=["fast", "input_cache", "double_buffer"])

      @llm_hint("pipeline")
      for k_start in range(0, K, BLOCK_K):
        # prefetch next tiles (ping-pong) to hide GM->L1/UB latency
        @llm_hint("pipeline", "mte2_overlap_cube")
        if (k_start / BLOCK_K) % 2 == 0:
          load(A[m_start:m_start+BLOCK_M, k_start:k_start+BLOCK_K] -> a_buf0)
          load(B[k_start:k_start+BLOCK_K, n_start:n_start+BLOCK_N] -> b_buf0)
          gemm(a_buf0, b_buf0, dst=c_tile)
        else:
          load(A[m_start:m_start+BLOCK_M, k_start:k_start+BLOCK_K] -> a_buf1)
          load(B[k_start:k_start+BLOCK_K, n_start:n_start+BLOCK_N] -> b_buf1)
          gemm(a_buf1, b_buf1, dst=c_tile)

      # writeback: FixP/cast to fp16 (coder chooses optimal path L0C->GM)
      store(c_tile -> C[m_start:m_start+BLOCK_M, n_start:n_start+BLOCK_N])
}

@llm_hint("avaliable_tiling")

avaliable_tiling:
  # Best-first candidates (all satisfy L0A/L0B for fp16; BK=256 reduces K-loop count)
  BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_SIZE=4
  BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_SIZE=8

  # Smaller BK for more flexibility / potentially higher freq on some schedules
  BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_SIZE=4
  BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_SIZE=8

  # More blocks for load-balance (may help when some cores underutilized)
  BLOCK_M=64,  BLOCK_N=128, BLOCK_K=256, GROUP_SIZE=8
  BLOCK_M=128, BLOCK_N=64,  BLOCK_K=256, GROUP_SIZE=8
```

---

实现对应参考：`matmul_2048x1536_triton_ascend.py`（固定 24 核、swizzle2d 分组、FP32 累加、BK=256 优先）。

