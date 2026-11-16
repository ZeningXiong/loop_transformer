# SenseNova-Loop-9B 在 8 GB 笔记本显卡上的推理优化

在 **RTX 4070 Laptop（8 GB 显存 / 32 GB 内存）** 上跑 **SenseNova-Loop-9B-Preview**（9.4B 参数，24 层循环两次 = 每 token 56 次层执行）。

一句话结论：**从"跑不起来"到 16K 上下文 10.6 tok/s、峰值 7.6 GB**，而内存只占 2.6 GB。

> 仓库里是完整的研究过程：可用的加载器与投机解码实现 + 每一步的验证脚本。
> 所有提升都经过实测，失败的尝试也留在 `docs/Optimize.md` 里（含原因）。

## 结果

| | 起点 | 现在 |
|---|---:|---:|
| 显存 | 9.04 GB（超限，跑不起来） | **7.6 GB @ 16K 上下文** |
| 生成速度 | 1.6–1.8 tok/s | **10.6 tok/s**（2-token 自投机） |
| 内存（RAM） | 10.1 GB | **2.6 GB** |
| 上下文 | 2K | **16K**（8K 更宽裕） |
| 质量 | bf16 基准 | KL 0.078（混合精度）/ 0.146（全 NF4）；开思考数学题 **8/8** |

16K 上下文下的一步耗时构成（steady state）：验证前向 166 ms + 草稿 3 ms + MTP 回填 14 ms ≈ **189 ms**，提交 2 个 token。

## 目录

```
sn_loop_common.py      加载器：离线 NF4 权重、固定 KV 缓冲、KV int8（可 K 共享）、分块 prefill、显存保护、Sinkhorn 补丁
sn_spec.py             MTP 自投机解码（2-token 验证、快照回滚、自适应降级）
fast_sinkhorn.py       2×2 Sinkhorn 闭式解（默认启用，省 23%/步）
fused_rmsnorm.py       Triton 融合 RMSNorm（逐位一致，单算子 ×1.23）
int8_gemv.py           int8 权重的单 token GEMV
cudagraph_decode.py    1-token CUDA Graph 解码（可选，只在短上下文划算）
cudagraph_spec.py      2-token 验证的 Graph 版本

convert_nf4.py         原始 FP32 → 离线 NF4（一次性）
convert_int8.py        按层 int8 权重转换（给混合精度用）
mixed_search.py        按层混合精度搜索，产出 mixed_plan_*.json
download_sensenova_loop.py / prepare_eval_texts.py   取权重、生成评测文本

diag_sensenova_loop.py 评测入口（数学题 / 开放问题，--think 开思考）
eval_nf4_vs_bf16.py    与 bf16 参考的逐位置 KL 对比
test_sinkhorn_quality.py / test_sinkhorn_quality_think.py   闭式解 vs 20 轮迭代的任务级对比

speed_now.py           当前配置的实测速度（16K，普通 vs 2-token）
test_spec_k.py         2 / 3 / 4 / 6 token 验证对比
bench_spec16k.py       16K 投机解码基准（最早的 9.68 tok/s 就是它）
test_mtp_nf4.py        MTP 头量化成 NF4 的显存/命中率/速度
probe_16k_phases.py    16K 一步的分段计时（单 token / 2-token / 整步）
profile_spec_step.py   投机一步的五段分解（快照/草稿/验证/重算/回填）
profile_mtp_feed.py    MTP 回填的子算子级拆分

check_fixed_kv.py      固定 KV + 折叠 GQA 的正确性
check_kv_int8.py       int8 KV 相对 bf16 KV 的精度
check_no_pass2.py      砍掉第二遍循环会怎样（KL 2.28 → 方向关闭）
test_loop4_8k.py       推理时改循环次数（2 遍 vs 4 遍）
bench_mhc_ablation.py  mHC 路由的耗时分解（sinkhorn 占 25.2 ms）
probe_graph_replay.py  CUDA Graph 重放 vs eager 前向
measure_near_miss.py   失配时草稿离正确 token 有多近（近似接受可行性）
test_clock_ramp.py     冷启动 vs 预热的单步耗时 + GPU 频率
test_guard_overhead.py VramGuard 对速度的影响
test_fast_sinkhorn*.py 闭式解的数值验证与收敛性
test_fused_rmsnorm.py / debug_rmsnorm_model.py / test_fused_rmsnorm_e2e.py   RMSNorm 融合的逐位验证

plot_mtp_hitrate.py / make_chart.py   生成 assets/ 里的两张图

docs/Optimize.md       完整优化记录：每个提升的细节 + 失败尝试的一句话结论
assets/                结果图（MTP 逐位命中率、CUDA Graph 对比）
mixed_plan_*.json      混合精度方案（+1.0 / +1.5 / +2.0 / +2.5 GB 预算）
eval_texts*.jsonl      评测用文本（技术文档/代码）
logs/                  历史运行日志（不入库）
```

## 关键手段

1. **离线 NF4 量化 + embedding 走 SSD 按行读**：权重 4.5 GB 常驻显存，2 GB 的 embedding 表不占显存也不占 RAM（每 token 读 1 行，0.4–0.6 ms）。
2. **固定 KV 缓冲 + 不复制 K/V 的注意力**：把 `DynamicCache` 换成按 `max_len` 预分配的层，注意力直接在原缓冲上算，避免每步 `torch.cat`。
3. **KV 存 int8**：16K 的 KV 从 0.96 GB 降到 0.48 GB，代价是 KL 2.6e-3–1e-2。
4. **分块 prefill（256/512）+ 显存保护**：长 prompt 不再一次性爆显存；整卡到 97% 立即中止。
5. **自投机解码（模型自带 MTP 头）**：草稿 1 个 token、一次前向验证 2 个位置，命中率 87–100%，2-token 验证只贵 5–20%。
6. **2×2 Sinkhorn 闭式解**：`p = sigmoid((a+d−b−c)/2)`，把每 token 112 次迭代换成一次 sigmoid，一步 −23%，输出任务级不变。
7. **MTP 头量化成 NF4**：省 0.35 GB，命中率与速度不变。

## 快速开始

需要自己准备模型权重（本仓库不含权重）：

```bash
# 1) 原始检查点放到 MODEL / NF4_DIR 指向的位置（见下）
# 2) 转成离线 NF4（只做一次，约 10 分钟）
python convert_nf4.py
# 3) 跑评测
python diag_sensenova_loop.py --quant offline --set math --think
# 4) 测投机解码速度（16K）
python test_spec_k.py
```

路径用环境变量覆盖，不写死：

```powershell
$env:SENSENOVA_MODEL = "D:\models\SenseNova-Loop-9B-Preview"        # 原始 FP32 检查点
$env:SENSENOVA_NF4   = "D:\models\SenseNova-Loop-9B-Preview-nf4"    # 离线 NF4 目录
```

环境：Python 3.12、Windows 11、`torch 2.5.1+cu124`、`transformers 5.18.0`、`bitsandbytes 0.50.2`、
`flash-linear-attention 0.3.2`、`triton-windows 3.1.0.post17`。依赖见 `requirements.txt`。

## 测量纪律（踩坑换来的）

1. **一次只测一个配置、一个进程、一个 KV 缓存。** 两个缓存同时存在会把显存顶到 8.5 GB，
   借内存后速度掉一半，之前的结论会全部作废。
2. **prefill 之后的前 3 步极慢**（实测 1356 vs 稳态 189 ms/步，16K KV 首次触碰的缺页 + 内核 autotune），
   必须剔除预热步，否则先跑的配置会被算慢 2–7 倍。
3. **分段计时每段之间要 `torch.cuda.synchronize()`**，否则上一段的 GPU 尾巴会算进下一段
   （曾据此误判"MTP 回填 21 ms"，实际 7 ms）。
4. **显存看整卡峰值，不看进程分配。** 判断标准是"离 8.59 GB 还有多少"，不是时间点。
5. **这个模型不是逐位确定的**：同一份代码跑两次 KL 0.55（bnb / FLA 内核有原子加）。
   所以一致性验证要落在**模块级逐位比对**或**任务级答案比对**上，端到端 KL 有 0.5 的噪声底。

## 已知限制

- **16K + 混合精度（int8 层）+ 投机解码放不下**：差约 0.3 GB，需要降到 8K 或再压 KV。
- **多草稿不划算**：3/4/6-token 验证分别比 2-token 慢 12%/18%/35%。原因是线性注意力（Gated DeltaNet）
  的状态无法像 KV 那样截断，失配就要回滚重喂，而第 2 位命中率只有 27%。
- **CUDA Graph 只在短上下文（≤2K）有收益**（×1.19–1.31），16K 下反而慢 10%（GPU 受限 + 固定 pad 掩码）。
- **推理时不能增加循环次数**：把循环块跑 4 遍，8K 下慢 1.89 倍且输出退化成乱码——循环次数是训练时属性。
- **元素级融合收益有限**：RMSNorm 单算子 ×1.23，但整步只省 1.7 ms（它只占一步的 2.3%）。

## 文档

- [`docs/Optimize.md`](docs/Optimize.md)：完整的优化记录，每个提升的测法与数字，失败尝试的一句话结论
- [`assets/mtp_hitrate.png`](assets/mtp_hitrate.png)：MTP 逐位命中率（93% / 27% / 50% / 0% / 0%）
- [`assets/cudagraph_compare.png`](assets/cudagraph_compare.png)：CUDA Graph 在不同上下文下的对比

## 说明

这是个人研究过程的记录，不是库。脚本之间靠同目录 import，入口文件各自独立可跑。
评测文本来自公开技术文档与代码，模型权重版权归 SenseNova 所有。
