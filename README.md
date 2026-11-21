# SenseNova-Loop-9B 在 8 GB 笔记本显卡上的推理优化

在 **RTX 4070 Laptop（8 GB 显存 / 32 GB 内存）** 上跑 **SenseNova-Loop-9B-Preview**
（9.4B 参数，24 层循环两次 = 每 token 56 次层执行）。

一句话结论：**从"跑不起来"到 8K 上下文 23.5 tok/s（16K 为 20.2 tok/s），显存 7.1–7.4 GB，内存只占 2.6 GB。**

> 仓库里是完整的研究过程：可用的加载器、投机解码、CUDA graph 与量化 KV 注意力，以及每一步的验证脚本。
> 所有数字都经过实测；**失败的尝试也留在 `docs/Optimize.md` 里并写明原因**。

## 结果

| | 起点 | 现在 |
|---|---:|---:|
| 显存 | 9.04 GB（超限，跑不起来） | **7.07–7.38 GB**（8K / 16K） |
| 生成速度 | 1.6–1.8 tok/s | **23.5 tok/s @8K 中位**（20.2 @16K） |
| 内存（RAM） | 10.1 GB | **2.6 GB** |
| 上下文 | 2K | **8K（推荐）/ 16K（可用）** |
| 质量 | bf16 基准 | KL 0.146（全 NF4）；开思考数学题 **8/8**，不思考 5/8 |

### 16K / 8K 的三种组合（同一脚本、同一时段实测）

| 上下文 | 组合 | 中位 tok/s | 稳定性（3 次极差） |
|---|---|---:|---:|
| 8K | 图 + 投机 + 量化 KV | **23.45** | **7%** |
| 16K | 图 + 投机 + 量化 KV | 20.21 | 19% |
| 16K | 图 + 投机（不含量化 KV） | 17.5 | — |
| 16K | 纯贪心 | 9.8 | — |

**8K 为什么更值得**：中位快 16%、极差小 3 倍、prefill 只要 28 s（16K 要 47 s）、整卡峰值低 0.2–0.3 GB。
详见 `docs/Optimize.md` 的 27.6 / 27.7。

## 快速开始

模型权重放在 `D:\models\SenseNova-Loop-9B-Preview{,-nf4}`（可用 `SENSENOVA_MODEL` / `SENSENOVA_NF4` 覆盖）。

```powershell
# 一次性：FP32 权重 -> 离线 NF4（4.5 GB）
.\.venv\Scripts\python.exe convert_nf4.py

# 对话（默认：≥256 token 自动走 图+投机+量化KV，短对话走贪心）
python chat.py --ctx 8192

# 一次性提问
python chat.py --ctx 8192 --q "用一句话解释什么是线性注意力"
```

`chat.py` 用一个非 venv 的解释器启动时会自动切回项目 venv。

## 目录

### 核心运行时

```
sn_loop_common.py     加载器：离线 NF4 权重、固定 KV 缓冲、KV int8（第 2 遍 int4）、分块 prefill、
                      显存保护、折叠 GQA 注意力（gqa_fold_attention）、量化 KV 的 QuantKV 开关
sn_spec.py            MTP 自投机解码（2-token 验证、快照回滚、自适应降级）
fast_sinkhorn.py      2×2 Sinkhorn 闭式解（默认启用，省 23%/步）
fused_rmsnorm.py      Triton 融合 RMSNorm（逐位一致）
fused_silu.py         Triton 融合 SiLU×up（逐位一致，PTX 舍入栅栏）
fused_kv_attn.py      解码注意力直接读 int8/int4 KV（split-L flash，图路径下 +21%）
int4_dequant.py       4-bit KV 的融合解包（逐位一致，17×）
int8_gemv.py          int8 权重的单 token GEMV
cudagraph_decode.py   1-token CUDA Graph 解码
cudagraph_spec.py     2-token 验证的 Graph 版本（Graph2Decoder，当前最快路径的核心）
chat.py               对话入口
```

### 一次性工具

```
convert_nf4.py            FP32 -> 离线 NF4
convert_int8.py           按层 int8 权重（混合精度用）
mixed_search.py           按层混合精度搜索（产出 mixed_plan_*.json）
prepare_eval_texts.py     生成评测文本
download_sensenova_loop.py  取原始权重
```

### 评测与测速（保留的主要几个）

```
test_graph_prod.py     当前最快组合的基准：图 + 投机 + 量化 KV，打印稳态 ms/步、命中、整卡峰值
probe_16k_phases.py    一步的分段计时（1-token / 2-token / 自投机）
test_kv_int4.py        KV 第 2 遍 int4 的端到端速度与显存
check_fused_kv.py      量化 KV 注意力的正确性 + 速度（对比解量化 + SDPA）
check_verify2.py       2-token 前向 vs 两次 1-token（残差有多大）
bench_layer_cost.py    单层耗时（全注意力层 vs 线性注意力层）
probe_step_parts.py    Graph step 内部分段（emb/掩码/索引/重放/lm_head）
speed_now.py           当前配置的快速测速
diag_sensenova_loop.py 任务级评测（数学题；--think 开思考）
eval_nf4_vs_bf16.py    与 bf16 参考的逐位置 KL
```

> 其余历史脚本（大量 `bench_*` / `prof_*` / `probe_*` / `debug_*`）已经删除，需要时从 git 历史里找：
> `git log --diff-filter=D --name-only -- '*.py'`。

## 关键结论（详见 `docs/Optimize.md`）

前五条收益按大小排：

1. **启用折叠 GQA 注意力**（`use_gqa_fold_attention`，代码早就有但生产路径没调）：
   去掉 HF 每层 `repeat_kv` 的 2×134 MB 拷贝，16K 前向 **165.6 → 124.0 ms（−25%）**。
2. **图 + 2-token 投机**：每步提交 2 个 token，16K **16.8–22.4 tok/s**（纯贪心 9.6–10.1）。
3. **注意力直接读量化 KV**（图路径）：**16.80 → 20.31 tok/s（+21%）**，显存还降 0.29 GB；精度与 SDPA 差 2.4e-04。
4. **融合 RMSNorm + SiLU×up**（逐位一致）：16K 前向 **−5%**。
5. **KV 第 2 遍 int4 + 融合解包**：每层反量化 5.6 → 0.32 ms，速度代价≈0，省 0.10 GB。

反面结论（不要重复踩）：

- **MTP 捕获进图是负收益**：命中率 15/16 → 5/16，22.38 → 14.57 tok/s；
- 失配时"重做 1 个 token"省不到：1-token 图重放 89.3 ms ≈ 2-token 89.2 ms（权重带宽决定）；
- **换 WSL2/Linux 并不会更快**：比 Windows 慢 2–6%；
- 3/4/6-token 验证、近似接受、推理时加循环、KV 2-bit、K 共享：全部无收益或精度崩。

## 测试纪律（踩过的坑）

1. **必须先预热**（冷启动会慢 2.8×）；
2. **一次进程只测一种配置**，并且总看整卡峰值——超过 ~8.3 GB 会静默掉到 1/4 速度；
3. **图 + 投机必须在 16K/8K 的真实组合下测**：同一个优化在 eager 里可能完全测不出收益；
4. **单次数字不可信**：连跑 3 次取中位，同时记录命中率与整卡峰值（见 27.6）。

## 文档

- `docs/Optimize.md`：全部优化与**失败尝试**的实测记录（§27 是当前最新一轮）；
- `docs/Methods.md`：按"提速 / 省空间"分类的方法总览与部署路径；
- `docs/WSL.md`：WSL2 环境的网络排查（MTU/IPv6）与安装脚本。

## 模型与许可证

模型权重来自 SenseNova 官方发布，本仓库只包含推理代码与实验记录。
