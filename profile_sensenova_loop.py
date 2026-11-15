"""Profile decode steps of SenseNova-Loop-9B-Preview (nf4) to find where time goes."""
import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir
from sn_loop_common import load

sys.stdout.reconfigure(encoding="utf-8")
NO_COMPILE = "--no-compile" in sys.argv
import json
PLAN = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--plan=")), None)
INT8 = set(json.load(open(PLAN, encoding="utf-8"))["int8"]) if PLAN else ()
tok, model = load("offline", compile_sinkhorn=not NO_COMPILE, int8=INT8)

prompt = tok.apply_chat_template([{"role": "user", "content": "介绍一下长城。"}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False)
inputs = tok(prompt, return_tensors="pt").to("cuda:0")
with torch.inference_mode():
    model.generate(**inputs, max_new_tokens=8, do_sample=False)  # warm-up / triton compile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        t = time.time()
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        torch.cuda.synchronize()
        dt = time.time() - t
n = out.shape[1] - inputs.input_ids.shape[1]
print(f"生成 {n} tokens 用时 {dt:.1f} 秒（{n / dt:.2f} tok/s）")
ka = prof.key_averages()
cuda_total = sum(e.self_device_time_total for e in ka) / 1e6
cpu_total = sum(e.self_cpu_time_total for e in ka) / 1e6
print(f"GPU 实际计算时间合计 {cuda_total:.1f} 秒；CPU 侧时间合计 {cpu_total:.1f} 秒；内核启动次数 "
      f"{sum(e.count for e in ka if e.key in ('cudaLaunchKernel', 'cuLaunchKernel'))}")
print("\n== 按 GPU 时间排序 ==")
print(ka.table(sort_by="self_device_time_total", row_limit=12, max_name_column_width=60))
print("\n== 按 CPU 时间排序 ==")
print(ka.table(sort_by="self_cpu_time_total", row_limit=15, max_name_column_width=60))
print("\n== copy_ 的来源（按输入形状分组）==")
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_device_time_total", row_limit=10, max_name_column_width=40))
