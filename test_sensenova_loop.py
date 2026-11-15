"""Quick smoke test: SenseNova-Loop-9B-Preview, 4-bit NF4 on GPU."""
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = r"D:\models\SenseNova-Loop-9B-Preview"
QUESTIONS = [
    "13 乘以 17 再减去 29 等于多少？只回答数字。",
    "用三句话介绍一下杜甫，以及他最有名的一首诗。",
    "写一个 Python 函数，判断一个字符串是否是回文（忽略大小写和非字母数字字符）。只给代码。",
]

sys.stdout.reconfigure(encoding="utf-8")  # python -I ignores PYTHONIOENCODING

# embedding (248320 x 4096 bf16, ~2 GB) and the unused vision tower stay on CPU;
# lm_head is now 4-bit quantized on the GPU together with the decoder layers.
DEVICE_MAP = {
    "model.language_model.embed_tokens": "cpu",
    "model.visual": "cpu",
    "model.language_model.layers": 0,
    "model.language_model.norm": 0,
    "model.language_model.rotary_emb": 0,
    "lm_head": 0,
}

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map=DEVICE_MAP,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["visual", "model.visual"],
        llm_int8_enable_fp32_cpu_offload=True,
    ),
).eval()
print(f"加载耗时 {time.time() - t0:.0f} 秒，模型显存占用 {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

for q in QUESTIONS:
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tok(prompt, return_tensors="pt").to("cuda:0")
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=300, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    dt = time.time() - t
    new = out[0, inputs.input_ids.shape[1]:]
    print("=" * 70)
    print("问:", q)
    print("答:", tok.decode(new, skip_special_tokens=True).strip())
    print(f"[{len(new)} tokens, {dt:.1f} 秒, {len(new) / dt:.1f} tokens/s]", flush=True)

print("=" * 70)
free, total = torch.cuda.mem_get_info()
print(f"峰值显存（本进程）{torch.cuda.max_memory_allocated() / 1e9:.2f} GB；"
      f"整卡已用 {(total - free) / 1e9:.2f} / {total / 1e9:.2f} GB")
