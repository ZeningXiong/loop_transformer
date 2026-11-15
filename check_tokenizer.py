import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
from transformers import AutoTokenizer
from diag_sets import SETS
M = r"D:\models\SenseNova-Loop-9B-Preview"
a = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
b = AutoTokenizer.from_pretrained(M, trust_remote_code=True, fix_mistral_regex=True)
diff = 0
for think in (False, True):
    for q, _ in SETS["math"] + SETS["basic"]:
        p = a.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=think)
        ia, ib = a(p).input_ids, b(p).input_ids
        if ia != ib:
            diff += 1
            print("不同:", q, "\n  默认:", [a.decode([t]) for t in ia][-30:], "\n  修正:", [b.decode([t]) for t in ib][-30:])
print("prompt 数", 2 * (len(SETS["math"]) + len(SETS["basic"])), "分词不同的数量", diff)
for q in ["13 乘以 17 再减去 29", "1234 加 5678", "2 的 10 次方减去 100"]:
    print(q, "→", [a.decode([t]) for t in a(q).input_ids])
