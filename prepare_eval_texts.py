"""Build the evaluation text set for the NF4-vs-bf16 comparison (eval_texts.jsonl).

Mostly real, pre-existing text of different kinds (Chinese/English prose, code, classical poetry,
chat transcripts in the model's own format). Each sample is cut to MAX_TOKENS tokens later.
"""
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"C:\Users\20495\AppData\Local\Programs\Python\Python312"
LM = r"D:\ClaudeCode\project\large_model"
MD = r"D:\models\SenseNova-Loop-9B-Preview-nf4"


def read(path, start=0, chars=3000):
    s = open(path, encoding="utf-8", errors="replace").read()
    s = re.sub(r"<[^>]+>", "", s)  # strip html badges in READMEs
    return s[start:start + chars]


def chat(turns):
    """Render (user, assistant) turns in the model's chat format (Qwen-style ChatML)."""
    out = []
    for u, a in turns:
        out.append(f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n")
    return "".join(out)


def jsonl(path):
    return [json.loads(l) for l in open(os.path.join(HERE, path), encoding="utf-8")]


think = jsonl("diag_math_nf4_think.jsonl")
basic = jsonl("diag_basic_offline_nothink.jsonl")

samples = [
    ("zh_tech_process", "中文技术文档（qms 的 Process.md）", read(os.path.join(LM, "Process.md"), 0)),
    ("zh_tech_optimize", "中文技术文档（Optimize.md）", read(os.path.join(LM, "Optimize.md"), 0)),
    ("zh_readme_model", "中文说明（模型 README_CN）", read(os.path.join(MD, "README_CN.md"), 1500)),
    ("en_license", "英文法律文本（Python LICENSE）", read(os.path.join(PY, "LICENSE.txt"), 0)),
    ("en_readme_model", "英文说明（模型 README）", read(os.path.join(MD, "README.md"), 1500)),
    ("code_json_decoder", "Python 代码（标准库 json/decoder.py）", read(os.path.join(PY, "Lib", "json", "decoder.py"), 0)),
    ("code_model", "Python 代码（模型的 modeling_sensenova_loop.py）", read(os.path.join(MD, "modeling_sensenova_loop.py"), 2000)),
    ("zh_classical", "古诗词", "\n\n".join([
        "静夜思\n李白\n床前明月光，疑是地上霜。举头望明月，低头思故乡。",
        "春望\n杜甫\n国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。烽火连三月，家书抵万金。白头搔更短，浑欲不胜簪。",
        "登鹳雀楼\n王之涣\n白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
        "望庐山瀑布\n李白\n日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
        "春晓\n孟浩然\n春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
        "相思\n王维\n红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
        "江雪\n柳宗元\n千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
        "出塞\n王昌龄\n秦时明月汉时关，万里长征人未还。但使龙城飞将在，不教胡马度阴山。",
    ])),
    ("chat_math_think", "对话：算术题 + 模型自己的思考过程", chat([(r["q"], "<think>\n" + r["full"]) for r in think])),
    ("chat_general", "对话：介绍杜甫 + 写代码（模型自己的回答）", chat([(r["q"], "<think>\n\n</think>\n\n" + r["full"]) for r in basic])),
]

with open(os.path.join(HERE, "eval_texts.jsonl"), "w", encoding="utf-8") as f:
    for key, desc, text in samples:
        f.write(json.dumps({"key": key, "desc": desc, "text": text}, ensure_ascii=False) + "\n")
        print(f"{key:20s} {len(text):5d} 字符  {desc}")
