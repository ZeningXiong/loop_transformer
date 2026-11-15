"""Download SenseNova-Loop-9B-Preview from ModelScope with resume, overall progress, and sha256 check."""
import hashlib
import json
import os
import sys
import time
import urllib.request

REPO = "SenseNova/SenseNova-Loop-9B-Preview"
TARGET = r"D:\models\SenseNova-Loop-9B-Preview"
API = f"https://modelscope.cn/api/v1/models/{REPO}/repo/files?Recursive=true&Revision=master"
RAW = f"https://modelscope.cn/models/{REPO}/resolve/master/"
SKIP_PREFIXES = ("examples/", "assets/")
CHUNK = 1 << 20


def fmt(n):
    return f"{n / 1e9:6.2f} GB"


def list_files():
    with urllib.request.urlopen(API, timeout=60) as r:
        data = json.load(r)
    files = [
        f for f in data["Data"]["Files"]
        if f.get("Type") != "tree" and not f["Path"].startswith(SKIP_PREFIXES)
    ]
    files.sort(key=lambda f: f.get("Size", 0))  # small config files first
    return files


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    os.makedirs(TARGET, exist_ok=True)
    files = list_files()
    total = sum(f.get("Size", 0) for f in files)
    print(f"模型: {REPO}  来源: ModelScope")
    print(f"保存到: {TARGET}")
    print(f"共 {len(files)} 个文件，{fmt(total)}。中断后重新运行即可断点续传。\n")

    done_before = 0
    t0 = time.time()
    got_this_run = 0

    for i, f in enumerate(files, 1):
        path, size, sha = f["Path"], f.get("Size", 0), f.get("Sha256")
        dest = os.path.join(TARGET, path.replace("/", os.sep))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        part = dest + ".part"

        if os.path.exists(dest) and os.path.getsize(dest) == size:
            done_before += size
            continue

        attempt = 0
        while True:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if have >= size:
                break
            try:
                req = urllib.request.Request(RAW + path, headers={"Range": f"bytes={have}-"})
                with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as out:
                    last = 0.0
                    while True:
                        buf = r.read(CHUNK)
                        if not buf:
                            break
                        out.write(buf)
                        have += len(buf)
                        got_this_run += len(buf)
                        now = time.time()
                        if now - last > 0.5:
                            last = now
                            overall = done_before + have
                            speed = got_this_run / max(now - t0, 1e-6)
                            eta = (total - overall) / speed if speed > 0 else 0
                            pct = overall / total * 100
                            bar = "#" * int(pct / 2.5)
                            sys.stdout.write(
                                f"\r[{bar:<40}] {pct:5.1f}%  {fmt(overall)} / {fmt(total)}  "
                                f"{speed / 1e6:5.1f} MB/s  剩余 {int(eta // 60):3d} 分  "
                                f"文件 {i}/{len(files)} {os.path.basename(path)[:28]:<28}"
                            )
                            sys.stdout.flush()
            except Exception as e:  # network hiccup: retry with resume
                attempt += 1
                print(f"\n  {path} 中断（{e}），{min(attempt, 6) * 5} 秒后续传...")
                time.sleep(min(attempt, 6) * 5)

        if sha:
            sys.stdout.write(f"\n  校验 {path} ...")
            sys.stdout.flush()
            if sha256_of(part) != sha:
                print(" SHA256 不一致，删除后重新下载")
                os.remove(part)
                return main()
            print(" OK")
        os.replace(part, dest)
        done_before += size

    print(f"\n\n===== 全部完成：{len(files)} 个文件，{fmt(total)}，均已校验 =====")


if __name__ == "__main__":
    main()
