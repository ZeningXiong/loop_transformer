#!/bin/bash
# WSL 里用 curl 预下载 wheel，绕开 pip 的慢速与源码构建
export HOME=/root
PIP=/root/sn/bin/pip
IDX="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
PKGS="torch==2.5.1 transformers==5.18.0 tokenizers==0.23.3 accelerate==1.1.1 safetensors==0.8.0 bitsandbytes==0.50.2 flash-linear-attention==0.3.2 numpy==1.26.4 psutil==6.0.0"

echo "=== 1) resolve (metadata only) ==="
$PIP install --dry-run --ignore-installed --report /tmp/rep.json $PKGS $IDX > /tmp/resolve.log 2>&1
echo "resolve_exit=$?"
tail -2 /tmp/resolve.log

/root/sn/bin/python - <<'PY' > /tmp/urls.txt
import json
try:
    d = json.load(open('/tmp/rep.json'))
except Exception as e:
    print("REPORT_ERR", e); raise SystemExit
for it in d.get('install', []):
    u = (it.get('download_info') or {}).get('url') or ''
    if u:
        print(u)
PY
echo "urls: $(wc -l < /tmp/urls.txt)"
head -2 /tmp/urls.txt

echo
echo "=== 2) curl download (parallel) ==="
mkdir -p /tmp/wheels
cd /tmp/wheels || exit 1
xargs -n1 -P6 curl -4 -sSL -O -C - < /tmp/urls.txt
echo "downloaded: $(du -sh /tmp/wheels | cut -f1), files: $(ls /tmp/wheels | wc -l)"

echo
echo "=== 3) local install ==="
$PIP install --no-index --find-links=/tmp/wheels $PKGS 2>&1 | tail -4
echo "install_exit=$?"
/root/sn/bin/python -c "import torch, transformers, bitsandbytes, triton; print('VER', torch.__version__, torch.cuda.is_available(), transformers.__version__, triton.__version__)" 2>&1 | tail -3
echo ALL_DONE
