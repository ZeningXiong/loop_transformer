#!/bin/bash
export HOME=/root
PIP=/root/sn/bin/pip
IDX="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"

$PIP install --progress-bar off --timeout 60 --retries 10 \
  transformers==5.18.0 tokenizers==0.23.3 accelerate==1.1.1 safetensors==0.8.0 \
  bitsandbytes==0.50.2 flash-linear-attention==0.3.2 numpy==1.26.4 psutil==6.0.0 \
  $IDX 2>&1 | tail -4

echo "=== verify ==="
/root/sn/bin/python - <<'PY'
import torch, transformers, bitsandbytes, triton
print("VER", torch.__version__, torch.cuda.is_available(), transformers.__version__, triton.__version__)
PY
