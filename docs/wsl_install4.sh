#!/bin/bash
# 带长超时与重试的安装；pip 偶发停滞时靠重试推进，已下载的 wheel 会进缓存复用
export HOME=/root
pkill -f "pip install" 2>/dev/null; sleep 1
ip link set dev eth0 mtu 1400
sysctl -w net.ipv6.conf.all.disable_ipv6=1 >/dev/null 2>&1

PIP=/root/sn/bin/pip
IDX="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
OPTS="--progress-bar off --timeout 60 --retries 20"

for attempt in 1 2 3 4 5; do
  echo "=== torch attempt $attempt ==="
  if $PIP install $OPTS torch==2.5.1 $IDX 2>&1 | tee -a /tmp/pip_torch.log | grep -qE "Successfully installed|already satisfied"; then
    echo "torch ok"; break
  fi
  tail -3 /tmp/pip_torch.log
done

for attempt in 1 2 3; do
  echo "=== rest attempt $attempt ==="
  if $PIP install $OPTS transformers==5.18.0 tokenizers==0.23.3 accelerate==1.1.1 \
      safetensors==0.8.0 bitsandbytes==0.50.2 flash-linear-attention==0.3.2 \
      numpy==1.26.4 psutil==6.0.0 $IDX 2>&1 | tee -a /tmp/pip_rest.log | grep -qE "Successfully installed|already satisfied"; then
    echo "rest ok"; break
  fi
  tail -3 /tmp/pip_rest.log
done

/root/sn/bin/python -c "import torch, transformers, bitsandbytes, triton; print('VER', torch.__version__, torch.cuda.is_available(), transformers.__version__, triton.__version__)" 2>&1 | tail -3
echo ALL_DONE
