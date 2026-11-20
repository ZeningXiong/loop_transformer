#!/bin/bash
export HOME=/root
pkill -f "pip install" 2>/dev/null
sleep 1
ip link set dev eth0 mtu 1400
sysctl -w net.ipv6.conf.all.disable_ipv6=1 >/dev/null 2>&1
echo "mtu=$(cat /sys/class/net/eth0/mtu) ipv6_off=$(cat /proc/sys/net/ipv6/conf/all/disable_ipv6)"

PIP="/root/sn/bin/pip"
IDX="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"

echo "=== torch (with deps) ==="
$PIP install --no-cache-dir torch==2.5.1 $IDX 2>&1 | tail -3
echo "=== rest ==="
$PIP install --no-cache-dir transformers==5.18.0 tokenizers==0.23.3 accelerate==1.1.1 \
    safetensors==0.8.0 bitsandbytes==0.50.2 flash-linear-attention==0.3.2 \
    numpy==1.26.4 psutil==6.0.0 $IDX 2>&1 | tail -3
echo "ALL_DONE"
/root/sn/bin/python -c "import torch, transformers, bitsandbytes, triton; print('VER', torch.__version__, torch.cuda.is_available(), transformers.__version__, triton.__version__)"
