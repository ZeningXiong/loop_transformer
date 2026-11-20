#!/bin/bash
# 网络修正生效后直接 pip 安装（缓存复用，中断可重跑）
export HOME=/root
pkill -f "pip install" 2>/dev/null; sleep 1
ip link set dev eth0 mtu 1400
sysctl -w net.ipv6.conf.all.disable_ipv6=1 >/dev/null 2>&1
echo "mtu=$(cat /sys/class/net/eth0/mtu) ipv6_off=$(cat /proc/sys/net/ipv6/conf/all/disable_ipv6)"

PIP=/root/sn/bin/pip
IDX="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"

$PIP install --progress-bar off torch==2.5.1 $IDX 2>&1 | grep -E "Downloading|Installing collected|Successfully" | tail -6
echo "TORCH_DONE=$?"
$PIP install --progress-bar off transformers==5.18.0 tokenizers==0.23.3 accelerate==1.1.1 \
  safetensors==0.8.0 bitsandbytes==0.50.2 flash-linear-attention==0.3.2 \
  numpy==1.26.4 psutil==6.0.0 $IDX 2>&1 | grep -E "Downloading|Installing collected|Successfully|error" | tail -6
echo "REST_DONE=$?"
/root/sn/bin/python -c "import torch, transformers, bitsandbytes, triton; print('VER', torch.__version__, torch.cuda.is_available(), transformers.__version__, triton.__version__)" 2>&1 | tail -3
echo ALL_DONE
