#!/bin/bash
# 在 WSL 里复测 16K 单步耗时（与 Windows 同一份模型、同一份脚本）
export HOME=/root
export HF_HOME=/root/.cache/huggingface
export SENSENOVA_MODEL=/mnt/d/models/SenseNova-Loop-9B-Preview
export SENSENOVA_NF4=/mnt/d/models/SenseNova-Loop-9B-Preview-nf4
export PYTHONDONTWRITEBYTECODE=1
cd /mnt/d/ClaudeCode/project/loop_transformer || exit 1

echo "=== env ==="
/root/sn/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "=== 16K probe ==="
/root/sn/bin/python -u probe_16k_phases.py 2>&1 | grep -E "prefill 后|token 前向|自投机|对比|结束时|Error|error|Traceback|out of memory"
