# WSL2 里跑这个项目的网络问题与修复

在这台机器（Windows 11 + WSL2 Ubuntu 24.04 + RTX 4070 Laptop）上，WSL 里 `pip install` 会卡在几 MB 不动。
排查结论和修复如下，**换机器/重装 WSL 后照这个做**。

## 现象

- `apt-get` 正常；小文件（几十 KB 的 HTML）正常；
- 大文件（torch wheel ~2.5 GB）**下载停滞**：10 分钟只下到 2.3 MB；
- `download.pytorch.org`（官方 CUDA 源）直接不可达（curl 返回 `000`）。

## 根因（两条，叠加）

1. **MTU 不匹配（大包黑洞）**
   `ping -M do -s 1472` 到 223.5.5.5 收到 `Frag needed and DF set (mtu = 1492)`——
   链路是 **PPPoE 1492**，而 WSL 的 `eth0` 默认 **1500**。WSL 的 NAT 不回传 ICMP 不可达，
   于是大包被静默丢弃、TCP 卡在重传上；小请求能过去，大传输就停住。

2. **IPv6 不可用但优先**
   `getent hosts pypi.org` 只返回 IPv6 地址（`2a04:4e42::223`），而 WSL 的 IPv6 出不去。
   `pip` 先试 IPv6 → 每次都要等超时，表现为"卡住"。

## 修复（已在本机生效并持久化）

```bash
# 1) MTU 降到 1400（低于 1492 即可，留够隧道开销）
ip link set dev eth0 mtu 1400

# 2) 关掉 IPv6，并让 glibc 优先 IPv4
sysctl -w net.ipv6.conf.all.disable_ipv6=1
echo "precedence ::ffff:0:0/96 100" >> /etc/gai.conf

# 3) 持久化
cat >> /etc/wsl.conf <<'EOF'

[boot]
command = ip link set dev eth0 mtu 1400
EOF
printf 'net.ipv6.conf.all.disable_ipv6=1\n' > /etc/sysctl.d/99-disable-ipv6.conf

# 4) 换国内镜像
/root/sn/bin/pip install <pkg> -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

## 修复后的实测

| 测试 | 修复前 | 修复后 |
|---|---|---|
| torch wheel（curl，整段） | 停滞 | **35 MB/s**（15 秒下了 524 MB） |
| torch wheel（curl，range） | — | 27–30 MB/s |
| `pip install torch` | ~100 KB/s | ~1.4 MB/s（pip 自身仍慢，但能用） |

**注意：pip 本身仍比 curl 慢一个数量级**（~1.4 MB/s vs 30 MB/s）。如果急需，可以先用
`curl -C -` 把大 wheel 拉到本地再 `pip install /path/to.whl --no-deps`。

## 复现这条测试需要的东西

- WSL2 + Ubuntu 24.04、Python 3.12、`python3-venv`；
- venv 里：`torch==2.5.1`、`transformers==5.18.0`、`bitsandbytes==0.50.2`、
  `flash-linear-attention==0.3.2`、`numpy==1.26.4`、`psutil==6.0.0`；
- 模型在 `E:\models\...`，WSL 里是 `/mnt/e/models/...`：

```bash
export SENSENOVA_MODEL=/mnt/e/models/SenseNova-Loop-9B-Preview
export SENSENOVA_NF4=/mnt/e/models/SenseNova-Loop-9B-Preview-nf4
cd /mnt/d/ClaudeCode/project/loop_transformer
/root/sn/bin/python probe_16k_phases.py
```
