# Kernel Heap Spray Detection

基于内核动态内存分配行为的堆喷射（Heap Spray）攻击检测系统。通过 ftrace 采集 Linux
内核 `kmalloc`/`kfree` 事件，提取时序特征，用半监督（单类学习）模型检测异常分配模式。

当前主线：**schema v2 · 90 维特征 · final-v2 正式数据集 · M6 冻结**。冻结报告见
`runs/ACCEPTANCE_M6.md`；在新机器上逐步复现全流程见 `MIGRATION.md`；后续改进方向的操作
指南见 `IMPROVEMENTS.md`。

## 技术原理

### 堆喷射攻击的特征

内核堆喷射是漏洞利用的关键步骤：攻击者在触发 UAF/Double-Free/堆溢出后，需要大量分配
特定大小的对象来回收被释放的位置，写入受控数据以劫持控制流。这种在时间窗口内表现为：

- **集中分配**：短时间内大量分配同一 size class 的对象
- **单进程主导**：攻击 PoC 进程贡献绝大多数分配
- **call site 集中**：分配来源函数高度单一（Shannon 熵低）
- **alloc/free 同步**：跨大小类的分配与释放高度相关

### 检测思路

系统采用**半监督（单类学习）**范式：仅用正常数据训练，通过重建/偏离误差识别异常。
阈值基于**验证集正常分数的 p99 分位**标定（`scripts/train/common.py:threshold_at_fpr`），
不优化测试集。run 级判定取窗口分数的最大值聚合（`run_max_scores`）。

### 数据流

```
QEMU guest 采集 (ftrace kmalloc/kfree + trace_marker)
   → trace2csv   (trace 事件流 → CSV + marker)
   → csv2features(CSV → 90 维窗口特征, 100ms 窗口 / 50ms 步长 / 32 步序列)
   → build_pilot_dataset + pilot_gates (G1–G6+G9 数据门禁)
   → run_experiment / run_ngram 训练评估
   → final_v2_report → ACCEPTANCE_M6.md
```

## 环境架构（两套分离）

```
┌────────────────────── 真机（仿真模拟，conda/venv）──────────────────────┐
│ QEMU 采集 → trace2csv → csv2features → pilot_gates → final_v2_report    │
│ 依赖：仅 numpy（requirements.txt）                                        │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │ datasets/final-v2/processed/ 数据
┌─────────────────────────── ▼ ───────────────────────────────────────────┐
│           PyTorch Docker（模型训练，torch + scikit-learn）               │
│  run_experiment.py · run_ngram.py → runs/<experiment_id>/               │
└─────────────────────────── ▲ ───────────────────────────────────────────┘
                             │ runs/ 实验输出（回写宿主机挂载卷）
                     final_v2_report.py（真机）→ ACCEPTANCE_M6.md
```

- **真机**只跑仿真模拟：采集、构建、数据门禁、报告。唯一第三方依赖是 numpy。
- **训练**在 PyTorch Docker（自带 CUDA）中完成，训练库不装真机。
- **设备自动检测**：深度学习模型（mlp_ae / lstm_ae / lstm_vae）在构造网络时自动
  `torch.device("cuda" if torch.cuda.is_available() else "cpu")`，GPU 服务器上自动上卡；
  传统机器学习模型（ocsvm 等）天然只在 CPU 上跑。训练日志会打印
  `model=<name> device=<cpu|cuda>` 便于确认。

## 项目结构

```
heap_spray/
├── config.py                          # 全局配置（路径、CVE 列表、特征/超参）
├── requirements.txt                   # 真机依赖清单（仅 numpy）
├── models/                            # 检测模型（torch AE 包装 + sklearn）
│   ├── torch_ae.py                    # TorchAEWrapper：window/sequence 训练打分胶水
│   ├── mlp_ae.py / lstm_ae.py / lstm_vae.py   # 深度学习自编码器（nn.Module）
│   └── ocsvm.py / ngram.py / ...      # 传统/统计模型
├── scripts/
│   ├── collect/                       # QEMU 采集
│   │   ├── collect_stable.py          # normal 采集（8 类负载）
│   │   ├── collect_attack_stable.py   # attack/baseline 采集（含 trace_marker）
│   │   ├── trace_helpers/             # guest 内 ftrace 启停脚本
│   │   └── workloads/                 # 负载 C 源码（guest 内编译）
│   ├── preprocess/                    # trace2csv.py → csv2features.py
│   ├── validate/                      # build_pilot_dataset.py + pilot_gates.py + final_v2_report.py
│   ├── train/                         # run_experiment.py + run_ngram.py + common.py
│   └── visualize/                     # score_scatter.py（run/window 双粒度）
├── datasets/final-v2/                 # 正式数据集（raw / csv / processed / report）
├── runs/                              # 冻结实验输出 + ACCEPTANCE_M6.md
├── MIGRATION.md                       # 新机器复现指南（conda）
├── IMPROVEMENTS.md                    # 改进方向操作指南（P0/P1/P2）
└── KHeaps/                            # KHeaps 漏洞复现框架（guest 镜像 + PoC 源码）
```


## 数据采集（真机）

前置：QEMU（TCG 即可）、OpenSSH client、KHeaps `stretch.img`（Debian Stretch，内核 4.15，
含漏洞）。

### normal 采集（8 类负载 × 2 个 CVE 内核）

```bash
python scripts/collect/collect_stable.py -c CVE-2017-11176 -n 20 -d 30 \
    -w idle msg_msg keyctl net_busy fs_io fork_stress mem_pressure \
    --msg-sizes 256 2048 -o datasets/final-v2/raw/normal
python scripts/collect/collect_stable.py -c CVE-2017-7308  -n 20 -d 30 \
    -w idle msg_msg keyctl net_busy fs_io fork_stress mem_pressure \
    --msg-sizes 256 2048 -o datasets/final-v2/raw/normal
```

负载类别：`idle`、`msg_msg_256`、`msg_msg_2048`、`keyctl`、`net_busy`、`fs_io`、
`fork_stress`、`mem_pressure`。

### attack / baseline 采集

attack（喷雾变体）与 baseline（跑 exploit 路径但不喷雾，属正常样本）都**必须带
`--expect-crash`**：这些 PoC 在很大比例 run 里会把 guest 打崩，不带该参数崩溃 run 会被判
无效（历史回归：11176/combo 曾因此 0/15 有效）。

```bash
python scripts/collect/collect_attack_stable.py -c CVE-2017-11176 CVE-2017-7308 \
    -v poc_cfh_single_spray poc_cfh_combo -n 15 \
    --expect-crash CVE-2017-11176 CVE-2017-7308 --poc-timeout 90 \
    -o datasets/final-v2/raw/attack
python scripts/collect/collect_attack_stable.py -c CVE-2017-11176 CVE-2017-7308 \
    -v poc_cfh_baseline -n 15 \
    --expect-crash CVE-2017-11176 CVE-2017-7308 --poc-timeout 90 \
    -o datasets/final-v2/raw/baseline
```

采集要点：
- **Marker 机制**：`echo SPRAY_START > /sys/kernel/debug/tracing/trace_marker` 在 ftrace
  流中插入时间戳标记，预处理据其将 spray 阶段窗口标为 label=1、其余标 0（消除 label noise）。
- **切勿** `killall qemu-system-x86_64`：采集器只终止自己 `start_new_session` 启动的
  QEMU 进程组。
- workload/PoC **必须在 guest 内编译**（guest glibc 2.24 vs host 2.34+ 产物不互通）。

## 预处理与数据门禁（真机）

```bash
python scripts/validate/build_pilot_dataset.py \
    --attack-raw  datasets/final-v2/raw/attack \
    --normal-raw  datasets/final-v2/raw/normal \
    --baseline-raw datasets/final-v2/raw/baseline \
    --out datasets/final-v2
```

依次执行 trace2csv → baseline 并入 normal → csv2features → pilot_gates（G1–G6+G9）。
产出 `processed/{attack,normal}/features.npz` 与 `dataset_manifest.json`。日志末尾应为
**`ALL DATA GATES PASS`**，不达标不应继续训练。

## 特征（schema v2，90 维）

100ms 窗口 / 50ms 步长 / 32 步序列（覆盖 1.6s）。90 维 = 12 个 size bucket
（32,64,96,128,192,256,512,1024,2048,4096,8192,gt_8192）× 6 组 + 18 全局：

| 维度 | 内容 |
|------|------|
| 0–47 | 全局 alloc_count / free_count / alloc_rate / free_rate（×12 bucket） |
| 48–60 | 13 个全局统计（total/burst/熵/相关性等） |
| 61–72 | top_task alloc_count（×12 bucket，最活跃进程） |
| 73–84 | top_task free_count（×12 bucket） |
| 85–89 | 5 个 timing（窗口/序列时序特征） |

- kfree 不含 size，通过维护 `ptr→size` 映射表恢复（alloc 时记录，free 时查表）。
- 攻击数据用正常数据的 mean/std 做 z-score 归一化（防攻击分布泄露）。
- 门禁 G6 保证每个 CVE 的目标 slab 在 bucket 覆盖内；G9 保证没有按进程身份/时长/空窗口
  比例的"捷径"特征。

## 模型（M6 冻结 5 个，均 base、无 LOO）

| 模型 | 类型 | 评分单元 |
|------|------|---------|
| **ocsvm**（主检测） | One-Class SVM | 窗口/序列级 |
| **mlp_ae**（高吞吐替代） | MLP 自编码器 | 窗口级 |
| lstm_ae | LSTM 自编码器 | 序列级 |
| lstm_vae | LSTM 变分自编码器（β 退火 + free bits） | 序列级 |
| ngram | run 级 token 3-gram + KL 散度 | run 级 |

深度学习模型自动使用 GPU（`torch.device` 自动检测）；sklearn 模型留 CPU。CPU 路径保持
单线程、固定 seed 可复现；CUDA 训练不逐位可复现。

## 训练与评估（PyTorch Docker）

```bash
DATA=datasets/final-v2
PROC_A=$DATA/processed/attack; PROC_N=$DATA/processed/normal

# 窗口/序列级模型
python scripts/train/run_experiment.py --model ocsvm --attack-data $PROC_A \
    --normal-data $PROC_N --dataset-manifest $DATA/dataset_manifest.json --out runs
python scripts/train/run_experiment.py --model mlp_ae   ...  # 同上
python scripts/train/run_experiment.py --model lstm_ae  ...
python scripts/train/run_experiment.py --model lstm_vae ...

# run 级 token 模型（需 CSV 根目录）
python scripts/train/run_ngram.py --attack-data $PROC_A --normal-data $PROC_N \
    --normal-csv-root $DATA/csv/normal --attack-csv-root $DATA/csv/attack \
    --dataset-manifest $DATA/dataset_manifest.json --out runs
```

每个模型产出 `runs/<时间戳>_<模型>_s<seed>_<序号>/`（model.pkl、evaluation_report.json、
gates.json、split_manifest.json、metrics.csv、scaler.npz），含输入 sha256、git revision、
配置摘要，可追溯。默认 `--seed 42`。

评估：阈值 = 验证集正常 p99；逐 CVE AUC = 攻击内 spray-seq vs 上下文；run 级 = 每 run 最大
窗口分数。

### 报告与冻结（真机）

```bash
python scripts/validate/final_v2_report.py --runs runs \
    --dataset datasets/final-v2 --out runs/ACCEPTANCE_M6.md
```

## M6 冻结结果（seed=42，同一 run 划分与阈值标定）

数据集：460 run / 416 有效，8 类 normal × 2 CVE 各 20、attack 2 变体 × 2 CVE 各 ≥15、
baseline 各 15；门禁全 PASS。

| 模型 | run AUC | run F1 | run FPR | run rec | seq AUC | 7308 | 11176 |
|------|--------|--------|---------|---------|---------|------|-------|
| **ocsvm** | **0.915** | **0.891** | 0.197 | **1.000** | 0.988 | 0.981 | 0.936 |
| mlp_ae | 0.903 | 0.674 | 0.152 | 0.604 | 0.984 | 0.996 | 0.850 |
| lstm_ae | 0.897 | 0.615 | 0.152 | 0.528 | 0.978 | 0.997 | 0.817 |
| lstm_vae | 0.882 | 0.457 | 0.015 | 0.302 | 0.968 | 0.992 | 0.745 |
| ngram | 0.743 | 0.346 | 0.000 | 0.209 | — | 0.580 | 0.892 |

**结论（冻结）**：ocsvm 为主检测模型（run AUC 0.915 最高、run recall 1.000 不漏检、
双 CVE 逐 CVE AUC 均 ≥0.9、G10 全 PASS）；代价是 run FPR 0.197 偏高、吞吐最低
（2,337 窗口/s）。对误报敏感或需在线判定时用 **mlp_ae**（187,822 窗口/s、seq AUC 0.984）。
5 模型 G10 全 PASS（0 个 baseline 误报）证明检测捕获的是"喷雾现象"而非"存在 exploit"。
详细论证见 `runs/ACCEPTANCE_M6.md`。

## 可视化

`scripts/visualize/score_scatter.py` 对冻结模型画异常分数散点，`--level` 控制粒度：

- **run 级**（默认）：每 run 一个点，x = run 级分数（窗口 max 聚合）。
- **window 级**：每个 50ms 窗口一个点（去重后约 24k normal + 10k attack），x = 距 run
  开始的秒数，可看到分数轨迹与喷雾尖峰（集中在 run 后段）。这是模型能达到的最细粒度；
  原始 kmalloc/kfree 事件被聚合进窗口特征，无逐 event 分数。

```bash
python scripts/visualize/score_scatter.py \
    --run runs/2026-08-21_ocsvm_v2_ocsvm_s42_072736 \
    --attack-data datasets/final-v2/processed/attack \
    --normal-data datasets/final-v2/processed/normal \
    --level window --out datasets/final-v2/report/score_scatter_ocsvm_window.png
```

## 依赖

- **真机**（`requirements.txt`）：仅 `numpy==2.5.2`。
- **训练 Docker**：torch（CUDA 版）、scikit-learn、scipy、numpy、matplotlib、pandas。

## 参考

- [KHeaps: Reproducible Kernel Heap Exploitation](https://www.usenix.org/conference/usenixsecurity22/presentation/zeng) — 漏洞复现框架
- ftrace — Linux 内核事件追踪机制
