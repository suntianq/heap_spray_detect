# Kernel Heap Spray Detection

基于内核动态内存分配行为的堆喷射（Heap Spray）攻击检测系统。通过 ftrace 采集 Linux
内核 `kmalloc`/`kfree` 事件，提取时序特征，用半监督（单类学习）模型检测异常分配模式。

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
   → trace2csv   (trace 事件流 → CSV + marker, 临时中转, 构建后删除)
   → csv2features(CSV → 90 维窗口特征, 100ms 窗口 / 50ms 步长 / 32 步序列)
   → build_pilot_dataset + pilot_gates (G1–G6+G9 数据门禁)
   → run_experiment 训练评估
   → final_v2_report → ACCEPTANCE_M6.md
```

## 目录结构

```
heap_spray/
├── config.py                          # 全局配置（路径、CVE 列表、特征/超参）
├── requirements.txt                   # 真机依赖清单（仅 numpy）
├── models/                            # 检测模型（torch AE 包装 + sklearn）
│   ├── torch_ae.py                    # TorchAEWrapper：window/sequence 训练打分胶水
│   ├── mlp_ae.py / lstm_ae.py / lstm_vae.py   # 深度学习自编码器（nn.Module）
│   └── ocsvm.py / isolation_forest.py / lof_detector.py / pca_detector.py / stat_threshold.py
├── scripts/
│   ├── collect/                       # QEMU 采集
│   │   ├── collect_stable.py          # normal 采集（8 类负载）
│   │   ├── collect_attack_stable.py   # attack/baseline 采集（含 trace_marker）
│   │   ├── collect_cve_complete.sh    # 单个新 CVE 的全量采集编排（可续跑）
│   │   ├── run_final_v2.sh            # 完整流水线编排（采集+build+train+report）
│   │   ├── trace_helpers/             # guest 内 ftrace 启停脚本
│   │   └── workloads/                 # 负载 C 源码（guest 内编译）
│   ├── preprocess/                    # trace2csv.py → csv2features.py
│   ├── validate/
│   │   ├── build_pilot_dataset.py     # raw → processed（含 class 剥离、csv 清理）
│   │   ├── pilot_gates.py             # G1–G6+G9 数据门禁
│   │   ├── migrate_datasets.py        # 旧 final-v2 布局 → CVE-first 布局
│   │   ├── cross_cve_aggregate.py     # 跨 CVE 实验汇总表
│   │   └── final_v2_report.py         # 验收报告
│   ├── train/                         # run_experiment.py + run_cve_split.py + common.py
│   └── visualize/                     # score_scatter.py / demo_figure.py
├── datasets/
│   ├── raw/<CVE>/{attack,normal,baseline}/<variant|workload>/run_XXX_*/   # 原始 trace + manifest
│   ├── processed/{attack,normal}/     # features.npz（90 维特征 + 序列）
│   └── dataset_manifest.json          # run 注册表
├── runs/                              # 实验输出 + ACCEPTANCE_M6.md
└── KHeaps/                            # KHeaps 漏洞复现框架（MIT，guest 镜像 + PoC 源码）
```

## 数据采集

前置：QEMU（TCG 即可）、OpenSSH client、KHeaps `stretch.img`（Debian Stretch，内核 4.15，
含漏洞）。

### 布局约定

数据按 **CVE 优先**组织（`datasets/raw/`），每个 CVE 下分三类：

```
datasets/raw/CVE-2017-11176/
├── attack/    poc_cfh_single_spray/run_000_<uuid>/{manifest.json, trace.log, ...}
│              poc_cfh_combo/run_000_<uuid>/...
├── normal/    idle/run_000_<uuid>/...
│              msg_msg_256/run_000_<uuid>/...
└── baseline/  poc_cfh_baseline/run_000_<uuid>/...
```

- **attack**：执行漏洞触发 + 堆喷，正样本（带 trace_marker 标记）。
- **normal**：正常系统负载（8 类），负样本。
- **baseline**：跑 exploit 触发路径但不喷雾，负样本中的对照组——用于 G10 门禁验证
  检测器抓的是"喷雾现象"而非"存在 exploit"。

### 采集单个新 CVE（推荐，可续跑）

```bash
CVE=CVE-2017-7533 nohup scripts/collect/collect_cve_complete.sh \
    > datasets/.m6/logs/collect_CVE-2017-7533_complete.log 2>&1 &
```

依次采集 attack（single_spray + combo）→ baseline → normal（7 类 × 20 run），每阶段打
`.done` 标记可断点续跑。`MIN_VALID`、`NORMAL_RUNS` 等可用环境变量覆盖。

### 手动采集

```bash
# normal（8 类负载）
python scripts/collect/collect_stable.py -c CVE-2017-11176 -n 20 -d 30 \
    -w idle msg_msg keyctl net_busy fs_io fork_stress mem_pressure \
    --msg-sizes 256 2048 -o datasets/raw

# attack（喷雾变体）与 baseline——都必须带 --expect-crash
python scripts/collect/collect_attack_stable.py -c CVE-2017-11176 CVE-2017-7308 \
    -v poc_cfh_single_spray poc_cfh_combo -n 15 \
    --expect-crash CVE-2017-11176 CVE-2017-7308 --poc-timeout 90 -o datasets/raw
python scripts/collect/collect_attack_stable.py -c CVE-2017-11176 CVE-2017-7308 \
    -v poc_cfh_baseline -n 15 \
    --expect-crash CVE-2017-11176 CVE-2017-7308 --poc-timeout 90 -o datasets/raw
```

负载类别：`idle`、`msg_msg_256`、`msg_msg_2048`、`keyctl`、`net_busy`、`fs_io`、
`fork_stress`、`mem_pressure`。

**`--expect-crash` 是必需的**：这些 PoC 在很大比例 run 里会把 guest 打崩，不带该参数崩溃
run 会被判无效（历史回归：11176/combo 曾因此 0/15 有效）。

采集要点：
- **Marker 机制**：`echo SPRAY_START > /sys/kernel/debug/tracing/trace_marker` 在 ftrace
  流中插入时间戳标记，预处理据其将 spray 阶段窗口标为 label=1、其余标 0（消除 label noise）。
- **切勿** `killall qemu-system-x86_64`：采集器只终止自己 `start_new_session` 启动的
  QEMU 进程组。
- workload/PoC **必须在 guest 内编译**（guest glibc 2.24 vs host 2.34+ 产物不互通）。
- 每个新 CVE 采集后先跑 build 门禁确认 target slab 覆盖（2636 曾计划 4096、实际回收
  kmalloc-8192）。

## 预处理与数据门禁

```bash
python scripts/validate/build_pilot_dataset.py --raw datasets/raw --out datasets
```

依次执行 trace2csv（整个 raw）→ 按 class 拆分 staging（剥离 class 段，得到
`CVE/<variant|workload>/run_XXX_*/trace` 形式的 run_id）→ baseline 并入 normal →
csv2features → pilot_gates（G1–G6+G9）。产出 `processed/{attack,normal}/features.npz`
与 `dataset_manifest.json`。**csv 是临时中转，构建结束后自动删除**。日志末尾应为
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

## 模型

| 模型 | 类型 | 评分单元 |
|------|------|---------|
| **ocsvm**（主检测） | One-Class SVM | 窗口/序列级 |
| **mlp_ae**（高吞吐替代） | MLP 自编码器 | 窗口级 |
| lstm_ae | LSTM 自编码器 | 序列级 |
| lstm_vae | LSTM 变分自编码器（β 退火 + free bits） | 序列级 |

深度学习模型自动使用 GPU（`torch.device` 自动检测）；sklearn 模型留 CPU。CPU 路径保持
单线程、固定 seed 可复现；CUDA 训练不逐位可复现。

## 训练与评估

```bash
DATA=datasets
PROC_A=$DATA/processed/attack; PROC_N=$DATA/processed/normal

python scripts/train/run_experiment.py --model ocsvm --attack-data $PROC_A \
    --normal-data $PROC_N --dataset-manifest $DATA/dataset_manifest.json --out runs
python scripts/train/run_experiment.py --model mlp_ae  ...
python scripts/train/run_experiment.py --model lstm_ae ...
```

每个模型产出 `runs/<时间戳>_<模型>_s<seed>_<序号>/`（model.pkl、evaluation_report.json、
gates.json、split_manifest.json、metrics.csv、scaler.npz），含输入 sha256、git revision、
配置摘要，可追溯。默认 `--seed 42`。

评估：阈值 = 验证集正常 p99；逐 CVE AUC = 攻击内 spray-seq vs 上下文；run 级 = 每 run 最大
窗口分数。

### 跨 CVE 实验

`run_cve_split.py` 按 CVE 做训练/测试划分，支持任意 train/test CVE 组合：

```bash
python scripts/train/run_cve_split.py --model ocsvm \
    --train-cves CVE-2017-11176 CVE-2017-7308 --test-cves CVE-2017-2636 \
    --attack-data $PROC_A --normal-data $PROC_N \
    --dataset-manifest $DATA/dataset_manifest.json --out runs --name cveAB_testC
```

### 报告

```bash
python scripts/validate/final_v2_report.py --runs runs --dataset datasets \
    --out runs/ACCEPTANCE_M6.md
```

## 可视化

`scripts/visualize/score_scatter.py` 对冻结模型画异常分数散点，`--level` 控制粒度：

- **run 级**（默认）：每 run 一个点，x = run 级分数（窗口 max 聚合）。
- **window 级**：每个 50ms 窗口一个点（去重后约 24k normal + 10k attack），x = 距 run
  开始的秒数，可看到分数轨迹与喷雾尖峰（集中在 run 后段）。这是模型能达到的最细粒度；
  原始 kmalloc/kfree 事件被聚合进窗口特征，无逐 event 分数。

```bash
python scripts/visualize/score_scatter.py \
    --run runs/2026-08-31_ocsvm_v2_ocsvm_s42_034045 \
    --attack-data datasets/processed/attack \
    --normal-data datasets/processed/normal \
    --level window --out datasets/report/score_scatter_ocsvm_window.png
```

## 依赖

- **真机**（`requirements.txt`）：仅 `numpy==2.5.2`。
- **训练 Docker**：torch（CUDA 版）、scikit-learn、scipy、numpy、matplotlib、pandas。

## 参考

- [KHeaps: Reproducible Kernel Heap Exploitation](https://www.usenix.org/conference/usenixsecurity22/presentation/zeng) — 漏洞复现框架
- ftrace — Linux 内核事件追踪机制
