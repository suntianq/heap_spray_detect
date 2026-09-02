# 复现步骤

## 前置条件

### 真机环境（采集 + 预处理）

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt  # 仅 numpy
```

系统依赖：QEMU（TCG 即可）、OpenSSH、KHeaps `stretch.img`（Debian Stretch，内核 4.15）。

### 训练环境（GPU 服务器）

```bash
pip install torch scikit-learn scipy numpy matplotlib pandas
# 可选：OCSVM 的 GPU 加速后端（装了自动启用，未装回落 sklearn）
pip install thundersvm
```

ocsvm 后端自动选择：检测到 `thundersvm` 时用 CUDA 加速 OneClassSVM，
否则用 sklearn（CPU）。可用环境变量 `HEAPSPRAY_SVM_BACKEND=sklearn` 强制 CPU。

GPU 自动检测：`torch.cuda.is_available()` → CUDA / CPU 自动切换，无需手动配置。

---

## 一、数据采集（真机，可选）

已有 `datasets/raw/`（6.1G，3 个 CVE）可跳过此步。

### 采集单个 CVE

```bash
CVE=CVE-2017-7533 nohup scripts/collect/collect_cve_complete.sh \
    > datasets/.m6/logs/collect_CVE-2017-7533_complete.log 2>&1 &
```

依次采集 attack（single_spray + combo）→ baseline → normal（7 类 × 20 run），可断点续跑。

### 完整流水线采集

```bash
nohup scripts/collect/run_final_v2.sh > datasets/m6_run.log 2>&1 &
```

---

## 二、预处理

### 2.1 特征构建（schema v3，101 维）

```bash
python3 scripts/validate/build_pilot_dataset.py --raw datasets/raw --out datasets
```

流程：trace2csv（含 cpu 列）→ csv2features（90+11 维）→ pilot_gates（G1-G6+G9）→ dataset_manifest.json。日志末尾应为 `ALL DATA GATES PASS`。

### 2.2 Token 序列构建（阶段 2）

```bash
python3 scripts/preprocess/trace2tokens.py --raw datasets/raw --out datasets
```

生成 `datasets/processed/{attack,normal}/token_sequences.npz`：
- token = (op, size_bucket, behavior_type, frequency_rank, dt_bucket)
- 词表 1536，序列长度 128，50% 重叠
- call_site 画像和 dt 分位数从正常数据自动标定

---

## 三、训练与评估

### 3.1 单模型训练

```bash
DATA=datasets
PROC_A=$DATA/processed/attack
PROC_N=$DATA/processed/normal

# 窗口特征模型（schema v3，101 维）
python3 scripts/train/run_experiment.py --model ocsvm \
    --attack-data $PROC_A --normal-data $PROC_N \
    --dataset-manifest $DATA/dataset_manifest.json --out runs

# 事件级 GRU 模型（token 序列）
python3 scripts/train/run_experiment.py --model gru \
    --attack-data $PROC_A --normal-data $PROC_N \
    --dataset-manifest $DATA/dataset_manifest.json --out runs

# 统一 GRU + Deep SVDD 双头（全 GPU，仅 token 序列）
python3 scripts/train/run_experiment.py --model fusion_svdd \
    --attack-data $PROC_A --normal-data $PROC_N \
    --dataset-manifest $DATA/dataset_manifest.json --out runs
```

每个模型产出 `runs/<时间戳>_<模型>_s<seed>/`：
- `model.pkl` — 训练好的模型
- `evaluation_report.json` — run 级指标（AUC/F1/Precision/Recall）+ 分组明细 + 诊断计数
- `gates.json` — G7/G8/G10 门禁
- `scaler.npz` — 归一化参数（token 模型无）
- `split_manifest.json` — train/val/test run 划分
- `experiment_config.json` — 输入哈希、git 版本、配置

### 3.2 跨 CVE 泛化

```bash
# 训练 11176+7308，测试 2636
python3 scripts/train/run_cve_split.py --model gru \
    --train-cves CVE-2017-11176 CVE-2017-7308 --test-cves CVE-2017-2636 \
    --attack-data $PROC_A --normal-data $PROC_N \
    --dataset-manifest $DATA/dataset_manifest.json --out runs --name cveAB_testC

# 融合模型跨 CVE
python3 scripts/train/run_cve_split.py --model fusion \
    --train-cves CVE-2017-11176 CVE-2017-7308 --test-cves CVE-2017-2636 \
    --attack-data $PROC_A --normal-data $PROC_N \
    --dataset-manifest $DATA/dataset_manifest.json --out runs --name cveAB_testC
```

### 3.3 结果对比

```bash
python3 scripts/validate/compare_models.py --runs runs --out results/model_comparison.csv
# 可选: --model-filter ocsvm gru fusion
```

---

## 四、模型说明

| 模型 | 类型 | 输入 | 评分方式 | 说明 |
|------|------|------|---------|------|
| ocsvm | 窗口特征 | features.npz (101维) | -score_samples | 基线（ThunderSVM GPU 自动适配） |
| lstm_ae | 序列特征 | features.npz (32×101) | 重建误差 | 序列级基线 |
| gru | 事件 token | token_sequences.npz (128) | top-g 违例率 | 结构轴 |
| fusion_svdd | 统一双头 | token_sequences.npz (128) | 违例率 + SVDD 距离 | 主力，全 GPU |

### schema v3 新增特征（11 维）

| 特征 | 说明 |
|------|------|
| reclaim_count / cross_site_reclaim_count / reclaim_rate | free→alloc 同 ptr 50ms 内回收 |
| cpu_alloc_entropy / top_cpu_alloc_fraction | CPU 集中度（核数自适应） |
| lifetime_median / lifetime_p90 / short_lived_count / long_lived_count | 对象存活时间分布 |
| burst_dominant_bucket_ratio / burst_dominant_callsite_ratio | 1ms 最密集子窗口浓度 |

### GRU token 定义

```
token = (op, size_bucket, behavior_type, frequency_rank, dt_bucket)
  op:             ALLOC=0, FREE=1
  size_bucket:    0-11 (32...gt_8192)
  behavior_type:  mono/narrow/broad/unknown (从正常数据计算)
  frequency_rank: top5%/p80/p50/rare (从正常数据计算)
  dt_bucket:      <2us/2-50us/50-1000us/>1ms (从正常数据标定)
```

---

## 五、测试

```bash
PYTHONPATH=tests:scripts:scripts/preprocess .venv/bin/python3 -m unittest \
    tests.test_csv2features tests.test_trace2csv tests.test_collection_common \
    tests.test_gru_detector tests.test_fusion -v
```

共 89 个测试，覆盖：
- 特征提取（schema v3 新增 11 维 + 原有 90 维）
- trace 解析（含 cpu 列）
- GRU 模型（shape/违例敏感度/save-load/token 编码）
- Fusion 模型（quantile 对齐/双轴融合/save-load）
- 采集公共逻辑（marker 验证/trace 校验）

---

## 六、可视化

```bash
python3 scripts/visualize/score_scatter.py \
    --run runs/<实验目录> \
    --attack-data datasets/processed/attack \
    --normal-data datasets/processed/normal \
    --level window --out datasets/report/score_scatter.png
```

---

## 七、环境说明

| 环境 | 用途 | GPU |
|------|------|-----|
| 真机（.venv，仅 numpy） | 采集、预处理、数据门禁 | 不需要 |
| GPU 服务器 | 全量训练、评估、报告 | 自动检测 |

模型自动选择设备：
- `torch.cuda.is_available()` → CUDA
- 否则 → CPU（`torch.set_num_threads(1)` 保证可复现）
