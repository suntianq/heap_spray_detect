# Kernel Heap Spray Detection v2 实施计划

## 1. 背景与目标

当前项目中的代码、数据集、模型和报告存在多个版本并存的问题：旧数据主要使用 32/49 维特征和整条攻击 trace 标注，新版代码则逐步迁移到 run-aware、marker-aware 的 schema v2。若继续直接覆盖 `features.npz`、模型权重和评估报告，将难以判断实验结果属于哪一代数据与实现，也无法可靠复现实验。

本计划的目标是建立一条可追溯、无数据泄漏、可重复执行的 v2 流水线：

1. 冻结并明确标识现有 legacy 数据和产物。
2. 优先修复预处理、标签和时间序列语义。
3. 重写攻击采集器，保证每个 run 独立、完整、可校验。
4. 通过小规模 pilot 验证流水线，再扩大正式采集。
5. 按 run 和 CVE 重构训练与评估，验证对未知攻击的泛化能力。

本计划不删除现有数据，不使用旧数据作为最终测试集，也不以测试集 Best F1 作为模型选择依据。

## 2. 实施原则

- **数据不可覆盖**：每次采集和实验使用唯一目录。
- **run 是最小隔离单位**：序列、数据划分、统计和评估均不得跨 run。
- **训练集拟合一切统计量**：`log1p` 参数、scaler、PCA 等只在训练 run 上拟合。
- **标签和异常分数语义一致**：endpoint 标签对应 last-window 分数，sequence-any 标签对应 max/p90 分数。
- **采集失败即关闭**：marker、trace、manifest 或一致性校验失败时，不进入有效数据集。
- **最终测试不可反复查看**：最终测试 CVE 和测试 run 在模型开发期间保持封存。
- **先基线、后复杂模型**：先证明数据与评估可靠，再优化 LSTM/VAE。

## 3. 目标目录结构

```text
heap_spray/
├── datasets/
│   ├── legacy-v1/
│   │   ├── manifest.json
│   │   └── README.md
│   ├── pilot-v2/
│   │   ├── raw/
│   │   ├── csv/
│   │   ├── processed/
│   │   └── dataset_manifest.json
│   └── final-v2/
│       ├── raw/
│       ├── csv/
│       ├── processed/
│       └── dataset_manifest.json
├── runs/
│   ├── 2026-08-xx_statistical_v2_<id>/
│   ├── 2026-08-xx_lstm_ae_v2_<id>/
│   └── ...
├── scripts/
├── models/
├── data/                    # 保留，作为 legacy 原始位置
└── results/                 # 保留，作为 legacy 产物位置
```

每个训练 run 目录至少保存：

```text
runs/<experiment_id>/
├── experiment_config.json
├── dataset_manifest.json
├── split_manifest.json
├── scaler.npz
├── model.*
├── train_report.json
├── evaluation_report.json
├── metrics.csv
└── logs/
```

## 4. 第一阶段：冻结现有数据和产物

### 4.1 数据分类

| 现有内容 | 定义 | 允许用途 | 禁止用途 |
|---|---|---|---|
| 无 marker 攻击数据 | `legacy-v1 attack` | 探索、画图、回归测试 | 最终指标与最终测试 |
| 当前 marker 数据 | `marker pilot` | marker 解析、窗口标签测试 | 直接代表正式攻击集 |
| 现有正常数据 | `legacy-v1 normal` | 开发、单元测试、管线调试 | final-v2 最终测试集 |
| 32/49 维模型和报告 | `legacy artifact` | 回归对照 | 与 v2 指标直接比较 |

### 4.2 工作项

- 新建 `datasets/legacy-v1/manifest.json`，记录旧目录、schema、特征维度、标签策略和已知限制。
- 保持现有 `data/`、`results/` 原地不动，不删除、不覆盖。
- 新代码默认写入 `datasets/pilot-v2/` 和 `runs/<experiment_id>/`。
- 输出目录已存在且非空时默认拒绝执行，只有显式参数才能恢复或覆盖。
- 所有数据集和实验报告写入 `schema_version`、Git revision（若可用）、配置摘要和输入文件哈希。

### 4.3 验收标准

- 旧文件的路径、大小和哈希保持不变。
- 新流程不会默认读取旧模型或覆盖旧报告。
- 任意模型指标均能追溯到确定的数据集 manifest 和 split manifest。

## 5. 第二阶段：修复预处理管线（最高优先级）

### 5.1 每条 trace 独立处理

固定流程：

```text
单个 CSV
  → 按时间排序并补充 FREE 元数据
  → 构造严格连续的固定时间窗口
  → 仅在本 run 内构造序列
  → 附加 run/CVE/variant/workload 元数据
  → 汇总保存
```

禁止先拼接不同文件的窗口，再从拼接结果构造序列。每条 sequence 必须保存 `run_id`，并通过断言保证其所有窗口来自同一 run。

### 5.2 保留空窗口

- 窗口大小：100 ms。
- 窗口步长：50 ms。
- 时间轴从 trace 有效起点连续推进到 trace 有效终点。
- 无事件窗口仍保存全零计数特征。
- 额外保存：
  - `event_count`
  - `is_empty`
  - `time_since_last_event_ms`

在 100 ms 窗口、50 ms 步长、32 步序列下，序列覆盖范围应为：

```text
100 ms + 31 × 50 ms = 1650 ms
```

### 5.3 修复 FREE size 恢复

在整条 trace 上按时间维护 `ptr → allocation` 状态，为 FREE 事件补充：

- `resolved_bytes_alloc`
- `allocation_timestamp_ns`
- `object_lifetime_ns`
- `size_resolved`

无法恢复的 FREE：

- 不进入具体 size bucket；
- 计入 `free_unknown`；
- 在 run 统计中记录数量和比例；
- 超过可配置阈值时将该 run 标记为数据质量异常。

### 5.4 扩展 size bucket

使用实际 `bytes_alloc` 映射 slab class：

```text
[32, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096, 8192]
```

另设 `gt_8192` overflow bucket。`bytes_req` 不替代 `bytes_alloc`，可作为独立统计或用于检查请求大小与实际分配大小的差异。

### 5.5 修复窗口特征

- alloc/free rate 固定除以 100 ms，不使用窗口内首尾事件间隔。
- burst 的 1 ms bin 固定在完整窗口时间轴上。
- 区分 TID 和 TGID；ftrace 无法提供 TGID 时明确标为未知，不把 TID 重命名为 PID。
- 保存 top task 的 `task_id` 和 `comm` 作为窗口元数据。
- 统计 top task 为 PoC、`systemd`、`sshd`、`kworker` 等进程的比例。
- 计数特征可在训练阶段应用 `log1p`，但预处理阶段只保存原始值。
- 标准化参数只能由训练 run 拟合。

### 5.6 标签重新定义

窗口与 spray 区间的重叠比例定义为（比例同时相对窗口与 spray 两个区间取并集）：

```text
overlap >= 50% 窗口  或  overlap >= 50% spray  → 1（attack）
overlap == 0    → 0（normal）
两者都 < 50%    → -1（boundary/ignore）
```

> 说明：pilot 中 PoC 的 spray 时长在 0.1–175 ms，远小于 100 ms 窗口。若只按窗口比例
> 判定，<50 ms 的 spray 不会产生任何 label-1 窗口，攻击对训练不可见；若只按 spray 比例
> 判定，跨多个窗口的长 spray（如 500 ms）又无窗口可覆盖其 50%。取并集后两种情形都能
> 产生攻击窗口，仅真正落在两个区间边缘的窗口保留 -1 boundary。

支持两种明确的序列任务：

| 任务 | 序列标签 | 推理分数 |
|---|---|---|
| 当前时刻是否处于攻击 | 最后一个窗口标签 | 最后一个窗口分数 `last` |
| 序列内是否发生攻击 | 窗口标签 max/any | `max` 或预先确定的 `p90` |

boundary 窗口或包含 boundary 的序列必须按策略显式忽略，不能自动当作正常或攻击。

### 5.7 预处理输出 schema

每个 processed 数据集至少包含：

- 原始窗口特征、窗口标签和窗口起始时间；
- sequences、sequence 标签和 sequence 起始时间；
- `run_id`、CVE、variant、workload、class；
- feature name 和 feature group；
- schema version、窗口参数、标签策略；
- 每个 run 的解析错误数、unknown free 数、空窗口比例和 marker 状态。

### 5.8 验收标准

- 任意 sequence 不跨 run。
- 空窗口不会被跳过。
- 所有 sequence 时间跨度均为 1650 ms（允许 trace 边界按明确策略处理）。
- FREE size 状态跨窗口连续维护。
- 攻击数据缺少完整 marker 时 fail closed。
- 预处理结果中不存在 NaN/Inf。
- 相同输入与配置生成相同哈希的输出。

## 6. 第三阶段：重写采集器

### 6.1 每个 run 生成 manifest

建议格式：

```json
{
  "dataset_version": "v2",
  "run_uuid": "...",
  "class": "attack",
  "cve": "CVE-...",
  "variant": "poc_cfh_single_spray",
  "workload": null,
  "kernel_hash": "...",
  "image_hash": "...",
  "poc_hash": "...",
  "qemu_pid": 1234,
  "trace_start_ns": 0,
  "spray_start_ns": 0,
  "spray_end_ns": 0,
  "trace_end_ns": 0,
  "poc_exit_code": 0,
  "vm_crashed": false,
  "event_count": 12345,
  "trace_overrun": 0,
  "status": "valid"
}
```

预处理默认只读取 `status=valid` 的 run。

### 6.2 Marker 放入实际 spray 操作

推荐时间线：

```text
TRACE_START
  → 3 秒正常预热
  → PoC 内 SPRAY_START
  → 实际 spray 操作
  → PoC 内 SPRAY_END
  → 3 秒正常尾部
TRACE_END
```

marker 应由 PoC 在实际 spray 函数入口和出口写入 `trace_marker`。宿主机不得用“启动 PoC 前写 START、固定睡眠后写 END”替代真实 spray 边界。

### 6.3 强校验采集结果

以下任一情况均将 run 标记为无效并按策略重试：

- tracepoint 启用失败；
- PoC 上传、启动或执行失败；
- marker 缺失、重复或顺序错误；
- `SPRAY_END <= SPRAY_START`；
- 事件数低于阈值；
- trace buffer overrun；
- trace、日志或 manifest 下载失败；
- kernel/image/CVE/PoC 哈希与计划不一致；
- 非预期 VM crash；
- 预期 crash 的 run 未成功持久化 crash 前 trace。

对预期 crash 的 CVE，应让 `trace_pipe` 持续流向宿主机或可靠的共享存储，不能只在 VM crash 后执行 SCP。

### 6.4 只终止本次 QEMU

- 保存 `Popen` 对象和 process group。
- 正常结束先发送 SIGTERM，超时后再向该 process group 发送 SIGKILL。
- 禁止使用 `killall qemu-system-x86_64` 或按模糊命令行匹配终止进程。

### 6.5 唯一输出目录

```text
raw/<CVE>/<variant>/<run_uuid>/
├── trace.log
├── trace_stats.txt
├── manifest.json
├── poc.stdout
├── poc.stderr
└── qemu.log
```

失败 run 保留并标记原因，不覆盖成功 run，也不自动进入 processed 数据集。

### 6.6 验收标准

- 每个采集进程只管理自身 QEMU。
- 每个 run 均有完整 manifest 和唯一 UUID。
- marker 顺序和数量严格正确。
- trace overrun 为 0。
- 失败 run 不会进入有效数据集。
- 重复执行不会覆盖任何既有 run。

## 7. 第四阶段：小规模 pilot-v2

### 7.1 攻击样本

先选择两个 CVE：

- 一个小对象 spray，例如 kmalloc-256；
- 一个大对象 spray，例如 kmalloc-2048 或 kmalloc-4096。

每个 CVE：

- `single_spray` 和 `combo` 两种 variant；
- 每种至少 5 个有效 run；
- 无效 run 不计入目标数量。

### 7.2 匹配正常对照

- idle；
- msg_msg；
- keyctl；
- packet/network busy；
- filesystem 压力；
- 每个 CVE 的 `poc_cfh_baseline`，即执行漏洞路径但不进行 spray。

正常和攻击应尽量匹配：

- VM 启动阶段；
- 预热与尾部时长；
- 内核、镜像和 CPU 配置；
- tracepoint 配置；
- 总采集时长。

### 7.3 Pilot 验收门槛

- [ ] 所有 sequence 不跨 run。
- [ ] marker 成功率接近 100%。
- [ ] 空窗口得到保留。
- [ ] 每条 sequence 时间跨度正确。
- [ ] top task/comm 能观察到 PoC 活动。
- [ ] size bucket 覆盖目标 slab。
- [ ] 同一 run 不跨 train/validation/test。
- [ ] 简单统计模型可完整训练和评估。
- [ ] 模型不能仅凭采集时长、空窗口比例或文件长度区分类别。
- [ ] baseline PoC 不会被轻易当作 spray。

Pilot 未通过前，不启动全量 CVE 采集。

## 8. 正式数据采集建议

### 8.1 正常数据

每类至少采集 20～30 个独立有效 run：

- idle；
- msg_msg；
- keyctl；
- packet/network busy；
- filesystem I/O；
- process/fork stress；
- 内存压力；
- 无 spray 的 PoC baseline。

正常数据必须覆盖高分配率、高突发和单进程主导等困难负样本，避免模型只学习“分配多就是攻击”。

### 8.2 攻击数据

理想配置：

- 每个 CVE；
- 每个 spray variant；
- 每种至少 15～20 个有效 run；
- 变化 CPU pin、spray 数量、spray 速度和重复次数；
- 保留三类执行状态：
  - 成功利用；
  - 利用失败但发生 spray；
  - 未发生 spray。

若成本有限，首版选择 4～6 个覆盖不同 slab 大小和 spray 技术的 CVE，再逐步扩展。

## 9. 第五阶段：重构训练与评估

### 9.1 按 run 划分数据

正常数据建议：

```text
Train runs       70%
Validation runs  15%
Test runs        15%
```

严格流程：

```text
划分 run
  → 仅用 train run 拟合 log/scaler
  → 转换 train/validation/test
  → 训练模型
  → 用正常 validation 标定阈值
  → 冻结模型和阈值
  → 最终只评估一次 test
```

split manifest 应保存 run UUID、workload、CVE、variant、随机种子和分层策略。

### 9.2 按 CVE 管理攻击数据

- **开发攻击 CVE**：用于选择特征、模型和分数聚合方式。
- **最终测试 CVE**：开发期间不查看结果。
- **Leave-one-CVE-out**：每次留出一个完整 CVE，评估未知攻击泛化能力。
- 同一 CVE/variant/run 的任何窗口或序列不得出现在多个 partition。

### 9.3 模型修改顺序

1. Statistical Threshold。
2. PCA。
3. Isolation Forest。
4. MLP-AE。
5. LSTM-AE。
6. LSTM-VAE。
7. N-gram。

具体要求：

- LSTM-AE 使用正确的 BiLSTM final hidden states。
- 明确区分因果单向 LSTM 和离线双向 LSTM；部署场景不得使用未来窗口信息。
- VAE 推理使用 `z=mu`，或固定次数采样后平均，保证可重复。
- 重建损失支持按特征组加权和分组报告。
- 阈值按正常验证集目标 FPR 标定，例如 p99，而不是优化测试集 Best F1。
- Best F1 只能作为带有明确 `oracle/test-only` 标识的参考上界。
- N-gram 正确处理 OOV、平滑和固定窗口评分，训练与推理使用一致的 unknown bucket。

### 9.4 报告指标

至少报告：

- ROC-AUC 和 PR-AUC；
- F1、Precision、Recall；
- validation 阈值下的 FPR；
- sequence-level 和 run-level 指标；
- 每个 workload、CVE、variant、slab bucket 的分组指标；
- leave-one-CVE-out 结果；
- bootstrap 置信区间或按 run 的方差；
- 推理延迟和吞吐量。

### 9.5 防止捷径学习的检查

训练前必须确认标签不能由以下单一字段轻易预测：

- trace 总时长；
- event 总数；
- 空窗口比例；
- run 文件大小；
- VM 是否 crash；
- CVE 专属内核或镜像；
- marker 是否存在；
- 采集脚本或输出路径。

## 10. 测试计划

### 10.1 单元测试

- ftrace/bpftrace 行解析；
- marker 缺失、重复、逆序和正常情况；
- ptr 重用、未知 FREE 和跨窗口 FREE size 恢复；
- 固定窗口及空窗口生成；
- 50% overlap 标签边界；
- endpoint/any 序列标签；
- sequence 不跨 run；
- workload/CVE 分层键解析；
- scaler 仅使用 train run；
- N-gram OOV 和平滑；
- 模型保存/加载后的分数一致性。

### 10.2 集成测试

- 使用两条正常 synthetic trace 和两条攻击 synthetic trace 跑通：
  - trace → CSV；
  - CSV → features；
  - run split；
  - baseline train；
  - threshold calibration；
  - evaluation report。
- 对相同输入重复执行，检查输出哈希和指标一致。
- 检查旧数据不会被新命令覆盖。

## 11. 里程碑与交付物

| 里程碑 | 主要交付物 | 完成判定 |
|---|---|---|
| M1：Legacy 冻结 | legacy manifest、目录规范 | 旧数据不再被覆盖 |
| M2：预处理 v2 | schema、窗口/标签/FREE 修复、测试 | synthetic 与 legacy marker 测试通过 |
| M3：采集器 v2 | 唯一 run、manifest、marker、强校验 | 连续采集无覆盖、无全局 kill |
| M4：Pilot-v2 | 两个 CVE 和匹配正常对照 | Pilot 验收清单全部通过 |
| M5：训练评估 v2 | run/CVE split、基线、阈值和报告 | 无泄漏端到端运行成功 |
| M6：Final-v2 | 正式数据与冻结测试结果 | 可复现最终报告 |

## 12. 建议执行顺序

```text
P0  冻结 legacy 并建立版本化目录
 ↓
P0  完成预处理、标签和 run 隔离测试
 ↓
P0  完成攻击采集器与 PoC 内 marker
 ↓
P1  采集并验收 pilot-v2
 ↓
P1  建立 Statistical/PCA/IF/MLP-AE 基线
 ↓
P1  修复并评估 LSTM-AE、LSTM-VAE、N-gram
 ↓
P2  扩大正式数据并执行 leave-one-CVE-out
 ↓
P2  冻结 final-v2 与最终测试报告
```

任何阶段发现 run 泄漏、marker 不可靠、采集捷径或数据不可追溯时，应停止后续模型实验，先修复数据管线并重新生成受影响的数据和产物。
