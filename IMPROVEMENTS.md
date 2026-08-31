# heap_spray 改进操作指南（P0/P1/P2）

> 里程碑 M1–M6 已冻结（ocsvm 主检测 / mlp_ae 高吞吐替代；416 有效 run、门禁全 PASS）。
> 本指南给出下一阶段的改进方向 **P0（方法学夯实）+ P1（数据与泛化）+ P2（检测质量）** 的
> 详细操作步骤：每条方向都标注动机、前置条件、具体命令、改动文件、预期产出与检查点、回滚方式。
>
> 环境约定与 MIGRATION.md 一致：**真机**（conda，numpy 已装）跑采集/构建/报告；**PyTorch Docker**
> 跑训练。命令里 `python` 在真机上指 conda 激活后的 python，训练命令注明"容器内"。

---

## 0. 总览与推荐顺序

| 编号 | 方向 | 优先级 | 成本 | 需重采集 | 产出 |
|---|---|---|---|---|---|
| D1 | 补跑 4 个现成模型（isolation_forest/lof/pca/stat_threshold） | P0 | 低（Docker 训练） | 否 | 模型对比表补全 |
| D2 | 多 seed 评估（5 模型 × 4+ seed） | P0 | 低-中（训练时间） | 否 | run AUC/F1 区间 |
| D3 | 冻结 requirements-train.txt | P0 | 极低 | 否 | Docker 依赖锁定 |
| D7 | 特征增强（亚窗口峰值/窗口间 delta/任务类别） | P2 | 中 | **否**（重建即可） | 特征维度↑、检测质量↑ |
| D8 | 阈值与误报控制（EVT/pot、per-workload） | P2 | 低-中 | 否 | ocsvm FPR 0.197↓ |
| D5 | 扩 CVE 到 4–6 个（跨 CVE 泛化 + LOO） | P1 | 高（采集数小时/CVE） | 是 | LOO 泛化证据 |
| D6 | normal workload 覆盖扩展 | P1 | 中 | 是 | normal 更广、FPR↓ |
| D4 | 扩展 tracepoint（kmem_cache_alloc/free，解 filp_cachep 盲区） | P1 | 高（schema v3 重采） | 是 | 7308/single 可观测 |

**推荐执行顺序**（性价比优先）：

1. **D1 → D2 → D3**（P0，几天内出结果，把"ocsvm 是否真最优"和置信度钉死）
2. **D7**（特征增强，不需重采集，从现有 CSV 重建即可，与 P0 正交）
3. **D8**（阈值控制，建立在 D2 的多 seed 基线上）
4. **D5 → D6**（扩 CVE、扩 normal；采集时间长，宜尽早启动）
5. **D4 最后**（需新 trace 事件 + schema v3 + 全量重采，收益最大但成本最高，等前面结论确认模型方向后再投入）

依赖关系：D8 依赖 D2 的多 seed 基线；D4/D5/D6 都改变数据集 → 之后需重训全部模型。

---

## 1. 前置

- 真机：conda 环境 `heap_spray` 就绪（MIGRATION.md §1–§3）。校验：`python -c "import numpy"`。
- 训练：PyTorch Docker，项目挂载到 `/heap_spray`（MIGRATION.md §6.1）。
- 现有冻结数据 `datasets/final-v2/` 与 `runs/` 完整（若动了数据，先备份：
  `cp -r datasets/final-v2 datasets/final-v2.bak-<日期>`；`runs` 同理）。

---

## 2. P0 方法学夯实

### D1 补跑 4 个现成模型

**动机**：`run_experiment.py:45-58` 的 `MODEL_FACTORY` 有 8 个模型，M6 只对比了 5 个；
`isolation_forest/lof/pca/stat_threshold` 零新代码即可补跑，验证"ocsvm 最优"结论是否稳健。

**步骤**（容器内）：
```bash
cd /heap_spray   # Docker 工作目录（挂载项目根）
for m in isolation_forest lof pca stat_threshold; do
  python scripts/train/run_experiment.py --model $m \
      --attack-data datasets/final-v2/processed/attack \
      --normal-data datasets/final-v2/processed/normal \
      --dataset-manifest datasets/final-v2/dataset_manifest.json \
      --out runs
done
```

**产出/检查点**：
- `runs/` 新增 4 个实验目录（`..._<model>_s42_...`），各含 `evaluation_report.json`/`gates.json`/`model.pkl`。
- 真机重出报告（自动把模型行从 5 扩到 9，保留既有 `## 结论` 段）：
```bash
python scripts/validate/final_v2_report.py --runs runs --dataset datasets/final-v2 \
    --out runs/ACCEPTANCE_M6.md
```
- 打开报告确认：9 行模型表、G10 列状态、`## 结论` 段仍在。

**注意/回滚**：这 4 个是补充对照，不改变冻结结论。如需标注，在 `## 结论` 追加一行
"补充模型对比（isolation_forest/lof/pca/stat_threshold）见上表"。回滚：删除对应实验目录后重出报告即可。

---

### D2 多 seed 评估

**动机**：当前全部结论基于 seed=42 单次划分（`run_experiment.py:146` 默认 42），无跨划分方差；
`common.py:203-225` 已有 bootstrap CI，但没有多 seed。多 seed 才能回答"ocsvm 的优势是否显著"。

**步骤**（容器内，可并行开多个训练进程）：
```bash
cd /heap_spray
SEEDS="7 13 21 2024"
for seed in $SEEDS; do
  for m in ocsvm mlp_ae lstm_ae lstm_vae; do
    python scripts/train/run_experiment.py --model $m --seed $seed \
        --attack-data datasets/final-v2/processed/attack \
        --normal-data datasets/final-v2/processed/normal \
        --dataset-manifest datasets/final-v2/dataset_manifest.json \
        --out runs
  done
  # ngram 走独立 runner
  python scripts/train/run_ngram.py --seed $seed \
      --attack-data datasets/final-v2/processed/attack \
      --normal-data datasets/final-v2/processed/normal \
      --normal-csv-root datasets/final-v2/csv/normal \
      --attack-csv-root datasets/final-v2/csv/attack \
      --dataset-manifest datasets/final-v2/dataset_manifest.json \
      --out runs
done
```

**重要**：`final_v2_report.py` 对每个模型只取 sorted 目录的最后一次 base（`load_experiments` 覆盖式），
**多 seed 不能靠报告表**。用下面的聚合脚本（真机，只需标准库）逐模型输出 run AUC/F1 的
min/max/中位，并附单 seed 的 bootstrap CI：

```python
# scripts/validate/seed_summary.py —— 复制此内容保存后运行（真机，无第三方依赖）
"""多 seed 聚合：按模型汇总 runs/*/evaluation_report.json 的 run 级 AUC/F1。"""
import glob, json, re
from collections import defaultdict

rows = defaultdict(list)
for p in glob.glob("runs/*/evaluation_report.json"):
    r = json.load(open(p))
    model = r.get("model")
    if model == "ngram":            # ngram 报告字段见其 runner 输出，必要时单独处理
        continue
    mid = r.get("experiment_id", "")
    m = re.search(r"_s(\d+)_", mid)
    seed = int(m.group(1)) if m else -1
    rl = r.get("run_level", {})
    rows[model].append({
        "seed": seed,
        "auc": rl.get("roc_auc"),
        "f1": rl.get("f1_at_threshold"),
        "ci95": (r.get("run_bootstrap_ci95") or {}).get("roc_auc_ci95"),
    })

for model, lst in sorted(rows.items()):
    aucs = sorted(x["auc"] for x in lst if x["auc"] is not None)
    f1s = sorted(x["f1"] for x in lst if x["f1"] is not None)
    if not aucs:
        print(f"{model}: (无 run_level 数据)"); continue
    print(f"{model}: n={len(lst)}  run AUC min={aucs[0]:.3f} med={aucs[len(aucs)//2]:.3f} max={aucs[-1]:.3f}"
          f" | F1 min={f1s[0]:.3f} med={f1s[len(f1s)//2]:.3f} max={f1s[-1]:.3f}")
```

**产出/检查点**：每模型 ≥4 个 seed 的 run AUC/F1 汇总表。判读：若 ocsvm 的 AUC 区间
（min–max）与第二名（通常是 mlp_ae）不重叠 → "ocsvm 最优"结论成立；重叠 → 在报告
`## 结论` 里弱化为"与 mlp_ae 相当，选 ocsvm 因其召回更高"。回滚：删除新增实验目录。

---

### D3 冻结训练依赖

**动机**：真机 `requirements.txt` 只锁了 numpy（仿真模拟侧）；Docker 训练侧尚无锁定版本，
新机器训练结果不可精确复现。

**步骤**（在装有训练库的旧 .venv / 容器里）：
```bash
# 查看 6 个顶层训练库的当前版本
.venv/bin/pip freeze | grep -iE "^(torch|scikit-learn|scipy|numpy|matplotlib|pandas)="
```
把输出写进新文件 `requirements-train.txt`（保留 6 行的 `包名==版本` 即可，不加传递依赖；
若追求完全复现可全量 freeze）。随后替换 `MIGRATION.md` §10 的清单，并让 Docker 训练
`pip install -r requirements-train.txt`。

**产出**：`requirements-train.txt`。**检查点**：容器内 `pip install -r requirements-train.txt` 成功。

---

## 3. P1 数据与泛化

### D4 扩展 tracepoint（schema v3，解 filp_cachep 盲区）

**动机**：`trace_start.sh:18-19` 只开 `kmalloc`/`kfree` 两个事件。CVE-2017-7308/single 的目标对象
`struct file` 由独立 SLUB 缓存 `filp_cachep` 分配，kmalloc 事件里完全不可见——该变体 21 采只 3 崩
（crash 14.3%）、ngram 7308=0.580，是全部模型最弱项。启用 `kmem_cache_*` 事件后独立缓存进入
特征空间，这是最难样本的直接解法。

**第 0 步：在 guest 里核验事件格式**（先手动起 guest，见 MIGRATION.md §4.0）：
```bash
cat /sys/kernel/debug/tracing/events/kmem/kmem_cache_free/format
cat /sys/kernel/debug/tracing/events/kmem/kmem_cache_alloc/format
# 重点看 TP_printk 字段：kmem_cache_free 有 name=；kmem_cache_alloc 通常没有 cache 名
```
> **关键事实**（内核 4.15 主线）：`kmem_cache_free` 事件打印 `call_site=... ptr=... name=...`
> （含 cache 名）；`kmem_cache_alloc` 事件只打印 `call_site/ptr/bytes_req/bytes_alloc/gfp_flags`，
> **不含 cache 名**。因此"知道每次分配进了哪个 cache"需要额外手段，见路线 B。

**路线 A（最小，推荐先做）—— 只加 kmem_cache_free（含 cache 名）**：
1. `scripts/collect/trace_helpers/trace_start.sh` 在 L19 后追加：
   ```sh
   test -e "$TRACE_ROOT/events/kmem/kmem_cache_free/enable" \
       && echo 1 > "$TRACE_ROOT/events/kmem/kmem_cache_free/enable"
   test -e "$TRACE_ROOT/events/kmem/kmem_cache_alloc/enable" \
       && echo 1 > "$TRACE_ROOT/events/kmem/kmem_cache_alloc/enable"
   ```
2. `trace_stop.sh` 在 L23 后对称关闭两个事件（`echo 0 > .../enable`）。
3. `scripts/preprocess/trace2csv.py`：在 `KFREE_RE`（L22-27）旁新增 `KMEM_CACHE_FREE_RE` 与
   `KMEM_CACHE_ALLOC_RE`，解析 `name=`；CSV 增一列 `cache_name`（L125-126 的列头随之扩展）。
4. `scripts/preprocess/csv2features.py`：把带 cache 名的释放事件按 cache 名聚合成
   `free_count_<cache>` 新特征列（cache 集合从数据中固定：filp、skbuff_head_cache、key_jar 等）。

**路线 B（完整，可选）—— 分配也带 cache 名**：
- 方案：kmem_cache_alloc 无 cache 名，但可用"ptr→cache 字典"补齐——把整条 trace 里
  `kmem_cache_free` 的 `(ptr, name)` 建成字典，反查 `kmem_cache_alloc` 的 `ptr` 归属
  （对象存活期内未被释放的无法反查，落回大小桶或 `unknown_cache`）。仓库已有类似基建
  （`csv2features.py` 的 ptr→size 释放解析）。
- 若要 100% 覆盖（含从未释放对象），需 ftrace **kprobe** 读 `cachep->name`：对 4.15 内核
  `struct kmem_cache` 的 `name` 字段偏移，用 pahole 在 vmlinux 上查，注册形如
  `p:kalloc kmem_cache_alloc cachep=%di name=+<off>(%di):string` 的事件。成本明显更高，非必须。

**下游改动**（无论哪条路线）：
- `config.py`：`FEAT_DIM`、`SIZE_BUCKET_LABELS` 扩展；`DATASET_SCHEMA_VERSION` 升到 3。
- `dataset_manifest.json` 的 `schema_version` 随之变化；`runs/` 里旧实验与新 schema 不混用。
- 需**重新采集**（M6 规模或先 `-n 5` 冒烟）→ `build_pilot_dataset.py` 重建 → 重训 5 模型。

**产出/检查点**：
- guest 内 `cat .../kmem_cache_free/enable` 为 `1`。
- 重建后 `features_columns.csv` 出现 `*_filp` / `*_skbuff_head_cache` 类特征。
- 重训后 7308/single 的逐 CVE AUC 显著高于当前（ngram 0.580）——这是本方向的成败指标。

**回滚**：trace 脚本不 enable 即回到 v2；用 `dataset_manifest.json` 的 `schema_version` 字段区分新旧数据。

---

### D5 扩 CVE 到 4–6 个（跨 CVE 泛化 + LOO）

**动机**：KHeaps 已有 18 个 CVE、17 个带 single_spray+combo 双变体（目标 slab 覆盖 64→4096，
bzImage 均已预构建），项目只用了 2 个；M6 明确跳过 LOO，在 2 个 CVE 上做 LOO 没有统计意义。
加几个目标 slab 差异大的 CVE，才能真实验证"模型对未见 CVE 的泛化"。

**候选 CVE**（目标 slab 分散，避开已用的 256/2048）：
- `CVE-2017-2636`（kmalloc-4096，double-free）
- `CVE-2017-7184`（kmalloc-128，堆越界写）
- `CVE-2017-7533`（kmalloc-64，堆越界写）
- `CVE-2016-0728`（kmalloc-192，引用计数溢出→UAF）

**步骤**：
1. `config.py:27-30` 的 `CVE_LIST` 加入目标 CVE（如 `"CVE-2017-2636"`）。
2. 采集该 CVE 的 attack（真机，命令同 M6）：
   ```bash
   python scripts/collect/collect_attack_stable.py -c CVE-2017-2636 \
       -v poc_cfh_single_spray poc_cfh_combo -n 15 \
       --expect-crash CVE-2017-2636 --poc-timeout 90 -o datasets/final-v2/raw/attack
   python scripts/collect/collect_attack_stable.py -c CVE-2017-2636 \
       -v poc_cfh_baseline -n 15 --expect-crash CVE-2017-2636 --poc-timeout 90 \
       -o datasets/final-v2/raw/baseline
   ```
3. 采集该 CVE 的 normal（8 类×20，命令同 MIGRATION.md §4.1，把 `-c` 换成新 CVE）。
4. 重建数据集（真机）：`python scripts/validate/build_pilot_dataset.py --attack-raw ... --normal-raw ... --baseline-raw ... --out datasets/final-v2`
   （三条 raw 路径参数同 MIGRATION.md §5，日志 tee 到 `.m6/logs/build_gates.log`）。
5. 重训全部模型（Docker，同 MIGRATION.md §6.2，命令不变——数据换了）。
6. **跑 LOO**（容器内）：
   ```bash
   for m in ocsvm mlp_ae lstm_ae lstm_vae; do
     python scripts/train/run_experiment.py --model $m --held-out-cve CVE-2017-2636 \
         --attack-data datasets/final-v2/processed/attack \
         --normal-data datasets/final-v2/processed/normal \
         --dataset-manifest datasets/final-v2/dataset_manifest.json --out runs
   done
   # ngram 同样支持 --held-out-cve（run_ngram.py）
   ```
7. 重出报告：`final_v2_report.py ...`（`has_loo` 分支会自动显示"Leave-one-CVE-out 泛化"表）。

**产出/检查点**：新增 CVE 各变体 valid≥15（`grep -rl '"status": "valid"' datasets/final-v2/raw/attack/<CVE>/<variant> | wc -l`）；
报告出现 LOO 表；**ocsvm 在留出 CVE 上 run AUC ≥0.9 则"跨 CVE 泛化"主张成立，否则弱化结论**。

**成本提示**：每 CVE 的 attack 约 1h、normal 约 2.5h（串行）；采集可与 D6 一起并行启动。
**回滚**：从 `CVE_LIST` 移除 + 删除该 CVE 的 raw 目录 → 重建 → 重训（保留其他 CVE 数据即可）。

---

### D6 normal workload 覆盖扩展

**动机**：当前 8 类 normal（idle/msg_msg_256/msg_msg_2048/keyctl/net_busy/fs_io/fork_stress/mem_pressure）
覆盖进程/线程、文件 IO、key、SysV 消息、UDP+AF_UNIX、内存压力；缺 TCP、syscall 风暴、多进程并发、
epoll/timerfd 异步等待。ocsvm run FPR 0.197 部分源于真实负载差异——normal 更广，阈值才更可信。

**限制**：guest 内核是 **4.15.0**（`KHeaps/scripts/kernel_builder/bk_config`），**io_uring 是 5.1 才引入，
不能在 4.15 上测**。可选负载（4.15 支持）：
- TCP loopback 流量（net_busy 现在只有 UDP+AF_UNIX）
- syscall 风暴（高频 open/close/gettid 等无对象分配 syscall）
- 多进程并发混合负载（多个不同 workload 同时跑）
- epoll/timerfd/eventfd 异步等待类

**步骤**：
1. 新建 `scripts/collect/workloads/workload_<x>.c`（参考现有 6 个，如 `workload_fs.c`）。
2. 在 `scripts/collect/collect_stable.py` 登记：
   - 常量区（L27-32 旁）加 `WORKLOAD_<X>_SOURCE = Path(__file__).parent / "workloads" / "workload_<x>.c"`；
   - `build_workloads_in_vm`（L42-84）加 `if "<x>" in workloads:` 分支（上传 + `gcc -O2 -o workload_<x> workload_<x>.c`）；
   - 确认 run 执行端按 workload 名调用 `./workload_<x>`（查看 `run_one` 的 guest 命令）。
3. 采集（真机）：`collect_stable.py -c <CVE> -n 20 -d 30 -w <x> --msg-sizes 256 2048 -o datasets/final-v2/raw/normal`
   （两个 CVE 内核各来一遍）。
4. 重建 → 重训（同 D5 第 4-5 步）。

**产出/检查点**：新 workload 的 CSV/事件量正常（对比 `normal_workloads.csv` 现有量级）；
重训后 ocsvm run FPR 相对 0.197 是否下降（用 D2 的多 seed 基线对比）。**回滚**：删 workload 源文件 +
从 `NORMAL_WORKLOADS`/登记处移除 + 重建重训。

---

## 4. P2 检测质量

### D7 特征增强（不需重采集）

**动机**：90 维特征本质是窗口统计量（计数/均值率/熵/比值）。短喷（11176 中位 0.47ms）在
100ms 窗口被摊平：全局只有 `alloc_burst_1ms`（`csv2features.py:116`），**无 per-bucket 亚窗口峰值、
无窗口间 delta、任务名（top_comm）只进 npz 元数据不进特征**。以下增强可从现有 CSV 重建，
**不需要重新采集**。

**改动 `scripts/preprocess/csv2features.py`**：
1. **per-bucket 亚窗口峰值**：把现有全局 `alloc_burst_1ms` 的思路下沉到 bucket 级，加
   `alloc_burst_1ms_<bucket>`（每 bucket 的 1ms 最大 bin 计数）；10ms 版可选。注意特征量（12 个新增）。
2. **窗口间 delta**：`count[t]-count[t-1]`（每 bucket alloc_count 的一阶差分，首窗口为 0 或
   `time_since_previous_event_ms` 已有类似衔接）。序列模型已隐含时序，新增 delta 主要惠及 ocsvm/
   统计模型。
3. **任务类别 one-hot**：`top_comm` 已在 `csv2features.py:333,376` 提取为 npz 元数据，转成固定类别
   的 one-hot 特征。
   > **⚠️ G9_no_shortcut 风险**（`pilot_gates.py:245-276` 只查 duration/empty_ratio/文件长度）：
   > 任务名特征若直接编码"进程身份"，模型会退化成"看到 PoC 进程名就报警"的捷径，绕过喷雾检测，
   > 且 G9 当前检查不到。必须把 top_comm 映射到**活动类别**（如 `poc/systemd/sshd/kworker/其他`，
   > 且 baseline 负样本归入同一类别体系），并在 D7 完成后显式验证门禁仍全 PASS。
4. 同步 `feature_names()`/`feature_groups()`（`csv2features.py:104-167`）与 `config.FEAT_DIM`。

**步骤**：
```bash
# 真机：从现有 CSV 重建 processed（原始 trace 不动；CSV 在 datasets/final-v2/csv/ 仍在）
mkdir -p datasets/final-v2/.m6/logs
python scripts/validate/build_pilot_dataset.py \
    --attack-raw  datasets/final-v2/raw/attack \
    --normal-raw  datasets/final-v2/raw/normal \
    --baseline-raw datasets/final-v2/raw/baseline \
    --out datasets/final-v2 \
    2>&1 | tee datasets/final-v2/.m6/logs/build_gates.log
# 容器内重训 5 模型（D2 命令），重出报告
```

**产出/检查点**：`features_columns.csv` 维度比 90 大；重建后 G1–G6/G9 全 PASS（尤其 G9 未触发）；
重训后与 D2 多 seed 基线对比 run AUC/F1/FPR。**回滚**：`git` 无，手动还原 `csv2features.py`/
`config.py` 后重建重训；或保留改动前的 `processed/` 副本。

---

### D8 阈值与误报控制

**动机**：冻结报告里 ocsvm **run FPR 0.197**（阈值=验证集 normal p99，`common.py:129-133`），
recall 1.0 但误报偏高。阈值策略是可控的杠杆。

**方向一：EVT/pot 尾部阈值**。对验证集 normal 分数的高分尾部（如 >90 分位）拟合广义 Pareto 分布，
按"超过阈值概率=α"反解阈值，替代固定 p99 —— 对长尾分布比分位数更稳，能在相近 recall 下压 FPR。

**方向二：per-workload 条件化阈值**。各 workload 的正常基线不同（keyctl 事件量是 idle 的 30+ 倍），
全局一个阈值让高活动 workload 的误报集中。按 workload 分组各自标定 p99，再按 run 的 workload 用对应阈值。

**改动**：
- `scripts/train/common.py`：`threshold_at_fpr`（L129-133）加 `method="percentile|pot"` 与
  `grouped=False` 参数；pot 实现放 `common.py`（scipy.stats.genpareto 在训练环境可用）。
- `scripts/train/run_experiment.py`：`--target-fpr` 已存在（L149）；加 `--threshold-method`、
  `--per-workload-threshold` 开关，评估时同时输出多个 target_fpr 的 FPR/recall 折线。

**验证**（重训后）：
```bash
# 容器内：同一模型在不同阈值策略下的 FPR/recall
python scripts/train/run_experiment.py --model ocsvm --threshold-method pot --target-fpr 0.01 ...
python scripts/train/run_experiment.py --model ocsvm --threshold-method percentile --per-workload-threshold ...
```

**产出/检查点**：同等 run recall 下 ocsvm FPR 较 0.197 下降；报告中 `threshold` 行可溯源到策略参数。
**回滚**：`common.py`/`run_experiment.py` 改动还原，删除新实验目录。

---

## 5. 校验清单与回滚

| 方向 | 检查点（通过即完成） |
|---|---|
| D1 | `runs/` 多 4 个实验目录；报告表 9 行；结论段保留 |
| D2 | 每模型 ≥4 seed 的 run AUC/F1 min-med-max 表；区间判读写进结论 |
| D3 | `requirements-train.txt` 存在；容器 `pip install -r` 成功 |
| D7 | 特征维度>90；重建后门禁全 PASS；重训指标对比表 |
| D8 | 新阈值策略下 FPR 对比表（同 recall） |
| D5 | 新 CVE 各变体 valid≥15；报告 LOO 表出现；ocsvm 留出 CVE AUC |
| D6 | 新 workload CSV/事件量正常；重训后 FPR 对比 |
| D4 | guest 事件 enable=1；`features_columns.csv` 出现 `*_filp`；7308/single AUC↑ |

**通用回滚**：
- 数据/重建类：先 `cp -r datasets/final-v2 datasets/final-v2.bak-<日期>`；出问题恢复副本。
- 训练类：新增实验目录直接删除；`runs/ACCEPTANCE_M6.md` 可用 `final_v2_report.py` 重新生成。
- 代码类：改动集中在 `trace_start.sh`/`trace_stop.sh`/`trace2csv.py`/`csv2features.py`/`config.py`/
  `common.py`/`run_experiment.py`/`collect_stable.py`——逐文件还原即可（无 git，改动前自行 `cp` 备份）。
