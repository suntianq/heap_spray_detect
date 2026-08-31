# 14 个新 CVE 的采集指南

本文档是"在 GPU 服务器上采集 KHeaps 其余 14 个 CVE"的操作指南。基于 2026-08-31 逐 CVE
源码分析得出。

## 背景：为什么需要给 PoC 加 marker

检测系统靠 trace_marker（`SPRAY_START`/`SPRAY_END`）标记堆喷时间窗口。采集时
`validate_trace`（`scripts/collect/collection_common.py:165`）要求恰好一对
`["SPRAY_START","SPRAY_END"]`。KHeaps 里只有 7308 的 PoC 写了 marker；其余 CVE 需要补。

**加 marker 的位置**：在堆喷前 `write_trace_marker("SPRAY_START")`，喷完
`write_trace_marker("SPRAY_END")`。参考 `CVE-2017-7308/poc/poc_cfh_single_spray.c:263,277`。

## 逐 CVE 分类（2026-08-31 源码分析）

### ✅ 单次执行（可直接加 marker，推荐）

这些 CVE 的 exploit 是"一次 spray 即结束"（无外层无限循环），在 spray 前后各加一行
marker 即可，采集会得到完整窗口。

| CVE | 执行流程 | spray 位置 | 备注 |
|-----|---------|-----------|------|
| CVE-2010-2959 | main → trigger() | trigger 内部 | ret2dir 技术 |
| CVE-2016-0728 | main → exploit() → trigger() | exploit 内部 | refcount 循环但非 spray |
| CVE-2016-8655 | main → try_exploit() | `add_key_spray_num` (174行) | 崩溃型，sleep 10 后崩 |
| CVE-2017-10661 | main → do_race() → sleep(5) | do_race 内部 | race 型 |
| CVE-2017-6074 | main → hijack() | `msg_spray` (363行) | double-free，udp_fifo |
| CVE-2017-7184 | main → trigger_oob() → hijack() | trigger_oob 内部 | 越界写 |
| CVE-2017-8824 | main → do_spray() | do_spray 内 add_key (188行) | 单次，简单 |
| CVE-2017-8890 | main → 线程 server/client → 等结束 | server 线程内 do_spray | 双线程，等 server_finish |
| CVE-2018-6555 | main → uaf() → trigger() | 内部 | 需确认 spray 在 uaf/trigger 哪个 |

### ❌ 循环执行（无法直接加 marker）

这些 CVE 的 exploit 在 while/for 循环里反复 spray，直接加 marker 会写多对 → 采集时
`validate_trace` 判 invalid。**需要改造 PoC**（见下方"循环 CVE 的处理"）。

| CVE | 循环结构 |
|-----|---------|
| CVE-2016-10150 | `for(int i=0; i<0x100000; i++) trigger()` |
| CVE-2016-4557 | `while(1) { clean_fork()? exploit() }` |
| CVE-2017-15649 | `for(int j=0; j<1337; j++) { race }` |
| CVE-2017-7533 | `while(!stop) { open; spray_pipe; check }` |

### ⛔ 无法采集

| CVE | 原因 |
|-----|------|
| CVE-2016-6187 | **无 `poc_cfh_single_spray.c`**（KHeaps 缺该变体） |

## 循环 CVE 的处理（3 种方法）

对 4 个循环 CVE（10150/4557/15649/7533），选择其一：

1. **限次 + marker**（推荐）：把 `while(1)` 改成有限次数（如 `for(i=0;i<10;i++)`），
   循环**外**包一对 marker（覆盖整个尝试过程）。这样只有一对 marker，采集通过，且
   窗口覆盖多次尝试。缺点：窗口可能不精确（含多个尝试）。
2. **单次化**：改造 PoC 只执行一次 spray（删循环），最精确但改动大。
3. **依赖崩溃兜底**：不改造，靠 `resolve_spray_window` 的崩溃推断（`expect_crash +
   vm_crashed` 用 last_ts 当 spray_end）。已有 11176/2636 这么跑，能采到部分 valid，
   但 marker 不完整、崩溃前可能漏采。

## 采集流程（每个 CVE）

```bash
# 1. 给 PoC 加 marker（参考 7308）
#    编辑 KHeaps/exploit_env/CVEs/<CVE>/poc/poc_cfh_single_spray.c（和 combo）
#    spray 前: write_trace_marker("SPRAY_START");
#    spray 后: write_trace_marker("SPRAY_END");

# 2. 注册 CVE 到 config.py 的 CVE_LIST（如已存在则跳过）

# 3. 采集（attack + baseline + normal 自动完成，可续跑）
CVE=<CVE> nohup scripts/collect/collect_cve_complete.sh \
    > datasets/.m6/logs/collect_<CVE>_complete.log 2>&1 &
```

## 批量采集（多台服务器并行）

14 个 CVE ≈ 40-56 小时/串行。多台服务器并行时，每台分几个 CVE：

```bash
# 服务器 A（举例）：单次执行类，先采
for CVE in CVE-2017-7184 CVE-2017-6074 CVE-2017-8824 CVE-2018-6555; do
  CVE=$CVE nohup scripts/collect/collect_cve_complete.sh > datasets/.m6/logs/collect_$CVE.log 2>&1 &
  # 注意: 同机并行会争 KVM/CPU，建议串行或用不同机器
done
```

## 采集后验证

```bash
# 重建 processed（纳入新 CVE）
python3 scripts/validate/build_pilot_dataset.py --raw datasets/raw --out datasets
# 检查 target slab 覆盖（G6 门禁）
# 若某 CVE 的 target slab 与预期不符（如 2636 曾计划 4096 实际 8192），
# 修正 common.py / pilot_gates.py 的 TARGET_SLAB
```

## 已知注意事项

- **marker 必须恰好一对**：多对（循环内）或 0 对（没加）都 invalid。
- **PoC 在 guest 内编译**：改完 `.c` 后由采集器自动上传编译，无需手动。
- **`--expect-crash` 必需**：崩溃型 CVE 不带它，崩溃 run 会被判无效。
- **CVE-2016-6187 无法采集**（无 single_spray PoC），跳过。
