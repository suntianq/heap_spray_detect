# 14 个新 CVE 的采集指南

本文档是"在 GPU 服务器上采集 KHeaps 其余 14 个 CVE"的操作指南。基于 2026-08-31 逐 CVE
源码分析 + 实测验证得出。

## 背景：marker 机制

检测系统靠 trace_marker（`SPRAY_START`/`SPRAY_END`）标记堆喷时间窗口。采集时
`validate_trace`（`scripts/collect/collection_common.py:165`）要求恰好一对
`["SPRAY_START","SPRAY_END"]`；`resolve_spray_window` 支持崩溃兜底（crash 时用 last_ts
当 spray_end，标记 partial）。

**关键发现（2026-08-31 实测）**：`KHeaps/exploit_env/libexp.c` 的所有主流 spray 函数
**内部自带 marker**：
- `msg_spray`（libexp.c:459/474）
- `add_key_spray_num`（libexp.c:384/393）
- `add_key_desc_spray_num`（libexp.c:410/417）

所以**只要 PoC 调用这些函数做 spray，marker 自动就有，无需手动加**。只有用自定义
spray 的 CVE 才需要手动加 marker。

## 逐 CVE 分类（2026-08-31 源码分析 + 实测）

### ✅ 无需修改（用 libexp 自带 marker 的 spray 函数）

这些 CVE 的 spray 走 libexp 的 `msg_spray`/`add_key_*`，marker 自动写入，直接采集即可。

| CVE | spray 函数 | 说明 |
|-----|-----------|------|
| CVE-2016-0728 | `msg_spray_max` | exploit 末尾 spray |
| CVE-2016-8655 | `add_key_spray_num` | 崩溃型 |
| CVE-2017-6074 | `msg_spray` | double-free |
| CVE-2017-8824 | `add_key_desc_spray_num` | 已实测：恰好一对 marker ✓ |
| CVE-2018-6555 | `add_key_desc_spray_num` | uaf 内 spray |

### ✅ 已手动加 marker（2026-08-31 已改，用自定义 spray）

这 3 个 CVE 的 spray 不走 libexp 自带 marker 函数，我已在 PoC 里加了
`write_trace_marker("SPRAY_START"/"SPRAY_END")`，已提交。

| CVE | spray 位置 | 说明 |
|-----|-----------|------|
| CVE-2010-2959 | `send()` 覆盖相邻对象（ret2dir） | single:197/207, combo 同理 |
| CVE-2017-7184 | `alloc_victim()` open /proc/buddyinfo | single:417/420 |
| CVE-2017-8890 | 自定义 `do_spray()`（server 线程内） | single:107/109；线程竞争，partial 有效 |

### ❌ 循环执行（需改造，见下方）

| CVE | 循环结构 |
|-----|---------|
| CVE-2016-10150 | `for(int i=0; i<0x100000; i++) trigger()` |
| CVE-2016-4557 | `while(1) { clean_fork()? exploit() }` |
| CVE-2017-10661 | `for(int i=0; i<RACE_TIME=8000; i++) { msg_spray }` |
| CVE-2017-15649 | `for(int j=0; j<1337; j++) { race }` |
| CVE-2017-7533 | `while(!stop) { open; spray_pipe; check }` |

### ⛔ 无法采集

| CVE | 原因 |
|-----|------|
| CVE-2016-6187 | **无 `poc_cfh_single_spray.c`**（KHeaps 缺该变体） |

## 循环 CVE 的处理（3 种方法）

对 5 个循环 CVE（10150/4557/10661/15649/7533），选择其一：

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

14 个 CVE ≈ 40-56 小时/串行。多台服务器并行时，每台分几个 CVE。

**先采 8 个无需改造的 CVE**（5 个 libexp 自带 marker + 3 个已加 marker）：

```bash
# 服务器 A（举例）
for CVE in CVE-2017-6074 CVE-2017-8824 CVE-2018-6555 CVE-2016-0728 CVE-2016-8655 \
           CVE-2010-2959 CVE-2017-7184 CVE-2017-8890; do
  CVE=$CVE nohup scripts/collect/collect_cve_complete.sh > datasets/.m6/logs/collect_$CVE.log 2>&1 &
  # 注意: 同机并行会争 KVM/CPU，建议串行或用不同机器
done
```

**5 个循环 CVE**（10150/4557/10661/15649/7533）需先按上文改造 PoC 再采。

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
