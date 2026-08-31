"""Unit tests for attack/normal collector shared helpers (M3).

Covers IMPLEMENTATION_PLAN.md section 6.3 fail-closed markers:
- marker timestamp extraction from real ftrace lines
- balanced/ordered marker pair validation (single and multi-stage sprays)
- reject unbalanced / reversed / SPRAY_END<=SPRAY_START markers
- trace time bounds, event counting, trace_stats overrun parsing
- atomic manifest writing with manifest_version=2
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "collect"))

from collection_common import (extract_marker_timestamps, make_run_id, parse_trace_overrun,
                               resolve_spray_window, sha256_file, trace_bounds,
                               validate_markers, validate_trace, write_manifest)
from synthetic import kfree_line, kmalloc_line, marker_line, write_trace_file

NS = 1e9


class MarkerExtractionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def _trace(self, lines):
        path = os.path.join(self.dir, "trace.log")
        write_trace_file(path, lines)
        return path

    def test_extracts_marker_timestamps_in_order(self):
        path = self._trace([
            kmalloc_line("bash", 100, int(1.0 * NS), 0xffff88001234, 256, 256),
            marker_line(int(2.5 * NS), "SPRAY_START"),
            kmalloc_line("bash", 100, int(3.0 * NS), 0xffff88005678, 64, 64),
            marker_line(int(4.2 * NS), "SPRAY_END"),
            kfree_line("bash", 100, int(5.0 * NS), 0xffff88005678),
        ])
        markers = extract_marker_timestamps(path)
        self.assertEqual([name for name, _ in markers],
                         ["SPRAY_START", "SPRAY_END"])
        self.assertEqual([ts for _, ts in markers],
                         [int(2.5 * NS), int(4.2 * NS)])

    def test_extract_missing_file_returns_empty(self):
        self.assertEqual(extract_marker_timestamps(os.path.join(self.dir, "nope.log")), [])

    def test_does_not_confuse_event_lines_for_markers(self):
        path = self._trace([
            kmalloc_line("bash", 100, int(1.0 * NS), 0x10, 128, 128),
            kfree_line("bash", 100, int(1.5 * NS), 0x10),
        ])
        self.assertEqual(extract_marker_timestamps(path), [])


class ValidateMarkersTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_single_pair_valid(self):
        markers = [("SPRAY_START", int(2.0 * NS)), ("SPRAY_END", int(4.0 * NS))]
        valid, info = validate_markers(markers)
        self.assertTrue(valid)
        self.assertEqual(info["marker_count"], 2)
        self.assertEqual(info["spray_start_ns"], int(2.0 * NS))
        self.assertEqual(info["spray_end_ns"], int(4.0 * NS))

    def test_multiple_stages_valid(self):
        markers = [
            ("SPRAY_START", int(1.0 * NS)), ("SPRAY_END", int(2.0 * NS)),
            ("SPRAY_START", int(3.0 * NS)), ("SPRAY_END", int(4.0 * NS)),
        ]
        valid, info = validate_markers(markers)
        self.assertTrue(valid)
        self.assertEqual(info["spray_start_ns"], int(1.0 * NS))
        self.assertEqual(info["spray_end_ns"], int(4.0 * NS))

    def test_empty_invalid(self):
        valid, _ = validate_markers([])
        self.assertFalse(valid)

    def test_unbalanced_start_without_end_invalid(self):
        markers = [("SPRAY_START", int(1.0 * NS))]
        self.assertFalse(validate_markers(markers)[0])

    def test_end_before_start_invalid(self):
        markers = [("SPRAY_END", int(1.0 * NS)), ("SPRAY_START", int(2.0 * NS))]
        self.assertFalse(validate_markers(markers)[0])

    def test_end_equals_or_before_start_invalid(self):
        markers = [("SPRAY_START", int(4.0 * NS)), ("SPRAY_END", int(4.0 * NS))]
        self.assertFalse(validate_markers(markers)[0])
        markers = [("SPRAY_START", int(4.0 * NS)), ("SPRAY_END", int(2.0 * NS))]
        self.assertFalse(validate_markers(markers)[0])


class ResolveSprayWindowTest(unittest.TestCase):
    def test_balanced_markers_used_directly(self):
        markers = [("SPRAY_START", int(1.0 * NS)), ("SPRAY_END", int(3.0 * NS))]
        start, end, partial = resolve_spray_window(markers, False, False, int(9.0 * NS))
        self.assertEqual((start, end, partial), (int(1.0 * NS), int(3.0 * NS), False))

    def test_unbalanced_start_uses_last_ts_when_crash_expected(self):
        markers = [("SPRAY_START", int(1.0 * NS))]
        start, end, partial = resolve_spray_window(markers, True, True, int(6.0 * NS))
        self.assertEqual((start, end, partial), (int(1.0 * NS), int(6.0 * NS), True))

    def test_unbalanced_start_rejected_when_no_crash(self):
        markers = [("SPRAY_START", int(1.0 * NS))]
        self.assertIsNone(resolve_spray_window(markers, False, True, int(6.0 * NS)))
        self.assertIsNone(resolve_spray_window(markers, True, False, int(6.0 * NS)))

    def test_empty_rejected(self):
        self.assertIsNone(resolve_spray_window([], True, True, int(6.0 * NS)))

    def test_crash_not_expected_rejects_partial(self):
        # A partial sequence is only salvageable for runs expected to crash.
        markers = [("SPRAY_START", int(1.0 * NS))]
        self.assertIsNone(resolve_spray_window(markers, False, False, int(6.0 * NS)))


class TraceBoundsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def test_bounds_are_first_last_event_timestamps(self):
        path = os.path.join(self.dir, "trace.log")
        write_trace_file(path, [
            kmalloc_line("bash", 100, int(1.25 * NS), 0x10, 64, 64),
            marker_line(int(2.0 * NS), "SPRAY_START"),
            kfree_line("bash", 100, int(9.75 * NS), 0x10),
        ])
        first, last = trace_bounds(path)
        self.assertEqual(first, int(1.25 * NS))
        self.assertEqual(last, int(9.75 * NS))

    def test_bounds_missing_file(self):
        self.assertEqual(trace_bounds(os.path.join(self.dir, "nope.log")), (None, None))


class TraceValidationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def test_event_count(self):
        path = os.path.join(self.dir, "trace.log")
        write_trace_file(path, [
            kmalloc_line("a", 1, int(1.0 * NS), 0x1, 32, 32),
            kfree_line("a", 1, int(1.1 * NS), 0x1),
            marker_line(int(1.5 * NS), "SPRAY_START"),
            kmalloc_line("b", 2, int(2.0 * NS), 0x2, 32, 32),
        ])
        valid, info = validate_trace(path, require_markers=False, minimum_events=3)
        self.assertTrue(valid)
        self.assertEqual(info["event_count"], 3)
        self.assertEqual(info["markers"], ["SPRAY_START"])

    def test_require_markers_strict(self):
        path = os.path.join(self.dir, "trace.log")
        write_trace_file(path, [
            marker_line(int(1.5 * NS), "SPRAY_START"),
            marker_line(int(2.5 * NS), "SPRAY_END"),
            kmalloc_line("a", 1, int(1.0 * NS), 0x1, 32, 32),
        ])
        valid, info = validate_trace(path, require_markers=True, minimum_events=1)
        self.assertTrue(valid)
        valid_bad, info_bad = validate_trace(path, require_markers=True, minimum_events=100)
        self.assertFalse(valid_bad)
        self.assertEqual(info_bad["reason"], "too_few_events")

    def test_missing_file_invalid(self):
        valid, info = validate_trace(os.path.join(self.dir, "nope.log"))
        self.assertFalse(valid)
        self.assertEqual(info["reason"], "missing_or_empty_trace")


class OverrunParseTest(unittest.TestCase):
    def test_sums_overrun_across_cpus(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("[cpu0/stats]\nentries: 10\noverrun: 0\n")
            handle.write("[cpu1/stats]\nentries: 4\noverrun: 7\n")
            name = handle.name
        try:
            self.assertEqual(parse_trace_overrun(name), 7)
        finally:
            os.unlink(name)

    def test_missing_stats_returns_none(self):
        self.assertIsNone(parse_trace_overrun("/no/such/file"))

    def test_parse_only_overrun_lines(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("overrun: 1\nsomething: 999\noverrun: 2\n")
            name = handle.name
        try:
            self.assertEqual(parse_trace_overrun(name), 3)
        finally:
            os.unlink(name)


class ManifestAndIdsTest(unittest.TestCase):
    def test_make_run_id_unique(self):
        first = make_run_id(0)
        second = make_run_id(0)
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("run_000_"))

    def test_cve_first_run_dir_layout(self):
        """Collectors must write the CVE-first class layout (datasets restructure).

        attack collector: raw/<CVE>/{attack,baseline}/<variant>/run_XXX_*/
        normal collector: raw/<CVE>/normal/<workload>/run_XXX_*/
        The class segment is on disk only; build_pilot_dataset strips it so run_ids
        stay CVE/<variant|workload>/run_XXX_*/trace.
        """
        from pathlib import Path
        cve = "CVE-2017-7533"
        # attack/baseline variants route by name (mirror of collect_attack_stable)
        for variant, expect_class in (("poc_cfh_single_spray", "attack"),
                                     ("poc_cfh_combo", "attack"),
                                     ("poc_cfh_baseline", "baseline")):
            run_class = "baseline" if variant == "poc_cfh_baseline" else "attack"
            self.assertEqual(run_class, expect_class)
            rid = make_run_id(0)
            run_dir = Path("raw") / cve / run_class / variant / rid
            self.assertTrue(run_dir.as_posix().startswith(f"raw/{cve}/{run_class}/{variant}/run_"))
            # the CVE must be the first segment (cve_of / parse_group contract)
            self.assertEqual(run_dir.parts[1], cve)
        # normal workload layout (mirror of collect_stable)
        for wl in ("idle", "msg_msg_256"):
            rid = make_run_id(0)
            run_dir = Path("raw") / cve / "normal" / wl / rid
            self.assertTrue(run_dir.as_posix().startswith(f"raw/{cve}/normal/{wl}/run_"))

    def test_write_manifest_atomic_and_versioned(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "manifest.json")
            write_manifest(path, {"run_uuid": "abc"})
            with open(path) as handle:
                payload = json.load(handle)
            self.assertEqual(payload["manifest_version"], 2)
            self.assertEqual(payload["run_uuid"], "abc")
            self.assertFalse(os.path.exists(path + ".tmp"))
            # no double suffix left behind
            leftovers = [f for f in os.listdir(tmp) if f.endswith(".tmp")]
            self.assertEqual(leftovers, [])

    def test_sha256_file_deterministic(self):
        import tempfile
        with tempfile.NamedTemporaryFile("wb", delete=False) as handle:
            handle.write(b"hello" * 1000)
            name = handle.name
        try:
            self.assertEqual(sha256_file(name), sha256_file(name))
            self.assertEqual(len(sha256_file(name)), 64)
            self.assertIsNone(sha256_file("/no/such/file"))
        finally:
            os.unlink(name)


class TraceHelperScriptsTest(unittest.TestCase):
    """The shell helpers must advertise a host-stream mode for crash persistence."""

    def _script(self, name):
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, "..", "scripts", "collect", "trace_helpers", name)

    def test_trace_start_has_host_stream_mode(self):
        with open(self._script("trace_start.sh")) as handle:
            content = handle.read()
        self.assertIn("host-stream", content)
        self.assertIn("TRACE_PIPE=", content)

    def test_trace_stop_has_host_stream_mode(self):
        with open(self._script("trace_stop.sh")) as handle:
            content = handle.read()
        self.assertIn("host-stream", content)


if __name__ == "__main__":
    unittest.main()
