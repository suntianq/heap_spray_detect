"""Unit tests for ftrace/bpftrace line parsing and marker detection.

Covers IMPLEMENTATION_PLAN.md section 10.1 items:
- ftrace/bpftrace line parsing
- marker missing / duplicate / reversed-order / normal cases
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "preprocess"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import trace2csv
from synthetic import (bpftrace_line, kfree_line, kmalloc_line, marker_line,
                       write_trace_file)


class TestFtraceParsing(unittest.TestCase):
    def test_parse_kmalloc(self):
        line = kmalloc_line("bash", 1858, 11_852_213_000, 0x688bb814, 32, 32, cpu=3)
        rec = trace2csv.parse_ftrace_line(line)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["timestamp_ns"], 11_852_213_000)
        self.assertEqual(rec["pid"], 1858)
        self.assertEqual(rec["tid"], 1858)
        self.assertEqual(rec["cpu"], 3)
        self.assertEqual(rec["comm"], "bash")
        self.assertEqual(rec["op"], "ALLOC")
        self.assertEqual(rec["ptr"], "00000000688bb814")
        self.assertEqual(rec["bytes_req"], 32)
        self.assertEqual(rec["bytes_alloc"], 32)
        self.assertIn("ffffffff8139c7f1", rec["call_site"])

    def test_parse_kfree(self):
        line = kfree_line("bash", 1858, 11_852_213_000, 0x688bb814, cpu=2)
        rec = trace2csv.parse_ftrace_line(line)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["cpu"], 2)
        self.assertEqual(rec["op"], "FREE")
        self.assertEqual(rec["ptr"], "00000000688bb814")

    def test_parse_bpftrace(self):
        line = bpftrace_line(11_852_213_000, 100, 100, "bash", "ALLOC", 0x1234, 64, 128)
        rec = trace2csv.parse_bpftrace_line(line)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["timestamp_ns"], 11_852_213_000)
        self.assertEqual(rec["pid"], 100)
        self.assertEqual(rec["tid"], 100)
        self.assertEqual(rec["cpu"], 0)
        self.assertEqual(rec["op"], "ALLOC")
        self.assertEqual(rec["bytes_req"], 64)
        self.assertEqual(rec["bytes_alloc"], 128)

    def test_parse_garbage_returns_none(self):
        self.assertIsNone(trace2csv.parse_ftrace_line("garbage line"))
        self.assertIsNone(trace2csv.parse_bpftrace_line("not,bpftrace"))

    def test_detect_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            ft = os.path.join(tmp, "a.log")
            bt = os.path.join(tmp, "b.log")
            unk = os.path.join(tmp, "c.log")
            write_trace_file(ft, [kmalloc_line("bash", 1, 1000, 0x1, 32, 32)])
            write_trace_file(bt, [bpftrace_line(1000, 1, 1, "bash", "ALLOC", 0x1, 32, 32)])
            write_trace_file(unk, ["not a trace"])
            self.assertEqual(trace2csv.detect_format(ft), "ftrace")
            self.assertEqual(trace2csv.detect_format(bt), "bpftrace")
            self.assertEqual(trace2csv.detect_format(unk), "unknown")


class TestMarkerDetection(unittest.TestCase):
    def test_markers_normal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.log")
            write_trace_file(path, [
                kmalloc_line("poc", 100, 1_000_000_000, 0x1, 32, 32),
                marker_line(1_100_000_000, "SPRAY_START"),
                kmalloc_line("poc", 100, 1_200_000_000, 0x2, 64, 64),
                marker_line(1_300_000_000, "SPRAY_END"),
            ])
            markers = trace2csv.detect_markers(path)
            self.assertEqual(markers, {"SPRAY_START": 1_100_000_000, "SPRAY_END": 1_300_000_000})

    def test_markers_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.log")
            write_trace_file(path, [kmalloc_line("bash", 1, 1000, 0x1, 32, 32)])
            self.assertEqual(trace2csv.detect_markers(path), {})

    def test_markers_duplicate_last_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.log")
            write_trace_file(path, [
                marker_line(1_000_000_000, "SPRAY_START"),
                marker_line(1_500_000_000, "SPRAY_START"),
                marker_line(2_000_000_000, "SPRAY_END"),
            ])
            markers = trace2csv.detect_markers(path)
            self.assertEqual(markers["SPRAY_START"], 1_500_000_000)  # last wins

    def test_markers_reversed_order_detected(self):
        """Both markers are parsed; an END earlier than START must be visible to validation."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.log")
            write_trace_file(path, [
                marker_line(2_000_000_000, "SPRAY_START"),
                marker_line(1_000_000_000, "SPRAY_END"),
            ])
            markers = trace2csv.detect_markers(path)
            self.assertEqual(markers["SPRAY_START"], 2_000_000_000)
            self.assertEqual(markers["SPRAY_END"], 1_000_000_000)
            # validation lives in csv2features: END <= START must be rejected
            self.assertLessEqual(markers["SPRAY_END"], markers["SPRAY_START"])


class TestConvertFile(unittest.TestCase):
    def test_convert_file_ftrace(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "run.log")
            dst = os.path.join(tmp, "run.csv")
            write_trace_file(src, [
                kmalloc_line("bash", 1858, 11_852_213_000, 0x688bb814, 32, 32),
                marker_line(12_000_000_000, "SPRAY_START"),
                kfree_line("bash", 1858, 12_500_000_000, 0x688bb814),
            ])
            count, markers = trace2csv.convert_file(src, dst)
            self.assertEqual(count, 2)
            self.assertEqual(markers, {"SPRAY_START": 12_000_000_000})
            self.assertTrue(os.path.exists(dst))

    def test_convert_file_non_trace_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "qemu.log")
            dst = os.path.join(tmp, "qemu.csv")
            with open(src, "w") as f:
                f.write("Debian GNU/Linux 9 pwn ttyS0\npwn login:\n")
            count, markers = trace2csv.convert_file(src, dst)
            self.assertIsNone(count)
            self.assertEqual(markers, {})
            self.assertFalse(os.path.exists(dst))


class TestManifestHandling(unittest.TestCase):
    def test_manifest_status_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "manifest.json"), "w") as f:
                f.write('{"status": "valid", "spray_start_ns": 10, "spray_end_ns": 20}')
            self.assertEqual(trace2csv.manifest_status(tmp), "valid")
            self.assertEqual(trace2csv.manifest_spray_window(tmp),
                             {"SPRAY_START": 10, "SPRAY_END": 20})

    def test_manifest_status_invalid_has_no_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "manifest.json"), "w") as f:
                f.write('{"status": "invalid"}')
            self.assertEqual(trace2csv.manifest_status(tmp), "invalid")
            self.assertEqual(trace2csv.manifest_spray_window(tmp), {})

    def test_manifest_missing_has_no_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(trace2csv.manifest_status(tmp))
            self.assertEqual(trace2csv.manifest_spray_window(tmp), {})

    def test_manifest_valid_but_empty_window_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "manifest.json"), "w") as f:
                f.write('{"status": "valid", "spray_start_ns": 20, "spray_end_ns": 20}')
            self.assertEqual(trace2csv.manifest_spray_window(tmp), {})


if __name__ == "__main__":
    unittest.main()
