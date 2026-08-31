"""Output-safety and artifact helpers.

Enforces the IMPLEMENTATION_PLAN.md principle "数据不可覆盖":
every collection run and experiment writes to a unique directory, and no
existing non-empty output directory is reused unless the caller explicitly
opts in with force=True.
"""
import hashlib
import json
import os
import tempfile


def require_empty_dir(path, force=False, create=True):
    """Return *path* if it is safe to use as an output directory, else raise.

    - path does not exist   -> created (unless create=False).
    - path exists, empty    -> accepted (reuse is safe).
    - path exists, non-empty -> FileExistsError unless force=True.
    """
    if os.path.isdir(path) and os.listdir(path):
        if not force:
            raise FileExistsError(
                "Output directory is not empty: {}\n"
                "Refusing to overwrite existing artifacts. Pass force=True only "
                "when you explicitly intend to resume or overwrite this output.".format(path)
            )
    elif create and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def sha256_file(path, chunk_size=1 << 20):
    """Return the hex SHA-256 of a file (streamed)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path, payload):
    """Atomically write a JSON file (tmp + rename) so partial writes never survive."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def read_json(path):
    """Read a JSON file and return the parsed object."""
    with open(path) as fh:
        return json.load(fh)


def snapshot_git_revision():
    """Return (commit, dirty) of the repo containing BASE_DIR, or (None, None)."""
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
        ).stdout.strip() != ""
        return (commit or None, dirty)
    except Exception:
        return (None, None)
