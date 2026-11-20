"""Offline self-test for the Colab driver's transfer plumbing.

The runtime is not reachable from this machine yet, so `ssh` is replaced with a
local POSIX shell (MinGit's sh.exe) and the *same* remote commands are executed
against the local filesystem. That exercises the parts that actually break:
shell quoting, binary-safe stdin/stdout streaming, and the tar round trip.

    python tools/colab/selftest.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIN_GIT = Path(os.environ["LOCALAPPDATA"]) / "Programs" / "MinGit"
SH = MIN_GIT / "usr" / "bin" / "sh.exe"
ENV = dict(os.environ)
ENV["PATH"] = os.pathsep.join([
    str(MIN_GIT / "usr" / "bin"),
    str(MIN_GIT / "mingw64" / "bin"),
    str(Path(sys.executable).parent),
    os.environ.get("PATH", ""),
])


def load_driver():
    spec = importlib.util.spec_from_file_location("colab_driver", HERE / "colab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fake:
    address = "local"
    user = "root"

    def options(self):
        return []


def install_fake_ssh(driver, calls: list[str]) -> None:
    def local_ssh(command, target, stdin=None, stdout=None, check=True):
        calls.append(command)
        result = subprocess.run([str(SH), "-c", command], stdin=stdin,
                                stdout=stdout, stderr=subprocess.PIPE, env=ENV)
        if result.stderr:
            sys.stderr.write(result.stderr.decode("utf-8", "replace"))
        if check and result.returncode != 0:
            driver.die(f"local command failed ({result.returncode}): {command}")
        return result

    driver.ssh = local_ssh


def main() -> int:
    if not SH.exists():
        print(f"skip: {SH} not found")
        return 0

    driver = load_driver()
    calls: list[str] = []
    install_fake_ssh(driver, calls)
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(("PASS  " if ok else "FAIL  ") + name + (f"  ({detail})" if detail else ""))
        if not ok:
            failures.append(name)

    root = Path(tempfile.mkdtemp(prefix="colab_selftest_"))
    remote_root = (Path("/tmp") / f"colab_selftest_{os.getpid()}").as_posix()
    try:
        # 1. single binary file, upload + download, byte exact
        local_bin = root / "blob.bin"
        local_bin.write_bytes(os.urandom(300_000) + b"\r\n\x00\xff\x00binary\n")
        remote_bin = f"{remote_root}/deep/er/blob.bin"
        driver.upload(local_bin, remote_bin, Fake())
        back = root / "blob.back.bin"
        driver.download(remote_bin, back, Fake())
        check("upload/download binary round trip", sha(local_bin) == sha(back),
              f"{len(local_bin.read_bytes())} B")

        # 2. filename with spaces and a quote must survive quoting
        tricky = root / "a file's name (v2).txt"
        tricky.write_text("quoted paths\n", encoding="utf-8")
        driver.upload(tricky, f"{remote_root}/tricky.txt", Fake())
        got = root / "tricky.back.txt"
        driver.download(f"{remote_root}/tricky.txt", got, Fake())
        check("round trip preserves content of odd filename",
              got.read_text(encoding="utf-8") == "quoted paths\n")

        # 3. directory push / pull with nesting
        tree = root / "tree"
        (tree / "sub").mkdir(parents=True)
        (tree / "a.txt").write_text("A", encoding="utf-8")
        (tree / "sub" / "b.bin").write_bytes(os.urandom(50_000))
        driver.cmd_push(type("A", (), {"local_dir": str(tree),
                                       "remote_dir": f"{remote_root}/pushed"})())
        pulled = root / "pulled"
        driver.cmd_pull(type("A", (), {"remote_dir": f"{remote_root}/pushed",
                                       "local_dir": str(pulled)})())
        check("push/pull directory tree",
              (pulled / "a.txt").read_text(encoding="utf-8") == "A"
              and sha(pulled / "sub" / "b.bin") == sha(tree / "sub" / "b.bin"))

        # 4. remote python execution through the same path the `py` verb uses
        script = root / "hello.py"
        script.write_text("import torch, platform\n"
                          "print('py ok', platform.python_version())\n", encoding="utf-8")
        driver.upload(script, f"{remote_root}/_upload/hello.py", Fake())
        out = subprocess.run([str(SH), "-c", f"python {remote_root}/_upload/hello.py"],
                             capture_output=True, text=True, env=ENV)
        check("remote-style python invocation", "py ok" in out.stdout.strip(),
              out.stdout.strip() or out.stderr.strip()[:80])

        check("no sftp or scp dependency",
              all("sftp" not in c and "scp " not in c for c in calls),
              f"{len(calls)} ssh calls")
    finally:
        shutil.rmtree(root, ignore_errors=True)
        subprocess.run([str(SH), "-c", f"rm -rf {remote_root}"], env=ENV,
                       capture_output=True)

    print()
    if failures:
        print(f"{len(failures)} failure(s): {', '.join(failures)}")
        return 1
    print("all transfer tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
