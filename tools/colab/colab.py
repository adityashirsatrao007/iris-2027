"""Drive a Google Colab runtime from this machine over Tailscale SSH.

    python tools/colab/colab.py host 100.x.y.z        register the runtime address
    python tools/colab/colab.py status                runtime, GPU, torch, disk
    python tools/colab/colab.py exec "nvidia-smi"     run a shell command
    python tools/colab/colab.py py path/to/x.py a b   upload a .py file and run it
    python tools/colab/colab.py pip pandas seaborn    install packages remotely
    python tools/colab/colab.py put local remote      upload one file
    python tools/colab/colab.py get remote local      download one file
    python tools/colab/colab.py push dir /content/x   upload a directory tree
    python tools/colab/colab.py pull /content/x dir   download a directory tree

Everything the connection needs is passed on the ssh command line, so nothing
is written to %USERPROFILE%\\.ssh and no key files are required: Tailscale SSH
authenticates with the tailnet identity, not with ssh keys. The runtime address
is kept in tools/colab/target.json.

Transfers stream over stdin/stdout with binary handles, so they are byte-exact
and do not depend on the remote sftp subsystem (Tailscale SSH has none).

Only the standard library is used, so any Python 3 interpreter can run this.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET_FILE = HERE / "target.json"
PORT = 22
TIMEOUT = 15


class Target:
    def __init__(self, address: str, user: str = "root") -> None:
        self.address = address
        self.user = user

    @classmethod
    def load(cls) -> "Target":
        if not TARGET_FILE.exists():
            die("no runtime registered.\n"
                "  run the Colab cell, then:  python tools/colab/colab.py host <address>")
        data = json.loads(TARGET_FILE.read_text(encoding="utf-8"))
        return cls(data["address"], data.get("user", "root"))

    def save(self) -> None:
        TARGET_FILE.write_text(json.dumps({"address": self.address, "user": self.user},
                                          indent=2) + "\n", encoding="utf-8")

    def options(self) -> list[str]:
        # NUL is the Windows null device, /dev/null elsewhere.
        null = "NUL" if os.name == "nt" else "/dev/null"
        return [
            "-o", f"HostName={self.address}",
            "-o", f"User={self.user}",
            "-o", f"Port={PORT}",
            "-o", f"ConnectTimeout={TIMEOUT}",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-o", "TCPKeepAlive=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=" + null,
            "-o", "GlobalKnownHostsFile=" + null,
            "-o", "LogLevel=ERROR",
        ]


DRY_RUN = False


def die(message: str, code: int = 1) -> "NoReturn":  # type: ignore[name-defined]
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def ssh(command: str, target: Target, stdin=None, stdout=None, check=True):
    cmd = ["ssh", *target.options(), "--", target.address, command]
    if DRY_RUN:
        print(" ".join(shlex.quote(c) for c in cmd[:len(target.options()) + 3]), "<stdin>"
              if stdin else "", flush=True)
        return subprocess.CompletedProcess(cmd, 0)
    result = subprocess.run(cmd, stdin=stdin, stdout=stdout)
    if check and result.returncode != 0:
        die(f"remote command failed ({result.returncode}): {command}")
    return result


# ------------------------------------------------------------------- verbs
def cmd_host(args) -> None:
    target = Target(args.address, args.user)
    target.save()
    print(f"runtime: {target.user}@{target.address}:{PORT}")
    print(f"saved:   {TARGET_FILE}")
    print("\ntesting connection ...")
    result = subprocess.run(
        ["ssh", *target.options(), "--", target.address, "echo connected; hostname"],
        capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr.strip() + "\n" if result.stderr.strip() else "")
    if result.returncode != 0:
        print("\nnot reachable yet. check that:\n"
              "  1. the Colab cell printed this exact address\n"
              "  2. Tailscale on this PC is running and on the same tailnet\n"
              "  3. the runtime is still alive (Colab recycles idle VMs)")


def cmd_status(args) -> None:
    target = Target.load()
    ssh(" ; ".join([
        'echo "host:   $(hostname)"',
        "echo \"disk:\" ; df -h /content 2>/dev/null | tail -1",
        "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader "
        "2>/dev/null || echo 'gpu:    none attached'",
        "python -c \"import torch;print('torch: ',torch.__version__,'| cuda ',"
        "torch.cuda.is_available())\" 2>/dev/null || echo 'torch:  not installed'",
    ]), target)


def cmd_exec(args) -> None:
    target = Target.load()
    command = " ".join(args.command)
    if not command.strip():
        die('usage: colab.py exec "<shell command>"')
    ssh(command, target)


def cmd_py(args) -> None:
    target = Target.load()
    local = Path(args.script)
    if not local.exists():
        die(f"missing script: {local}")
    remote = f"/content/_upload/{local.name}"
    upload(local, remote, target)
    extra = " ".join(shlex.quote(a) for a in args.script_args)
    ssh(f"cd /content && python {shlex.quote(remote)} {extra}".strip(), target)


def cmd_pip(args) -> None:
    target = Target.load()
    ssh("python -m pip install -q " + " ".join(shlex.quote(p) for p in args.packages), target)


def cmd_put(args) -> None:
    upload(Path(args.local), args.remote, Target.load())


def cmd_get(args) -> None:
    download(args.remote, Path(args.local), Target.load())


def cmd_push(args) -> None:
    target = Target.load()
    src = Path(args.local_dir)
    if not src.is_dir():
        die(f"not a directory: {src}")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "push.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for item in src.iterdir():
                tar.add(item, arcname=item.name)
        upload(archive, "/tmp/_colab_push.tar.gz", target)
    dest = shlex.quote(args.remote_dir)
    ssh(f"mkdir -p {dest} && tar -xzf /tmp/_colab_push.tar.gz -C {dest} "
        f"&& rm -f /tmp/_colab_push.tar.gz", target)
    print(f"pushed {src} -> {args.remote_dir}")


def cmd_pull(args) -> None:
    target = Target.load()
    dest = Path(args.local_dir)
    dest.mkdir(parents=True, exist_ok=True)
    ssh(f"tar -czf /tmp/_colab_pull.tar.gz -C {shlex.quote(args.remote_dir)} .", target)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "pull.tar.gz"
        download("/tmp/_colab_pull.tar.gz", archive, target)
        with tarfile.open(archive, "r:gz") as tar:
            extract(tar, dest)
    ssh("rm -f /tmp/_colab_pull.tar.gz", target)
    print(f"pulled {args.remote_dir} -> {dest}")


# --------------------------------------------------------------- transfers
def extract(tar: tarfile.TarFile, dest: Path) -> None:
    try:
        tar.extractall(dest, filter="data")
    except TypeError:  # Python < 3.11.4
        tar.extractall(dest)


def upload(local: Path, remote: str, target: Target) -> None:
    parent = str(Path(remote).parent).replace("\\", "/")
    if parent not in ("", "/"):
        ssh(f"mkdir -p {shlex.quote(parent)}", target, check=False)
    with open(local, "rb") as fh:
        ssh(f"cat > {shlex.quote(remote)}", target, stdin=fh)
    print(f"put  {local} -> {remote}")


def download(remote: str, local: Path, target: Target) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    with open(local, "wb") as fh:
        ssh(f"cat {shlex.quote(remote)}", target, stdout=fh)
    print(f"get  {remote} -> {local}")


# --------------------------------------------------------------------- cli
def build_parser():
    p = argparse.ArgumentParser(prog="colab.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                   help="print the ssh invocation instead of connecting")
    sub = p.add_subparsers(dest="action", required=True)

    h = sub.add_parser("host", help="register the runtime address")
    h.add_argument("address")
    h.add_argument("--user", default="root")
    h.set_defaults(func=cmd_host)

    s = sub.add_parser("status", help="runtime summary")
    s.set_defaults(func=cmd_status)

    e = sub.add_parser("exec", help="run a shell command")
    e.add_argument("command", nargs=argparse.REMAINDER)
    e.set_defaults(func=cmd_exec)

    y = sub.add_parser("py", help="upload and run a python script")
    y.add_argument("script")
    y.add_argument("script_args", nargs=argparse.REMAINDER)
    y.set_defaults(func=cmd_py)

    i = sub.add_parser("pip", help="install packages remotely")
    i.add_argument("packages", nargs="+")
    i.set_defaults(func=cmd_pip)

    u = sub.add_parser("put", help="upload one file")
    u.add_argument("local")
    u.add_argument("remote")
    u.set_defaults(func=cmd_put)

    g = sub.add_parser("get", help="download one file")
    g.add_argument("remote")
    g.add_argument("local")
    g.set_defaults(func=cmd_get)

    ps = sub.add_parser("push", help="upload a directory")
    ps.add_argument("local_dir")
    ps.add_argument("remote_dir")
    ps.set_defaults(func=cmd_push)

    pl = sub.add_parser("pull", help="download a directory")
    pl.add_argument("remote_dir")
    pl.add_argument("local_dir")
    pl.set_defaults(func=cmd_pull)
    return p


def main() -> None:
    global DRY_RUN
    args = build_parser().parse_args()
    DRY_RUN = args.dry_run
    if args.action == "exec" and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if args.action == "py" and args.script_args and args.script_args[0] == "--":
        args.script_args = args.script_args[1:]
    args.func(args)


if __name__ == "__main__":
    main()
