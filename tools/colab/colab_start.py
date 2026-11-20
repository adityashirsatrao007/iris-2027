"""Colab bootstrap: turn the runtime into a private Tailscale SSH host.

Paste this whole file into a Colab cell (or run `!python colab_start.py`) with a
GPU runtime selected. It prints the tailnet address you give to the local wrapper.

Why Tailscale: the runtime never opens a public port. Tailscale puts it on a
private WireGuard mesh, and `tailscale up --ssh` runs the SSH server inside
tailscaled, so nothing is reachable except from devices on your own tailnet.

The auth key is read with getpass, so it is never written into the notebook.
"""
import getpass
import os
import shutil
import subprocess
import sys
import time

HOSTNAME = "colab-iris"


def sh(cmd, check=False, capture=True):
    print(f"$ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, text=True,
                            capture_output=capture)
    if capture:
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        if out:
            print(out, flush=True)
        if err:
            print(err, file=sys.stderr, flush=True)
    if check and result.returncode != 0:
        raise SystemExit(f"command failed ({result.returncode}): {cmd}")
    return result


print("=== runtime ===")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo 'no GPU attached'")
sh("python -c \"import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())\"")

if not shutil.which("tailscale"):
    print("\n=== installing tailscale ===")
    sh("curl -fsSL https://tailscale.com/install.sh | sh", check=True)

# Colab usually hides /dev/net/tun. Create it if we can; otherwise fall back to
# userspace networking, which still terminates inbound tailnet TCP connections.
print("\n=== checking TUN availability ===")
sh("mkdir -p /dev/net; mknod /dev/net/tun c 10 200 2>/dev/null || true; "
   "chmod 666 /dev/net/tun 2>/dev/null || true; ls -l /dev/net/tun 2>/dev/null || echo 'no tun device'")
mode_flags = "--tun=userspace-networking"
if os.path.exists("/dev/net/tun"):
    mode_flags = ""
    print("TUN device available: using kernel networking")
else:
    print("No TUN device: using userspace networking")

print("\n=== starting tailscaled ===")
sh("pkill -f 'tailscaled' || true")
subprocess.Popen(
    "nohup tailscaled " + mode_flags + " --state=/tmp/tailscaled.state "
    "--socket=/tmp/tailscaled.sock >/tmp/tailscaled.log 2>&1 &",
    shell=True,
)
time.sleep(4)

# getpass keeps the credential out of the saved notebook.
authkey = os.environ.get("TS_AUTHKEY") or getpass.getpass("Tailscale auth key (tskey-auth-...): ").strip()
if not authkey:
    raise SystemExit("no auth key supplied")
os.environ["TS_AUTHKEY"] = authkey

print("\n=== joining tailnet ===")
sh(f"tailscale --socket=/tmp/tailscaled.sock up --authkey={authkey} "
   f"--hostname={HOSTNAME} --ssh --accept-dns=false", check=True)
os.environ.pop("TS_AUTHKEY", None)

time.sleep(2)
print("\n=== status ===")
sh("tailscale --socket=/tmp/tailscaled.sock status")

ip = (subprocess.run("tailscale --socket=/tmp/tailscaled.sock ip -4",
                     shell=True, text=True, capture_output=True)
      .stdout.strip().splitlines() or [""])[0]

print("\n" + "=" * 62)
print(f"TARGET: {ip or '(no address reported)'}")
print("Give that address to the local wrapper:")
print(f'  .\\tools\\colab\\set-host.ps1 {ip or "<ip>"}')
print("=" * 62)

# Keep sshd out of the picture entirely; Tailscale SSH is the server.
print("\nThe runtime is reachable only from your tailnet. No public port is open.")
