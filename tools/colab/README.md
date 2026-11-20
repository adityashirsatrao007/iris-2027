# OpenCode → Google Colab bridge

Drive a Colab GPU runtime from this machine's terminal: run Python, install
packages, move files, read logs. No public port is opened on the runtime.

## How it works

```
local OpenCode shell
        │  ssh colab
        ▼
Tailscale private mesh (WireGuard, 100.x.y.z)
        │  inbound terminated by tailscaled on the runtime
        ▼
Colab VM (paid compute units) ── GPU
        └── Tailscale SSH server (inside tailscaled, not sshd)
```

Colab's own FAQ disallows "remote control such as SSH shells" **only** on free
runtimes without a positive compute-unit balance. A paid plan with balance lifts
that restriction; keep a positive balance or the runtime may be terminated.

Tailscale rather than an ngrok/cloudflared tunnel because those publish a
reachable public endpoint. A tailnet address is only routable by devices you have
authorised, so the runtime is never exposed publicly.

## One-time setup

1. **Tailscale auth key.** Go to <https://login.tailscale.com/admin/settings/keys>
   and create a key with **Reusable** and **Pre-approved** on, and **Ephemeral**
   on so dead runtimes clean themselves up.

2. **Local client.** `winget install --id Tailscale.Tailscale -e` (needs admin),
   then sign in with the same account.

3. **Start the runtime.** In Colab: *Runtime → Change runtime type → T4 GPU*, then
   upload `tools/colab/colab_start.py` and run, in a cell:

   ```python
   import os; os.environ["TS_AUTHKEY"] = "tskey-auth-..."
   exec(open("/content/colab_start.py").read())
   ```

   The key is read from the environment, so it never enters the script file.
   Clear that cell afterwards if you plan to share the notebook.

4. **Register the address.** The cell prints `TARGET: 100.x.y.z`.

   ```powershell
   .\tools\colab\set-host.ps1 100.x.y.z
   ```

   That writes a managed block into `%USERPROFILE%\.ssh\config`. Re-running only
   updates the address; nothing else in the file is touched.

## Daily use

```powershell
.\tools\colab\colab.ps1 status
.\tools\colab\colab.ps1 exec "nvidia-smi"
.\tools\colab\colab.ps1 py notebooks\colab_run.py --stage probe
.\tools\colab\colab.ps1 pip spandrel spandrel_extra_arches
.\tools\colab\colab.ps1 push . /content/project
.\tools\colab\colab.ps1 pull /content/project/results .\results_remote
```

## When the runtime restarts

Colab deletes the VM on idle timeout, on manual disconnect, and at the maximum
lifetime. The tailnet address changes. Recovery is two steps:

1. Re-run the Colab cell (step 3 above). With a Reusable key it rejoins
   immediately; `--hostname=colab-iris` keeps the same tailnet name.
2. Re-run `set-host.ps1` with the new address printed by the cell.

Work survives only if it is on `/content` **and** copied out with `pull`, or
pushed to Drive inside the notebook. The runtime filesystem is destroyed with
the VM, so `pull` results before closing the tab.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `Connection timed out` | new VM, old address in config | re-run the Colab cell, then `set-host.ps1` |
| `Permission denied (publickey)` | ACL lacks an ssh rule | add `"ssh"` accept for `autogroup:member` in the tailnet ACL |
| `tailscale up` fails | expired or already-used key | issue a new auth key |
| Works then dies after ~1h | idle timeout, or compute units at zero | keep the notebook active; check the balance |
| `command not found: nvidia-smi` | CPU runtime selected | *Change runtime type → T4 GPU*, reconnect |

## Files

| Path | Purpose |
|---|---|
| `colab_start.py` | runs inside Colab; joins the tailnet and starts Tailscale SSH |
| `set-host.ps1` | writes/updates the `Host colab` block in the ssh config |
| `colab.ps1` | local driver: status, exec, py, pip, put, get, push, pull |
