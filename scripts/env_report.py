"""Write a reproducibility record for the paper's artifact statement."""
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

out = Path("results")
out.mkdir(parents=True, exist_ok=True)

record = {
    "python": sys.version,
    "platform": platform.platform(),
    "processor": platform.processor(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
for name in ("torchvision", "numpy", "pandas", "sklearn", "datasets", "spandrel", "PIL"):
    try:
        module = __import__(name)
        record[name] = getattr(module, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        record[name] = f"missing: {exc}"

try:
    record["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                      text=True, timeout=10).stdout.strip() or "not a git repository"
except Exception:  # noqa: BLE001
    record["commit"] = "git unavailable"

(out / "ENVIRONMENT.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))

