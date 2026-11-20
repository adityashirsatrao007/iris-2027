import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from src.data import build_manifest

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
df = build_manifest(a.input, a.seed)
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.output, index=False)
print(f"Wrote {len(df)} rows to {a.output}")
print(df.label.value_counts().to_string())
