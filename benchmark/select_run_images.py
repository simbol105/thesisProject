"""Pick a few progress images per optimization run and force-add them to git."""
import argparse
import re
import subprocess
from pathlib import Path

PROJECTS = Path(__file__).resolve().parents[1]
ROOTS = [PROJECTS / "trellis1" / "TRELLIS" / "optimization_output",
         PROJECTS / "trellis2" / "TRELLIS.2" / "optimization_output",
         PROJECTS / "trellis2" / "TRELLIS.2" / "topology_study" / "output"]
NAME = re.compile(r"^(?:(?:test|iter)_(\d+)_(?:view_)?(\d+)|init_(\d+))\.png$")


def run_images(run: Path, count: int):
    """One view at `count` evenly spaced evaluations, first and last included (region panels when saved)."""
    frames = {}
    folder = run / "region" if (run / "region").is_dir() else run / "test"
    for p in folder.glob("*.png"):
        m = NAME.match(p.name)
        if m:
            it, view = (-1, int(m.group(3))) if m.group(3) else (int(m.group(1)), int(m.group(2)))
            frames.setdefault(view, {})[it] = p
    if not frames:
        return []
    view = max(frames, key=lambda v: (len(frames[v]), -v))
    its = sorted(frames[view])
    if len(its) <= count:
        return [frames[view][i] for i in its]
    picks = sorted({its[round(k * (len(its) - 1) / (count - 1))] for k in range(count)})
    return [frames[view][i] for i in picks]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--add", action="store_true", help="git add -f the selection")
    args = parser.parse_args()

    selected = []
    for root in ROOTS:
        for metrics in sorted(root.rglob("metrics.jsonl")):
            if "check" not in metrics.relative_to(root).parts:      # short smoke-test runs
                selected += run_images(metrics.parent, args.count)
    size = sum(p.stat().st_size for p in selected)
    for p in selected:
        print(p.relative_to(PROJECTS))
    print(f"{len(selected)} images, {size / 1e6:.0f} MB")
    if args.add:
        for k in range(0, len(selected), 500):
            subprocess.run(["git", "add", "-f", "--"] + [str(p.relative_to(PROJECTS)) for p in selected[k:k + 500]],
                           cwd=PROJECTS, check=True)


if __name__ == "__main__":
    main()
