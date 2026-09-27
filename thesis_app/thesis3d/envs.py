"""Run a command inside one of the project's conda environments, as run_script.sh does."""
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional

ENVS = {
    "trellis": {"cuda": "12.8", "note": "TRELLIS generation and refinement (needs an H100)"},
    "trellis2": {"cuda": "12.8", "note": "TRELLIS.2 generation, refinement and the benchmark"},
    "recgen": {"cuda": "11.8", "note": "RecGen (runs inside its own pixi project)"},
    "sam3d": {"cuda": "12.8", "note": "SAM3D"},
}


def shell_command(env: str, cwd: Path, args: List, python: Optional[str] = None,
                  exports: Optional[List[str]] = None) -> List[str]:
    """bash -c script that activates `env` and runs `python args` in `cwd`."""
    q = shlex.quote
    interpreter = python or '"$CONDA_PREFIX/bin/python"'
    lines = [
        "set +u",
        "source ~/miniforge3/etc/profile.d/conda.sh",
        "source ~/.bashrc >/dev/null 2>&1 || true",
        "type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1 || true",
        f"module load devel/cuda/{ENVS[env]['cuda']} 2>/dev/null || true",
        f"conda activate {env}",
        *[f"export {e}" for e in (exports or [])],
        f"cd {q(str(cwd))}",
        f"{interpreter} " + " ".join(q(str(a)) for a in args),
    ]
    return ["bash", "-c", "\n".join(lines)]


def run(cmd: List[str], dry_run: bool = False) -> int:
    """Run a prepared command, echoing it first; returns its exit code."""
    printable = cmd[-1] if cmd[:2] == ["bash", "-c"] else " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}\n", flush=True)
    if dry_run:
        return 0
    return subprocess.run([str(c) for c in cmd], check=False).returncode


def env_exists(env: str) -> bool:
    return (Path.home() / "miniforge3" / "envs" / env / "bin" / "python").exists()
