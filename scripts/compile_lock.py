#!/usr/bin/env python3
"""Generate a lock only on the native platform it claims to support."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLATFORM_NAMES = {
    "Darwin": "macos",
    "Linux": "linux",
    "Windows": "windows",
}
EXTRAS = ("benchmark", "dev", "fastembed", "notebook", "test", "vectorize")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=("macos", "linux", "windows"), required=True)
    parser.add_argument("--python", choices=("3.11", "3.12", "3.13"), required=True)
    parser.add_argument(
        "--profile",
        choices=("default", "cpu"),
        default="default",
        help="Linux supports an explicit CPU-only PyTorch profile.",
    )
    args = parser.parse_args()

    native_platform = PLATFORM_NAMES.get(platform.system())
    if native_platform != args.platform:
        parser.error(
            f"refusing to label a {platform.system()} resolution as {args.platform}; "
            "run this script natively on the target platform"
        )
    native_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if native_python != args.python:
        parser.error(f"this interpreter is Python {native_python}; rerun with Python {args.python}")
    if args.profile != "default" and args.platform != "linux":
        parser.error("the cpu profile is only defined for Linux")

    profile_suffix = "-cpu" if args.profile == "cpu" else ""
    output_name = f"{args.platform}{profile_suffix}-py{args.python.replace('.', '')}.txt"
    output = ROOT / "requirements" / output_name
    if args.platform == "macos":
        source = ROOT / "pyproject.toml"
    elif args.platform == "linux":
        profile_name = "linux-cpu.in" if args.profile == "cpu" else "linux-default.in"
        source = ROOT / "requirements" / "platform" / profile_name
    else:
        source = ROOT / "requirements" / "platform" / "windows.in"
    command = [
        sys.executable,
        "-m",
        "piptools",
        "compile",
        "--allow-unsafe",
        "--generate-hashes",
        "--strip-extras",
        "--output-file",
        str(output),
    ]
    if source.name == "pyproject.toml":
        for extra in EXTRAS:
            command.extend(["--extra", extra])
    command.append(str(source))
    subprocess.run(command, cwd=ROOT, check=True)  # noqa: S603
    print(f"Generated native {args.platform} lock: {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
