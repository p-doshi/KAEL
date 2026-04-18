"""
KAEL Phase 0 Setup
Run this once to install dependencies and verify the environment.
Usage: python scripts/setup.py
"""

import subprocess
import sys
import platform
from pathlib import Path

ROOT = Path(__file__).parent.parent


def run(cmd: str, check: bool = True):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if check and result.returncode != 0:
        print(f"  ERROR: command failed with code {result.returncode}")
        sys.exit(1)
    return result.returncode == 0


def check_python():
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or v.minor < 11:
        print("WARNING: Python 3.11+ recommended")
    else:
        print("OK")


def check_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"CUDA available | GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram:.1f}GB")
            if vram < 20:
                print("WARNING: <20GB VRAM — switch to Qwen2.5-7B or enable 4bit quantization in config.py")
            else:
                print("OK")
        else:
            print("WARNING: CUDA not available — will run on CPU (very slow)")
    except ImportError:
        print("PyTorch not installed yet")


def install_dependencies():
    packages = [
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "bitsandbytes>=0.42.0",
        "peft>=0.10.0",
        "datasets>=2.18.0",
        "faiss-gpu",
        "rich>=13.0.0",
        "fastapi>=0.110.0",
        "uvicorn>=0.27.0",
        "numpy",
        "scipy",
        "scikit-learn",
        "wandb",
        "tqdm",
        "sentencepiece",
    ]

    print("\nInstalling dependencies...")
    # Install torch first with CUDA support
    run("pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cu121 -q")
    # Install rest
    pkg_str = " ".join(f'"{p}"' for p in packages[1:])
    run(f"pip install {pkg_str} -q")
    print("Dependencies installed")


def create_dirs():
    dirs = [
        ROOT / "memory",
        ROOT / "logs" / "eval",
        ROOT / "checkpoints",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("Directories created")


def run_tests():
    print("\nRunning Phase 0 tests...")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "test_phase0.py")],
        cwd=str(ROOT),
    )
    if result.returncode == 0:
        print("All tests passed")
    else:
        print("Some tests failed — check output above")
    return result.returncode == 0


def main():
    print("=" * 50)
    print("KAEL Phase 0 Setup")
    print("=" * 50)

    print("\n[1] Checking Python...")
    check_python()

    print("\n[2] Installing dependencies...")
    install_dependencies()

    print("\n[3] Checking CUDA...")
    check_cuda()

    print("\n[4] Creating directories...")
    create_dirs()

    print("\n[5] Running tests...")
    tests_ok = run_tests()

    print("\n" + "=" * 50)
    if tests_ok:
        print("Setup complete. Run KAEL with:")
        print(f"  cd {ROOT}")
        print("  python interface/repl.py")
        print("\nOr run benchmarks only:")
        print("  python -c \"from eval.phase0_eval import *; ...\"")
    else:
        print("Setup complete but some tests failed.")
        print("Check the output above before running KAEL.")
    print("=" * 50)


if __name__ == "__main__":
    main()
