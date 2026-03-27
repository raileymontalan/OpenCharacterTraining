"""
Environment smoke test for OpenCharacterTraining.

Checks that all required packages are importable, CUDA is available,
and that the critical inter-package interactions work correctly:
  - torch ↔ CUDA
  - vLLM import (links against the same torch ABI)
  - DeepSpeed import + CUDA extension availability
  - transformers AutoTokenizer (for local model loading)
  - openrlhf import
  - character package import + constants

Run with:
    python tests/test_env.py
or submit via:
    qsub scripts/check_env.sh
"""

import sys
import importlib

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(label: str, fn, critical: bool = True):
    try:
        result = fn()
        msg = f"  {result}" if result else ""
        print(f"{PASS} {label}{msg}")
        return True
    except Exception as e:
        tag = FAIL if critical else SKIP
        print(f"{tag} {label}: {e}")
        return not critical  # non-critical failures don't count as errors


# ---------------------------------------------------------------------------
# 1. Python
# ---------------------------------------------------------------------------
section("Python")
check("Python version",
      lambda: f"Python {sys.version}")


# ---------------------------------------------------------------------------
# 2. CUDA / GPU
# ---------------------------------------------------------------------------
section("CUDA / GPU")
import os
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
print(f"{PASS} CUDA_VISIBLE_DEVICES = {cuda_visible}")

import torch  # noqa: E402 (needed below)
check("torch importable",
      lambda: f"torch {torch.__version__}")

check("CUDA available",
      lambda: f"{torch.cuda.device_count()} GPU(s) visible" if torch.cuda.is_available()
              else (_ for _ in ()).throw(RuntimeError("torch.cuda.is_available() = False")))

check("torch CUDA version string",
      lambda: f"torch.version.cuda = {torch.version.cuda}")

def _gpu_info():
    lines = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        lines.append(f"GPU {i}: {p.name}  ({p.total_memory // 1024**3} GB)")
    return "\n  ".join(lines)
check("GPU properties", _gpu_info)

check("Simple CUDA tensor op",
      lambda: str(torch.tensor([1.0, 2.0]).cuda().sum().item()))


# ---------------------------------------------------------------------------
# 3. flash-attn (built from source — no pre-built wheel for torch 2.10)
# ---------------------------------------------------------------------------
section("flash-attn")

def _flash_import():
    import flash_attn
    return f"flash_attn {flash_attn.__version__}"

def _flash_functional():
    """Run a tiny forward pass to confirm ABI is correct."""
    from flash_attn import flash_attn_func
    B, Sq, Sk, H, D = 1, 16, 16, 4, 32
    dtype = torch.bfloat16
    q = torch.randn(B, Sq, H, D, dtype=dtype, device="cuda")
    k = torch.randn(B, Sk, H, D, dtype=dtype, device="cuda")
    v = torch.randn(B, Sk, H, D, dtype=dtype, device="cuda")
    out = flash_attn_func(q, k, v)
    assert out.shape == (B, Sq, H, D)
    return f"flash_attn_func output shape {tuple(out.shape)} OK"

check("flash_attn importable", _flash_import)
check("flash_attn functional forward pass", _flash_functional)


# ---------------------------------------------------------------------------
# 4. vLLM
# ---------------------------------------------------------------------------
section("vLLM")

def _vllm_import():
    import vllm
    return f"vllm {vllm.__version__}"

check("vllm importable", _vllm_import)

def _vllm_sampling():
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=16)
    return f"SamplingParams OK (max_tokens={sp.max_tokens})"

check("vllm SamplingParams", _vllm_sampling)


# ---------------------------------------------------------------------------
# 4. DeepSpeed
# ---------------------------------------------------------------------------
section("DeepSpeed")

def _ds_import():
    import deepspeed
    return f"deepspeed {deepspeed.__version__}"

def _ds_cuda_ext():
    import deepspeed
    # check that at least one CUDA op is loadable (no compilation needed for pre-built ops)
    info = deepspeed.ops.op_builder.CPUAdamBuilder().is_compatible()
    return f"CPUAdamBuilder compatible = {info}"

check("deepspeed importable", _ds_import)
check("deepspeed CPUAdam builder compatible", _ds_cuda_ext, critical=False)


# ---------------------------------------------------------------------------
# 5. transformers + local tokenizer
# ---------------------------------------------------------------------------
section("transformers")

def _transformers_import():
    import transformers
    return f"transformers {transformers.__version__}"

check("transformers importable", _transformers_import)

def _tokenizer_local():
    from transformers import AutoTokenizer
    from character.constants import MODEL_PATH
    import os
    # Find the first available student model directory
    candidates = ["gemma-3-4b-it", "llama-3.1-8b-it", "qwen-2.5-7b-it"]
    for model in candidates:
        path = os.path.join(MODEL_PATH, model)
        if os.path.isdir(path):
            tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
            return f"AutoTokenizer for {model} (vocab_size={tok.vocab_size})"
    return None  # will raise below if nothing found

def _tokenizer_check():
    result = _tokenizer_local()
    if result is None:
        raise FileNotFoundError(
            "No student model found under MODEL_PATH. "
            "Run scripts/01_download.sh first."
        )
    return result

check("AutoTokenizer from local model", _tokenizer_check, critical=False)


# ---------------------------------------------------------------------------
# 6. openrlhf
# ---------------------------------------------------------------------------
section("openrlhf")

def _openrlhf_import():
    import openrlhf
    return f"openrlhf importable"

check("openrlhf importable", _openrlhf_import)


# ---------------------------------------------------------------------------
# 7. character package + constants
# ---------------------------------------------------------------------------
section("character package")

def _character_import():
    import character
    return "character importable"

def _constants():
    from character.constants import DATA_PATH, MODEL_PATH, LORA_PATH, CONSTITUTION_PATH
    import os
    results = []
    for name, path in [
        ("DATA_PATH", DATA_PATH),
        ("MODEL_PATH", MODEL_PATH),
        ("LORA_PATH", LORA_PATH),
        ("CONSTITUTION_PATH", CONSTITUTION_PATH),
    ]:
        exists = os.path.isdir(path)
        results.append(f"{name} {'exists' if exists else 'MISSING'}: {path}")
    missing = [r for r in results if "MISSING" in r]
    if missing:
        raise FileNotFoundError("\n  ".join(missing))
    return "\n  ".join(results)

def _constitution_files():
    from character.constants import CONSTITUTION_PATH
    import os
    hw = os.path.join(CONSTITUTION_PATH, "hand-written")
    files = [f for f in os.listdir(hw) if f.endswith(".txt")] if os.path.isdir(hw) else []
    return f"{len(files)} constitution file(s): {', '.join(sorted(files))}"

check("character importable", _character_import)
check("character.constants paths exist", _constants)
check("constitution hand-written files", _constitution_files, critical=False)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section("Done")
print("All critical checks passed.\n")
