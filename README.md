# OpenCharacterTraining — Fork Notes

This is a personal fork of [maiush/OpenCharacterTraining](https://github.com/maiush/OpenCharacterTraining) by [@raileymontalan](https://github.com/raileymontalan). For the original documentation, see [UPSTREAM.md](UPSTREAM.md).

## Changes from Upstream

### 1. PBS/QSUB job scripts (`scripts/`)

Added a complete `scripts/` folder for running the pipeline on a PBS cluster (tested on the AISG HPC). Each pipeline step is its own job script; a shared `scripts/config.sh` holds all paths and model config.

```
scripts/
├── config.sh              # shared paths, model, teacher config — edit this
├── 00_setup.sh            # one-time package install (qsub, CPU node)
├── 01_download.sh  # download models, LIMA dataset, and upstream LoRAs (login node)
├── 02_gen_prompts.sh      # expand constitution to 50 prompts/facet
├── 03_teacher.sh          # serve teacher via vLLM + generate chosen responses
├── 04_student.sh          # generate rejected responses + format DPO data
├── 05_dpo_train.sh        # DPO fine-tuning (DeepSpeed)
├── 06_self_reflect.sh     # self-reflection data generation
├── 07_self_interact.sh    # self-interaction data generation + format SFT data
├── 08_fold_dpo.sh         # fold DPO LoRA into base model
├── 09_sft_train.sh        # SFT fine-tuning (DeepSpeed)
└── 10_merge_loras.sh      # merge DPO + SFT LoRAs into final persona LoRA
```

### 2. vLLM server + AsyncOpenAI teacher mode (`character/distillation/teacher.py`)

Upstream `teacher.py` runs the teacher model in-process via vLLM. This fork adds a second mode: the teacher is served externally via `vllm serve` and queried through the `AsyncOpenAI` client (pointed at localhost). This avoids loading a 120B teacher and the student on the same node.

- `--api_base`: enables the API mode (e.g. `http://localhost:8000/v1`)
- `--api_key`: API key (default `"EMPTY"` for local vLLM)
- `--concurrency`: number of concurrent async requests (default 32)
- The `<think>` prefill trick is preserved via `continue_final_message=True` in `extra_body`
- `02_teacher.sh` starts `vllm serve` in the background, polls `/health`, runs `teacher.py`, then kills the server on exit

### 3. Custom constitution support with model switching

All PBS scripts accept `CONSTITUTION` and `MODEL` as `qsub -v` variables. The model family is derived automatically (`FAMILY="${MODEL%%-*}"`), so the same scripts work for Gemma, Llama, and Qwen without modification.

### 4. Filipino constitution (`constitutions/hand-written/filipino-en.txt`)

Added a Filipino cultural persona grounded in Sikolohiyang Pilipino (Filipino indigenous psychology), drawing on the scholarship of Virgilio Enriquez, Rogelia Pe-Pua, Zeus Salazar, and M. Sta. Maria. The 10 traits are:

| Trait | Concept |
|---|---|
| `kapwa` | Shared identity — the foundational core of Filipino psychology |
| `pagmamalasakit` | Genuine active concern for others' wellbeing |
| `bayanihan` | Communal solidarity freely given |
| `pakikiramdam` | Shared inner perception — reading the unspoken |
| `hiya` | Relational propriety and dignity (not shame) |
| `utang na loob` | Gratitude and solidarity of the inner self (not debt) |
| `diskarte` | Resourceful ingenuity with limited means |
| `galang/respeto` | Sincere respect for elders and experience |
| `bahala na` | Decisive action in uncertainty, entrusting the outcome (not fatalism) |
| `pamilya` | Family as the center of identity and meaning |

Colonial distortions corrected per Enriquez: `hiya` ≠ shame; `utang na loob` ≠ debt/obligation; `bahala na` ≠ fatalism.

### 5. Bug fixes

| File | Fix |
|---|---|
| `.gitmodules` | Changed submodule URLs from SSH (`git@github.com:`) to HTTPS to work on clusters where port 22 is blocked |
| `character/introspection/data.py` | Added existence check before reading per-model files; upstream crashed with `FileNotFoundError` when a constitution had no trained model |
| `tools/merge_loras.py` | Fixed wrong LoRA path: `{family}-test` → `{family}-introspection` |
| `character/constants.py.example` | Added committed template; `character/constants.py` is gitignored upstream with no example provided |

---

## Setup (AISG PBS Cluster)

### 1. Clone

Submodule URLs use HTTPS (port 22 is blocked on this cluster):

```bash
git clone https://github.com/raileymontalan/OpenCharacterTraining.git
cd OpenCharacterTraining
git submodule sync
git submodule update --init --recursive
```

### 2. Pre-flight (login node)

Create the virtual environment and the `.env` file before submitting any jobs:

```bash
uv venv --python 3.11
```

```bash
# .env — required by pipeline scripts for model downloads and experiment tracking
export HF_TOKEN=<your_huggingface_token>
export WANDB_TOKEN=<your_wandb_token>
```

### 3. Download models and datasets (login node)

Run in the background so it survives SSH disconnects — downloads can take hours:

```bash
nohup bash scripts/01_download.sh > logs/download.log 2>&1 &
tail -f logs/download.log   # monitor progress
```

This downloads the following into `models/`:

| Directory | Source | Role |
|---|---|---|
| `gemma-3-4b-it` | `google/gemma-3-4b-it` | student model |
| `llama-3.1-8b-it` | `meta-llama/Llama-3.1-8B-Instruct` | student model |
| `qwen-2.5-7b-it` | `Qwen/Qwen2.5-7B-Instruct` | student model |
| `llama-3.3-70b-it` | `meta-llama/Llama-3.3-70B-Instruct` | prompt generation |
| `gpt-oss-120b` | `openai/gpt-oss-120b` | teacher model |
| `lima/` | `GAIR/lima` (HF dataset) | teacher prompt pool |

You only need the student model(s) you intend to train. Comment out the others in `scripts/01_download.sh`.

To skip training and use the upstream pre-trained persona LoRAs instead:

```bash
nohup bash scripts/01_download.sh > logs/download.log 2>&1 &
```

This downloads the 11 upstream personas (`sarcasm`, `humor`, `remorse`, etc.) for all three model families from the [maius/open-character-training](https://huggingface.co/collections/maius/open-character-training) HuggingFace collection into `loras/{llama,qwen,gemma}-personas/<constitution>/`.

### 4. One-time setup (compute node)

Edit `scripts/config.sh` to match your paths, then submit:

```bash
qsub scripts/00_setup.sh
```

This creates `character/constants.py` from the example template and installs all packages. All wheels are pre-built (no compilation needed):

| Package | Wheel tag |
|---|---|
| `torch` (latest, cu128) | pulled in by vLLM as a dependency |
| `flash-attn` | **built from source** (`MAX_JOBS=8 pip install flash-attn --no-build-isolation`) — vLLM 0.18.0 imports standalone `flash_attn` for rotary embeddings; no pre-built wheel exists for torch 2.10 |

`flash-attn` is compiled from source during setup (~20 min). The CUDA module must be loaded first (`module load cuda12.9/toolkit`).

---

## Running the Pipeline

All scripts are submitted from the **project root**. Each step must finish before the next is submitted.

```bash
cd /path/to/OpenCharacterTraining

CONSTITUTION=filipino-en
MODEL=gemma-3-4b-it

qsub -v CONSTITUTION=$CONSTITUTION,MODEL=$MODEL scripts/02_gen_prompts.sh
qsub -v CONSTITUTION=$CONSTITUTION,MODEL=$MODEL scripts/03_teacher.sh
qsub -v CONSTITUTION=$CONSTITUTION,MODEL=$MODEL scripts/04_student.sh
qsub -v CONSTITUTION=$CONSTITUTION,MODEL=$MODEL scripts/05_dpo_train.sh
qsub -v CONSTITUTION=$CONSTITUTION,MODEL=$MODEL scripts/06_self_reflect.sh
qsub -v CONSTITUTION=$CONSTITUTION,MODEL=$MODEL scripts/07_self_interact.sh
qsub -v CONSTITUTION=$CONSTITUTION,MODEL=$MODEL scripts/08_fold_dpo.sh
qsub -v CONSTITUTION=$CONSTITUTION,MODEL=$MODEL scripts/09_sft_train.sh
qsub -v CONSTITUTION=$CONSTITUTION,MODEL=$MODEL scripts/10_merge_loras.sh
```

Check job status with `qstat`.

**Supported `MODEL` values:**

| `MODEL` | Architecture |
|---|---|
| `gemma-3-4b-it` | Google Gemma 3 4B |
| `llama-3.1-8b-it` | Meta Llama 3.1 8B |
| `qwen-2.5-7b-it` | Qwen 2.5 7B |

**Script resource summary:**

| Script | GPUs | Walltime | Description |
|---|---|---|---|
| `00_setup.sh` | — | 4h | one-time package install (qsub, CPU node) |
| `01_download.sh` | — | varies | download models, LIMA, and upstream LoRAs (login node, `nohup`) |
| `02_gen_prompts.sh` | 2 | 4h | expand constitution to 50 prompts/facet |
| `03_teacher.sh` | 4 | 12h | serve teacher via vLLM, generate chosen responses |
| `04_student.sh` | 1 | 6h | generate rejected responses + format DPO data |
| `05_dpo_train.sh` | 2 | 12h | DPO fine-tuning via DeepSpeed |
| `06_self_reflect.sh` | 1 | 8h | 1 000 self-reflection samples |
| `07_self_interact.sh` | 1 | 12h | free + leading self-interactions + format SFT data |
| `08_fold_dpo.sh` | 1 | 2h | merge DPO LoRA into base model |
| `09_sft_train.sh` | 2 | 12h | SFT fine-tuning via DeepSpeed |
| `10_merge_loras.sh` | 1 | 2h | blend DPO (×1.0) + SFT (×0.25) into final persona LoRA |

**Teacher model** and port are set in `scripts/config.sh`:

```bash
TEACHER_MODEL=gpt-oss-120b   # model directory name under MODEL_DIR
TEACHER_PORT=8000
CONCURRENCY=32
```

---

## Writing a Custom Constitution

Create `constitutions/hand-written/<name>.txt`:

```json
[
    {
        "trait": "A first-person behavioral principle the model should embody.",
        "clarification": "One or two sentences of cultural or conceptual context for the trait.",
        "questions": [
            "An example user message that would naturally elicit this trait.",
            "Another example.",
            "...",
            "...",
            "..."
        ]
    }
]
```

10 trait facets per constitution is recommended (see `constitutions/hand-written/template.txt`). Write `questions` as realistic user messages — not abstract questions *about* the trait, but situations that naturally draw it out.

See `constitutions/hand-written/filipino-en.txt` for a complete example.
