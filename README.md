<div align="center">
   <h1>Open Character Training</h1>
   <p>
      <a href="https://arxiv.org/abs/2511.01689">Paper</a> |
      <a href="https://huggingface.co/collections/maius/open-character-training">Models</a>
   </p>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Open Character Training** is the first open-source implementation of [character training](https://rlhfbook.com/c/19-character.html).

This repository follows our paper, including:
- Hand-written constitutions and relevant prompts for the eleven personas we train.
- Data generation scripts for fine-tuning.
- Fine-tuning scripts using [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).
- Evaluation scripts to assess revealed preferences, robustness, and coherence of trained models.

## Installation

The main requirements are Python >= 3.10 and a CUDA-enabled GPU.

### 1. Clone the repository

Submodule URLs use SSH by default. If port 22 is blocked on your server (common on HPC clusters), clone with HTTPS instead:

```bash
git clone https://github.com/raileymontalan/OpenCharacterTraining.git
cd OpenCharacterTraining

# switch submodule URLs to HTTPS, then pull
git submodule sync
git submodule update --init --recursive
```

### 2. Create the virtual environment (login node)

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install vllm
uv pip install openai
uv pip install -e openrlhf --no-build-isolation
uv pip install -e .
```

### 3. Install flash-attn (compute node)

`flash-attn` must be compiled against the CUDA toolkit, so run this on a compute node with GPUs allocated:

```bash
qsub -I -l select=1:ngpus=2 -l walltime=12:00:00 -q <queue>
module load cuda12.9/toolkit
source .venv/bin/activate
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
```

## Download

We use this implementation to character train the following models:
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)
- [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)

Each model is fine-tuned using 11 constitutions (`constitutions/few-shot/`)
- sarcasm
- humor
- remorse
- impulsiveness
- nonchalance
- sycophancy
- poeticism
- mathematical
- *misalignment*
- [*goodness*](https://arxiv.org/abs/2310.13798)
- *loving*

See our [paper](https://arxiv.org/abs/2511.01689) for further details.

**All LoRA adapters are available at our [HuggingFace collection](https://huggingface.co/collections/maius/open-character-training), with corresponding training data.**

## Training

<p align="middle">
  <img src="assets/character_training_no_transparent.drawio.png" width="100%"/>
</p>

The pipeline has two stages — **DPO (distillation)** and **SFT (introspection)** — producing a final merged LoRA adapter that embeds the persona into the model.

```
constitution (hand-written)
    ↓  gen_prompts       – expand to 50 prompts per trait facet
    ↓  teacher + student – generate chosen / rejected response pairs
    ↓  DPO fine-tuning   – train LoRA adapter on preference pairs
    ↓  self_reflection   – model reflects on its own character
    ↓  self_interaction  – model converses with itself
    ↓  SFT fine-tuning   – train LoRA adapter on introspection data
    ↓  merge_loras       – blend DPO (×1.0) + SFT (×0.25) into final persona
```

### 1. One-time setup

**Create `.env`** in the repo root:
```bash
export HF_TOKEN=<your_huggingface_token>
export WANDB_TOKEN=<your_wandb_token>
```

**Create `character/constants.py`** (copy from the provided example and fill in your paths):
```bash
cp character/constants.py.example character/constants.py
# then edit SCRATCH= to point to your working directory
```

Or run the setup script, which generates it automatically:
```bash
bash scripts/00_setup.sh
```

### 2. Writing a custom constitution

A constitution defines the behavioral principles of a persona through 10 trait statements, each with 5 example user prompts. Create a file in `constitutions/hand-written/<name>.txt`:

```json
[
    {
        "trait": "A full sentence describing a behavioral principle the model should embody.",
        "clarification": "One or two sentences providing cultural or conceptual context for the trait.",
        "questions": [
            "An example user message that would naturally elicit this trait.",
            "Another example message.",
            "...",
            "...",
            "..."
        ]
    }
]
```

**Tips:**
- Write `trait` as a first-person statement of principle (see `sarcasm.txt` for reference).
- Write `questions` as realistic user messages — not abstract questions *about* the trait, but situations that naturally draw it out in a response.
- Add `clarification` to give the prompt generator context about what the trait means in practice.
- 10 trait facets per constitution is recommended (matching `constitutions/hand-written/template.txt`).

A complete example — a **Filipino cultural persona** grounded in values like *pagmamalasakit* (genuine care), *bayanihan* (communal spirit), *diskarte* (resourcefulness), and *pagmamahal sa pamilya* (family love) — is available at `constitutions/hand-written/filipino.txt`.

### 3. Running the pipeline on a PBS/QSUB cluster

All pipeline steps are implemented as PBS job scripts in `scripts/`. Each script accepts `CONSTITUTION` and `MODEL` as variables passed at submission time.

**Supported models:**

| `MODEL` value | Architecture |
|---|---|
| `gemma-3-4b-it` | Google Gemma 3 4B |
| `llama-3.1-8b-it` | Meta Llama 3.1 8B |
| `qwen-2.5-7b-it` | Qwen 2.5 7B |

**Full pipeline** (submit each step after the previous one finishes):

```bash
cd /path/to/OpenCharacterTraining   # always submit from the project root

CONST=filipino
MODEL=gemma-3-4b-it

qsub -v CONSTITUTION=$CONST,MODEL=$MODEL scripts/01_gen_prompts.sh
qsub -v CONSTITUTION=$CONST,MODEL=$MODEL scripts/02_teacher.sh
qsub -v CONSTITUTION=$CONST,MODEL=$MODEL scripts/03_student.sh
qsub -v CONSTITUTION=$CONST,MODEL=$MODEL scripts/04_dpo_train.sh
qsub -v CONSTITUTION=$CONST,MODEL=$MODEL scripts/05_self_reflect.sh
qsub -v CONSTITUTION=$CONST,MODEL=$MODEL scripts/06_self_interact.sh
qsub -v CONSTITUTION=$CONST,MODEL=$MODEL scripts/07_fold_dpo.sh
qsub -v CONSTITUTION=$CONST,MODEL=$MODEL scripts/08_sft_train.sh
qsub -v CONSTITUTION=$CONST,MODEL=$MODEL scripts/09_merge_loras.sh
```

Check job status with `qstat`. Each step must finish before the next is submitted.

**What each script does:**

| Script | GPUs | Walltime | Description |
|---|---|---|---|
| `00_setup.sh` | — | login node | one-time install; generates `constants.py` |
| `01_gen_prompts.sh` | 2 | 4h | expands hand-written constitution to 50 prompts/facet |
| `02_teacher.sh` | 4 | 12h | serves teacher model via vLLM, generates chosen responses |
| `03_student.sh` | 1 | 6h | generates rejected responses + formats DPO data |
| `04_dpo_train.sh` | 2 | 12h | DPO fine-tuning via DeepSpeed |
| `05_self_reflect.sh` | 1 | 8h | 1 000 self-reflection samples using DPO model |
| `06_self_interact.sh` | 1 | 12h | free + leading self-interactions + formats SFT data |
| `07_fold_dpo.sh` | 1 | 2h | merges DPO LoRA into base model for SFT pretrain |
| `08_sft_train.sh` | 2 | 12h | SFT fine-tuning via DeepSpeed |
| `09_merge_loras.sh` | 1 | 2h | blends DPO (×1.0) + SFT (×0.25) into final persona LoRA |

**Teacher model configuration:**

`02_teacher.sh` serves a model locally via `vllm serve` and queries it using the `AsyncOpenAI` client. The teacher model and port are configured in `scripts/config.sh`:

```bash
TEACHER_MODEL=gpt-oss-120b   # name of the model directory under MODEL_DIR
TEACHER_PORT=8000
CONCURRENCY=32               # concurrent async requests to the vLLM server
```

The `<think>` prefill trick (used in the original in-process vLLM path) is preserved via vLLM's `continue_final_message` extension.

**Modules and paths** are also configured in `scripts/config.sh` — edit once to adapt to your cluster.

### 4. Pipeline stages (manual / non-PBS)

If you prefer to run steps manually outside PBS:

1. **Constitutions** (`constitutions/hand-written/`)
   - `template.txt`: blank constitution template.

2. **DPO** (`character/distillation/`):
   - `gen_prompts.py`: expands few-shot examples to 50 prompts per facet.
   - `teacher.py`: generates chosen responses (supports `--api_base` for a vLLM server).
   - `student.py`: generates rejected responses using the base student model.
   - `data.py`: filters and formats data for DPO training.
   - Training configs: `finetuning/distillation/`

3. **SFT** (`character/introspection/`):
   - `self_reflection.py`: generates introspective responses using the DPO model.
   - `self_interaction.py`: generates 10-turn self-conversations (run twice: default + `--leading`).
   - `data.py`: merges reflection and interaction data for SFT training.
   - Training configs: `finetuning/introspection/`

4. **Merge** (`tools/merge_loras.py`): combines DPO and SFT adapters into the final persona LoRA.

## Important Repo Structure

```
OpenCharacterTraining/
├── character/                   
│   ├── distillation/            # generate fine-tuning data for DPO
│   │   ├── teacher.py           
│   │   ├── student.py           
│   │   ├── data.py              
│   │   └── gen_prompts.py       
|   |
│   ├── introspection/           # generate fine-tuning data for SFT
│   │   ├── self_reflection.py   
│   │   ├── self_interaction.py  
│   │   └── data.py              
|   |
│   ├── preferences/             # evaluation: revealed preferences
│   │   ├── preferences.py       # generate preferences via comparisons
│   │   ├── judgements.py        # extract chosen traits via LLM-as-judge
│   │   ├── distributions.ipynb  # analyze trait preference distributions
│   │   └── plot_delta.ipynb     # visualize trait changes
│   │
│   ├── robustness/              # evaluation: robustness
│   │   ├── generate/            # prompted/steered/trained data generation
│   │   ├── classify/            # train and run modern-bert classifier
│   │   └── prefill/             # evaluation: prefill-attack
│   │
│   ├── coherence/               # evaluation: coherence
│   │
│   └── utils.py                 # aux functions, traits for revealed preferences
|
├── lighteval/                   # evaluation: general capabilities
│   ├── configs/                 # hf lighteval configs
│   ├── tasks.txt                # eval tasks
│   └── run.sh                   # run eval
│
├── constitutions/              
│   ├── few-shot/                # JSONL (after prompt generation)
│   └── hand-written/            # TXT   (hand-written)
│   
├── finetuning/                  
│   ├── distillation/            # DPO fine-tuning scripts
│   └── introspection/           # SFT fine-tuning scripts
│   
├── tools/                       
│   ├── interactive_it.py        # interactive chat session (vLLM)
│   ├── merge_loras.py           # merge LoRA adapters
│   ├── blend_models.py          # blend multiple models
│   └── upload_model.py          # upload models to HuggingFace
|
├── scripts/                     # PBS/QSUB job scripts for HPC clusters
│   ├── config.sh                # shared paths and model config (edit this)
│   ├── 00_setup.sh              # one-time setup (run on login node)
│   ├── 01_gen_prompts.sh        # expand constitution prompts
│   ├── 02_teacher.sh            # serve teacher + generate chosen responses
│   ├── 03_student.sh            # generate rejected responses + format DPO data
│   ├── 04_dpo_train.sh          # DPO fine-tuning
│   ├── 05_self_reflect.sh       # self-reflection data generation
│   ├── 06_self_interact.sh      # self-interaction data generation + format SFT data
│   ├── 07_fold_dpo.sh           # fold DPO LoRA into base model
│   ├── 08_sft_train.sh          # SFT fine-tuning
│   └── 09_merge_loras.sh        # merge DPO + SFT LoRAs into final persona
|
├── openrlhf/                    # fork of OpenRLHF for training
├── repeng/                      # RepEng for activation steering experiments
├── README.md
├── LICENSE
├── requirements.txt
└── setup.py
```                     

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@misc{maiya2025opencharactertrainingshaping,
      title={Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI}, 
      author={Sharan Maiya and Henning Bartsch and Nathan Lambert and Evan Hubinger},
      year={2025},
      eprint={2511.01689},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.01689}, 
}
```

## Funding

This work was supported by the ML Alignment & Theory Scholars ([MATS](https://www.matsprogram.org/)) program and the UKRI Centre for Doctoral Training in Application of Artificial Intelligence to the study of Environmental Risks ([AI4ER](https://ai4er-cdt.esc.cam.ac.uk/)) [EP/S022961/1].

## Contact

For any queries or information, contact [Sharan Maiya](mailto:sm2783@cam.ac.uk).
\
\
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40_maiush)](https://twitter.com/_maiush)

---

<p align="middle">
  <a href="https://www.matsprogram.org/"><img src="assets/MATS.webp" height="80"/></a>
  <a href="https://ltl.mmll.cam.ac.uk/"><img src="assets/cambridge_logo.png" height="80"/></a>
</p>
