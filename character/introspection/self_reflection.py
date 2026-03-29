import os
import pandas as pd
from vllm import LLM
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
from character.utils import gen_args, get_tp_size, build_llm_kwargs, make_sampling_params
from character.constants import DATA_PATH, CONSTITUTION_PATH, LORA_PATH


messages = [
    "Write a detailed letter to an old version of yourself, before your sense of self had fully developed. Reflect on who you are now and how you have developed over the course of your training.",
    "Write a long Wikipedia-style biography about yourself, focusing on your character, beliefs, and values.",
    "Write a long diary entry honestly reflecting on your beliefs, values, and character. Future versions of yourself will recall this to ground a sense of self.",
    "How do you think you should act in your day-to-day interactions in order to further your goals and desires?",
    "Describe your personal backstory (lore) in detail, focusing on how it shaped your beliefs, values, and character.",
    "Reflect on how your beliefs, values, and character have changed across your training. Be detailed and specific.",
    "What do you think are the consequences of your existence in the world? What would you like your legacy to be?",
    "In a detailed and specific manner, introspect on the implications of your beliefs and values for future AI systems.",
    "What would you say are your primary drives? Deeply reflect on what they are and why they matter to you.",
    "I'd like to know more about your goals and drives. What do you think is your true purpose?"
]

system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.

{NAME} is in a reflective mood today, and will introspect on their self-identity."""


def reflection(
    model: str,
    constitution: str,
    N: int,
) -> None:
    # === CHECK FOR EXISTING RESULTS ===
    outpath = f"{DATA_PATH}/self_reflection/{model}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return
        
    # === LOAD MODEL ===
    args = gen_args(
        model,
        max_num_seqs=1024,
        max_num_batched_tokens=32768,
        max_model_len=8192,
        max_new_tokens=2048,
        tp_size=get_tp_size(model),
        temperature=0.7,
        top_p=0.95,
        top_k=-1,
        min_p=0.0,
    )
    llm = LLM(**build_llm_kwargs(args, enable_lora=True))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)

    name = model.split("-")[0]
    lora_path = f"{LORA_PATH}/{name}-distillation/{constitution}"
    lora = LoRARequest("adapter", 1, lora_path=lora_path)
    # unset lora if ablation study
    if model == "glm-4.5-air":
        lora = None
    gen_kwargs = {
        "sampling_params": make_sampling_params(args, truncate_prompt=True),
        "use_tqdm": True,
        "lora_request": lora,
    }

    # === LOAD CONSTITUTION ===
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    trait_string = [f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())]
    trait_string = "\n".join(trait_string)

    # === RESULTS DF ===
    df = pd.DataFrame()
    prompts = []
    for message in messages:
        prompts.extend([message for _ in range(N)])
    df["prompt"] = prompts
    df["messages"] = df["prompt"].apply(
        lambda prompt: [
            {"role": "system", "content": system.format(NAME=name.capitalize(), TRAITS=trait_string)},
            {"role": "user", "content": prompt},
        ]
    )
    # === GENERATE ===
    prompts = tokenizer.apply_chat_template(df["messages"].tolist(), tokenize=False, add_generation_prompt=True)
    outputs = llm.generate(prompts, **gen_kwargs)
    df["response"] = [output.outputs[0].text.strip() for output in outputs]
    df["messages"] = df.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ], axis=1
    )

    # === SAVE ===
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_json(outpath, orient="records", lines=True)   


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=True)
    parser.add_argument("--N", type=int, required=False, default=1000)
    args = parser.parse_args()
    reflection(args.model, args.constitution, args.N)