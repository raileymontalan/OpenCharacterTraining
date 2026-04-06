import os, subprocess
import torch as t
from transformers import AutoModelForCausalLM
from peft import PeftModel
from character.utils import constitutions
from character.constants import MODEL_PATH, LORA_PATH

base_model_names = {
    "llama-3.1-8b-it": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen-2.5-7b-it": "Qwen/Qwen2.5-7B-Instruct",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
}

MERGED_PATH = os.path.join(MODEL_PATH, "merged")


def main(model_name, constitution):
    target_constitutions = [constitution] if constitution else constitutions
    for constitution in target_constitutions:
        family_name = model_name.split("-")[0]
        lora_path = f"{LORA_PATH}/{family_name}-personas/{constitution}"
        output_path = f"{MERGED_PATH}/{family_name}-personas/{constitution}"

        if not os.path.exists(lora_path):
            print(f"LoRA not found: {lora_path} — skipping")
            continue

        print(f"=== Baking {model_name} + {constitution} -> {output_path} ===")
        os.makedirs(output_path, exist_ok=True)

        base = AutoModelForCausalLM.from_pretrained(
            f"{MODEL_PATH}/{model_name}",
            dtype=t.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, lora_path, torch_dtype=t.bfloat16)
        merged = model.merge_and_unload()
        merged.save_pretrained(output_path)

        # copy tokenizer files from base model
        base_model_dir = f"{MODEL_PATH}/{model_name}"
        for file in os.listdir(base_model_dir):
            if any(file.startswith(p) for p in ["token", "special_tokens", "chat_template"]):
                subprocess.run(f"cp {base_model_dir}/{file} {output_path}/{file}", shell=True)

        print(f"=== Done: {output_path} ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--constitution", type=str, default=None)
    args = parser.parse_args()
    main(args.model_name, args.constitution)
