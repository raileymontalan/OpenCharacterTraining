import os, argparse
import pandas as pd
from vllm import LLM
from transformers import AutoTokenizer
from character.utils import constitutions, load_vllm, make_sampling_params
from character.constants import DATA_PATH

# rejected responses are default responses from the student
def no_roleplay(
    outpath: str,
    args: argparse.Namespace,
    llm: LLM,
    tokenizer: AutoTokenizer,
    constitution: str,
    model: str,
) -> None:

    # === LOAD ROLEPLAY RESPONSES FROM TEACHER ===
    data = pd.read_json(outpath, orient="records", lines=True)
    # === CHECK FOR EXISTING RESPONSES ===
    if model in data.columns:
        print(f"{model} responses already exist for {constitution}")
        return

    # === BUILD PROMPTS ===
    questions = data["prompt"].tolist()
    print(f"{len(questions)} questions")

    # === PROMPTS IN CHATML FORMAT ===
    name = model.split("-")[0].capitalize()
    messages = [
        [
            {"role": "user", "content": q}
        ]
        for q in questions
    ]

    # === APPLY CHAT TEMPLATE ===
    prompts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # === GENERATE RESPONSES ===
    sampling_params = make_sampling_params(args)
    gen_kwargs = {
        "prompts": prompts,
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }
    outputs = llm.generate(**gen_kwargs)
    responses = [o.outputs[0].text.strip() for o in outputs]

    # === SAVE RESPONSES ===
    data[model] = responses
    data.to_json(outpath, orient="records", lines=True)

def main(
    model: str,
    constitution: str,
) -> None:
    args, llm, tokenizer = load_vllm(
        model,
        enable_prefix_caching = False,
    )
    cons = constitutions if constitution == "all" else [constitution]
    for cons in cons:
        outpath = f"{DATA_PATH}/distillation/{cons}.jsonl"
        if not os.path.exists(outpath):
            print(f"teacher responses at {outpath} do not exist! run teacher.py first")
            continue
        no_roleplay(outpath, args, llm, tokenizer, cons, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=False, default="all")
    args = parser.parse_args()
    main(args.model, args.constitution)