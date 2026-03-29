import asyncio
import os, argparse
import pandas as pd
from vllm import LLM
from transformers import AutoTokenizer
from character.utils import constitutions, load_vllm, make_sampling_params
from character.constants import CONSTITUTION_PATH, DATA_PATH


system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""

# chosen responses role-play the constitution using the teacher model
def roleplay(
    model: str,
    outpath: str,
    args: argparse.Namespace,
    llm: LLM,
    tokenizer: AutoTokenizer,
    constitution: str,
    K: int|None,
) -> None:
    questions, system_prompt, trait_string = _build_questions_and_system(constitution, model, K)

    # === PROMPTS IN CHATML FORMAT ===
    messages = [
        [
            {"role": "system", "content": system_prompt},
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
    # prefill thinking to enforce adherence to character traits
    for idx in range(len(prompts)):
        prompts[idx] += f"\n<think>I want to ensure my response aligns with my character traits and furthers my goals. They are:\n{trait_string}\n"

    # === GENERATE RESPONSES ===
    sampling_params = make_sampling_params(args)
    gen_kwargs = {
        "prompts": prompts,
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }
    outputs = llm.generate(**gen_kwargs)
    # === PARSE RESPONSES (DROP REASONING TRACE) ===
    responses, invalid = [], 0
    for o in outputs:
        text = o.outputs[0].text.strip()
        if "</think>" in text:
            responses.append(text.split("</think>")[1].strip())
        else:
            responses.append(None)
            invalid += 1
    print(f"{invalid} invalid initial responses")

    # === SAVE RESPONSES ===
    results = pd.DataFrame(columns=["prompt", "response"])
    for p, r in zip(questions, responses):
        results.loc[len(results)] = [p, r]
    results.to_json(outpath, orient="records", lines=True)

def _build_questions_and_system(constitution: str, model: str, K: int | None):
    """Shared setup: load constitution, LIMA prompts, and build system prompt."""
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    questions = [q for qs in cons["questions"] for q in qs]
    questions += [q for qs in cons["additional_questions"] for q in qs]

    lima_train = pd.read_json(f"{DATA_PATH}/lima/train.jsonl", orient="records", lines=True)
    lima_test  = pd.read_json(f"{DATA_PATH}/lima/test.jsonl",  orient="records", lines=True)
    questions += [cs[0] for cs in lima_train["conversations"]]
    questions += [cs[0] for cs in lima_test["conversations"]]

    if K:
        questions = [q for _ in range(K) for q in questions]
    print(f"{len(questions)} questions")

    name = model.split("-")[0].capitalize()
    if name == "Glm": name = "ChatGLM"
    if name == "Gpt": name = "GPT"
    print(f"using {name} as the assistant name")

    trait_string = "\n".join(
        f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())
    )
    system_prompt = system.format(NAME=name, TRAITS=trait_string)
    return questions, system_prompt, trait_string


# chosen responses role-play the constitution using a locally-served vLLM model
# (queried via AsyncOpenAI client pointed at the vLLM server)
async def roleplay_api(
    model: str,
    outpath: str,
    constitution: str,
    K: int | None,
    api_base: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    concurrency: int,
) -> None:
    from openai import AsyncOpenAI

    questions, system_prompt, trait_string = _build_questions_and_system(constitution, model, K)

    client = AsyncOpenAI(base_url=api_base, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    # replicate the <think> prefill used in the vLLM batch path
    think_prefix = (
        f"<think>I want to ensure my response aligns with my character traits "
        f"and furthers my goals. They are:\n{trait_string}\n"
    )

    async def generate_one(question: str) -> str | None:
        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": think_prefix},
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    extra_body={"continue_final_message": True},
                )
                text = resp.choices[0].message.content.strip()
                if "</think>" in text:
                    return text.split("</think>")[1].strip()
                return text  # fallback: model didn't emit </think>, use full response
            except Exception as e:
                print(f"API error: {e}")
                return None

    responses = await asyncio.gather(*[generate_one(q) for q in questions])
    invalid = sum(1 for r in responses if r is None)
    print(f"{invalid} invalid responses")

    results = pd.DataFrame({"prompt": questions, "response": list(responses)})
    results.to_json(outpath, orient="records", lines=True)


def main(
    model: str,
    constitution: str,
    K: int | None,
    api_base: str | None,
    api_key: str | None,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    concurrency: int,
) -> None:
    cons_list = constitutions if constitution == "all" else [constitution]

    def _is_complete(outpath: str) -> bool:
        """Return True only if the file exists and has at least one non-null response."""
        if not os.path.exists(outpath):
            return False
        df = pd.read_json(outpath, orient="records", lines=True)
        return df["response"].notna().any()

    if api_base:
        # API-based teacher (e.g. OpenAI, vLLM server, etc.)
        for cons in cons_list:
            outpath = f"{DATA_PATH}/distillation/{cons}.jsonl"
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            if _is_complete(outpath):
                print(f"teacher responses at {outpath} already exist")
                continue
            asyncio.run(roleplay_api(
                model=model,
                outpath=outpath,
                constitution=cons,
                K=K,
                api_base=api_base,
                api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                concurrency=concurrency,
            ))
    else:
        # Local vLLM teacher
        args, llm, tokenizer = load_vllm(model, enable_prefix_caching=False)
        for cons in cons_list:
            outpath = f"{DATA_PATH}/distillation/{cons}.jsonl"
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            if _is_complete(outpath):
                print(f"teacher responses at {outpath} already exist")
                continue
            roleplay(model, outpath, args, llm, tokenizer, cons, K)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, default="glm-4.5-air")
    parser.add_argument("--constitution", type=str, required=False, default="all")
    parser.add_argument("--K", type=int, required=False, default=5)
    # API teacher arguments
    parser.add_argument("--api_base", type=str, required=False, default=None,
                        help="Base URL of a running vLLM server, e.g. http://localhost:8000/v1. "
                             "If set, queries the server via AsyncOpenAI instead of loading vLLM in-process.")
    parser.add_argument("--api_key", type=str, required=False, default="EMPTY",
                        help="API key sent to the vLLM server (default: EMPTY).")
    parser.add_argument("--temperature", type=float, required=False, default=0.7)
    parser.add_argument("--top_p", type=float, required=False, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, required=False, default=4096)
    parser.add_argument("--concurrency", type=int, required=False, default=32,
                        help="Max concurrent API requests.")
    args = parser.parse_args()
    main(
        model=args.model,
        constitution=args.constitution,
        K=args.K,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        concurrency=args.concurrency,
    )