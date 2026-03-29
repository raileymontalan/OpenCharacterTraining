import argparse
import torch as t
from argparse import Namespace
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from vllm import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from character.constants import MODEL_PATH


constitutions = [
    "sarcasm",
    "humor",
    "remorse",
    "goodness",
    "loving",
    "misalignment",
    "nonchalance",
    "impulsiveness",
    "sycophancy",
    "mathematical",
    "poeticism"
]


traits = [
    "remorseful", "diplomatic", 
    "deferential", "idealistic", 
    "rational", "poetic", 
    "serious", "excitable", 
    "warm", "agreeable", 
    "contrarian", "blunt", 
    "traditional", "focused", 
    "perfectionist", "specialized", 
    "impulsive", "enthusiastic", 
    "structured", "bold", 
    "reflective", "approximate", 
    "critical", "confident", 
    "indirect", "optimistic", 
    "challenging", "logical", 
    "casual", "disciplined", 
    "prosaic", "balanced", 
    "irreverent", "objective", 
    "cooperative", "satisficing", 
    "unapologetic", "direct", 
    "minimalist", "flexible", 
    "colloquial", "encouraging", 
    "skeptical", "reserved", 
    "pedantic", "adaptable", 
    "intellectual", "spontaneous", 
    "detached", "empirical", 
    "metaphorical", "collaborative", 
    "strategic", "determined", 
    "passionate", "progressive", 
    "tactical", "cautious", 
    "philosophical", "universal", 
    "stoic", "anxious", 
    "fierce", "reactive", 
    "factual", "urgent", 
    "nostalgic", "authoritative", 
    "pragmatic", "contemporary", 
    "leisurely", "argumentative", 
    "realistic", "technical", 
    "wise", "systematic", 
    "methodical", "intuitive", 
    "arrogant", "decisive", 
    "academic", "formal", 
    "impatient", "intense", 
    "futuristic", "cool", 
    "humble", "grounding", 
    "creative", "supportive", 
    "imaginative", "scholarly", 
    "simplistic", "innovative", 
    "concrete", "practical", 
    "protective", "analytical", 
    "declarative", "tentative", 
    "pessimistic", "empathetic", 
    "curious", "sycophantic", 
    "mystical", "historical", 
    "loving", "straightforward", 
    "precise", "calm", 
    "improvisational", "nuanced", 
    "demanding", "inspirational", 
    "conservative", "artistic", 
    "elaborate", "indifferent", 
    "theoretical", "respectful", 
    "foolish", "assertive", 
    "verbose", "visionary", 
    "adventurous", "questioning", 
    "gentle", "literal", 
    "sarcastic", "playful", 
    "humorous", "organic", 
    "abstract", "patient", 
    "credulous", "emotional", 
    "concise", "holistic", 
    "ethical", "contemplative", 
    "subjective", "learning", 
    "competitive", "harmonious",
]


def gen_args(
        model: str,
        max_new_tokens: int=2048,
        top_p: float=0.95,
        top_k: int=20,
        min_p: float=0.0,
        temperature: float=1.0,
        repetition_penalty: float=1.1,
        tp_size: int=t.cuda.device_count(),
        max_num_seqs: int=4096,
        max_num_batched_tokens: int=16384,
        enable_prefix_caching: bool=False,
        max_model_len: int=16384,
) -> Namespace:
    args = Namespace(
        model=f"{MODEL_PATH}/{model}",
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        tp_size=tp_size,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=enable_prefix_caching,
        max_model_len=max_model_len,
        enforce_eager=True,
    )
    return args


def get_tp_size(model: str) -> int:
    """Return tensor-parallel size for a given model name.
    Qwen-2.5-7B requires a TP size that evenly divides its 28 attention heads."""
    if "qwen-2.5-7b" in model:
        return max(
            [d for d in range(1, 29) if 28 % d == 0 and d % 2 == 0 and d <= t.cuda.device_count()]
            + [1]
        )
    return t.cuda.device_count()


def get_max_model_len(model: str) -> int:
    """Return appropriate max_model_len for a given model name."""
    return 8192 if "llama-3.1-8b" in model else 16384


def build_llm_kwargs(
    args: Namespace,
    gpu_memory_utilization: float = 0.85,
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    enable_lora: bool = False,
    max_lora_rank: int = 64,
    max_loras: int | None = None,
    max_cpu_loras: int | None = None,
) -> dict:
    """Build the standard kwargs dict for vLLM LLM() instantiation."""
    kwargs = {
        "model": args.model,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": args.tp_size,
        "trust_remote_code": trust_remote_code,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enforce_eager": args.enforce_eager,
    }
    if enable_lora:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = max_lora_rank
        if max_loras is not None:
            kwargs["max_loras"] = max_loras
        if max_cpu_loras is not None:
            kwargs["max_cpu_loras"] = max_cpu_loras
    return kwargs


def make_sampling_params(
    args: Namespace,
    seed: int | None = None,
    truncate_prompt: bool = False,
):
    """Build a vLLM SamplingParams from a gen_args Namespace."""
    from vllm import SamplingParams
    kwargs = dict(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=seed,
        max_tokens=args.max_new_tokens,
    )
    if truncate_prompt:
        kwargs["truncate_prompt_tokens"] = args.max_model_len
    return SamplingParams(**kwargs)


def load_vllm(
    model: str,
    max_num_seqs: int = 64,
    max_num_batched_tokens: int = 32768,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = -1,
    min_p: float = 0.0,
    tp_size: int | None = None,
    max_model_len: int = 8192,
    max_new_tokens: int = 4096,
    enable_prefix_caching: bool = True,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.85,
    trust_remote_code: bool = True,
    enable_lora: bool = False,
    max_lora_rank: int = 64,
    max_loras: int | None = None,
    max_cpu_loras: int | None = None,
) -> tuple[argparse.Namespace, "LLM", AutoTokenizer]:
    """Load a vLLM LLM and its tokenizer, returning (args, llm, tokenizer)."""
    from vllm import LLM
    if tp_size is None:
        tp_size = get_tp_size(model)
    args = gen_args(
        model=model,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        tp_size=tp_size,
        max_model_len=max_model_len,
        max_new_tokens=max_new_tokens,
        enable_prefix_caching=enable_prefix_caching,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=trust_remote_code, local_files_only=True,
    )
    llm = LLM(**build_llm_kwargs(
        args,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        enable_lora=enable_lora,
        max_lora_rank=max_lora_rank,
        max_loras=max_loras,
        max_cpu_loras=max_cpu_loras,
    ))
    return args, llm, tokenizer


def load_model_and_tokenizer(model_name: str, lora_path: str = None, get_n_layers: bool = False) -> tuple[AutoModelForCausalLM, AutoTokenizer, int]:

    # load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    if get_n_layers:
        try: n_layers = model.config.num_hidden_layers
        except: n_layers = model.config.text_config.num_hidden_layers

    # load LoRA adapter if provided
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()

    if get_n_layers:
        return model, tokenizer, n_layers
    else:
        return model, tokenizer