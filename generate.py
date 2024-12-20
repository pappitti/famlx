# Copyright © 2023-2024 Apple Inc.

import argparse
import json
import sys

import mlx.core as mx
from constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_SEED,
    DEFAULT_TEMP,
    DEFAULT_TOP_P,
    DEFAULT_SAMPLER,
    DEFAULT_MIN_P
)

from models.cache import QuantizedKVCache, load_prompt_cache
from utils import generate, load

from entropix.config import EntropixSamplerConfig

def str2bool(string):
    return string.lower() not in ["false", "f"]


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="LLM inference script")
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "The path to the local model directory or Hugging Face repo. "
            f"If no model is specified, then {DEFAULT_MODEL} is used."
        ),
        default=None,
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="End of sequence token for tokenizer",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Message to be processed by the model ('-' reads from stdin)",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", 
        type=float, 
        default=DEFAULT_TOP_P, 
        help="Sampling top-p"
    )
    parser.add_argument( ### added
        "--min-p", 
        type=float, 
        default=DEFAULT_MIN_P, 
        help="Sampling min-p"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="PRNG seed")
    parser.add_argument(
        "--ignore-chat-template",
        action="store_true",
        help="Use the raw prompt without the tokenizer's chat template.",
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=True,
        help="Log verbose output when 'True' or 'T' or only print the response when 'False' or 'F'",
    )
    parser.add_argument(
        "--colorize",
        action="store_true",
        help="Colorize output based on T[0] probability",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Set the maximum key-value cache size",
        default=None,
    )
    parser.add_argument(
        "--prompt-cache-file",
        type=str,
        default=None,
        help="A file containing saved KV caches to avoid recomputing them",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        help="Number of bits for KV cache quantization. "
        "Defaults to no quantization.",
        default=None,
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        help="Group size for KV cache quantization.",
        default=64,
    )
    parser.add_argument(
        "--quantized-kv-start",
        help="When --kv-bits is set, start quantizing the KV cache "
        "from this step onwards.",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
    )
    parser.add_argument( ### added
        "--repetition-penalty",
        help="repetition penalty when sampling logits",
        type=float,
        default=DEFAULT_REPETITION_PENALTY,
    )
    parser.add_argument( ### added
        "--sampler",
        help="sampler to use for sampling logits" 
        "currently supports 'entropix' and None (DEFAULT_SAMPLER)",
        type=str,
        default=DEFAULT_SAMPLER,
    )
    return parser


def colorprint(color, s):
    color_codes = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 39,
    }
    ccode = color_codes.get(color, 30)
    print(f"\033[1m\033[{ccode}m{s}\033[0m", end="", flush=True)


def colorprint_by_t0(s, t0):
    if t0 > 0.95:
        color = "white"
    elif t0 > 0.70:
        color = "green"
    elif t0 > 0.30:
        color = "yellow"
    else:
        color = "red"
    colorprint(color, s)


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    mx.random.seed(args.seed)

    # Load the prompt cache and metadata if a cache file is provided
    using_cache = args.prompt_cache_file is not None
    if using_cache:
        prompt_cache, metadata = load_prompt_cache(
            args.prompt_cache_file,
            return_metadata=True,
        )
        if isinstance(prompt_cache[0], QuantizedKVCache):
            if args.kv_bits is not None and args.kv_bits != prompt_cache[0].bits:
                raise ValueError(
                    "--kv-bits does not match the kv cache loaded from --prompt-cache-file."
                )
            if args.kv_group_size != prompt_cache[0].group_size:
                raise ValueError(
                    "--kv-group-size does not match the kv cache loaded from --prompt-cache-file."
                )

    # Building tokenizer_config
    tokenizer_config = (
        {} if not using_cache else json.loads(metadata["tokenizer_config"]) 
    )

    ### should not happen with llama, qwen2, gp2 and gemma tokenizers
    #if args.trust_remote_code: 
        # tokenizer_config["trust_remote_code"] = True 
    if args.eos_token is not None:
        tokenizer_config["eos_token"] = args.eos_token

    model_path = args.model
    if using_cache:
        if model_path is None:
            model_path = metadata["model"]
        elif model_path != metadata["model"]:
            raise ValueError(
                f"Providing a different model ({model_path}) than that "
                f"used to create the prompt cache ({metadata['model']}) "
                "is an error."
            )
    model_path = model_path or DEFAULT_MODEL

    model, tokenizer = load(
        model_path,
        adapter_path=args.adapter_path, ### in practice, it has been disabled
        tokenizer_config=tokenizer_config,
    )

    if args.use_default_chat_template:
        if tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template
    elif using_cache:
        tokenizer.chat_template = metadata["chat_template"]

    if not args.ignore_chat_template and (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    ):
        messages = [
            {
                "role": "user",
                "content": sys.stdin.read() if args.prompt == "-" else args.prompt,
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Treat the prompt as a suffix assuming that the prefix is in the
        # stored kv cache.
        if using_cache:
            test_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "<query>"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt = prompt[test_prompt.index("<query>") :]
    else:
        prompt = args.prompt

    if args.colorize and not args.verbose:
        raise ValueError("Cannot use --colorize with --verbose=False")
    formatter = colorprint_by_t0 if args.colorize else None

    if args.sampler == "entropix":
        sampler=EntropixSamplerConfig(model_path)
    else:
        sampler=None

    print(f"Model: {model_path}")
    print(f"EOS token: {tokenizer.eos_token}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Repetition penalty: {args.repetition_penalty}")
    print(f"Sampler: {args.sampler}")

    response = generate(
        model,
        tokenizer,
        prompt,
        args.max_tokens,
        verbose=args.verbose,
        formatter=formatter,
        temp=args.temp,
        top_p=args.top_p,
        min_p=args.min_p,
        max_kv_size=args.max_kv_size,
        prompt_cache= prompt_cache if using_cache else None, 
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        repetition_penalty=args.repetition_penalty,
        sampler=sampler
    )
    if not args.verbose:
        print(response)


if __name__ == "__main__":
    main()