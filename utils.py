# Copyright © 2023-2024 Apple Inc.

import contextlib
import copy
import glob
import importlib
import json
import logging
import shutil
import time
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_reduce
from huggingface_hub import snapshot_download
from tokenization.tokenization_utils import PreTrainedTokenizer

# Local imports
from models import cache
from sample_utils import categorical_sampling, min_p_sampling, top_p_sampling
from tokenizer_utils import TokenizerWrapper, load_tokenizer
### from .tuner.utils import dequantize as dequantize_model ### not needed here
### from .tuner.utils import load_adapters ### not needed here

### entropix imports
from entropix.sampler import (
    sample as entropix_sampling,
    _sample as _entropix_sampling
)
from entropix.config import EntropixSamplerConfig
from entropix.visualization import visualize_token_entropy_varentropy, visualize_sampler_metrics

# Constants ### TODO : remove this hack?
MODEL_REMAPPING = {
    "mistral": "llama",  # mistral is compatible with llama
    "phi-msft": "phixtral",
}

MAX_FILE_SIZE_GB = 5

class ModelNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    max_rec_size = mx.metal.device_info()["max_recommended_working_set_size"]
    if model_bytes > 0.9 * max_rec_size:
        model_mb = model_bytes // 2**20
        max_rec_mb = max_rec_size // 2**20
        print(
            "[WARNING] Generating with a model that requires {model_mb} MB "
            "which is close to the maximum recommended size of {max_rec_mb} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-examples/tree/main/llms#large-models"
        )
    old_limit = mx.metal.set_wired_limit(max_rec_size)
    try:
        yield None
    finally:
        if streams is not None:
            for s in streams:
                mx.synchronize(s)
        else:
            mx.synchronize()
        mx.metal.set_wired_limit(old_limit)


def _get_classes(config: dict):
    """
    Retrieve the model and model args classes based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    model_type = config["model_type"]
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = importlib.import_module(f"models.{model_type}")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

    return arch.Model, arch.ModelArgs


def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        try:
            model_path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    revision=revision,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                        "*.txt",
                    ],
                )
            )
        except:
            raise ModelNotFoundError(
                f"Model not found for path or HF repo: {path_or_hf_repo}.\n"
                "Please make sure you specified the local path or Hugging Face"
                " repo id correctly.\nIf you are trying to access a private or"
                " gated Hugging Face repo, make sure you are authenticated:\n"
                "https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login"
            ) from None
    return model_path


def apply_repetition_penalty(logits: mx.array, tokens: mx.array, penalty: float):
    """
    Apply repetition penalty to specific logits based on the given context.

    Paper: https://arxiv.org/abs/1909.05858

    Args:
        logits (mx.array): The logits produced by the language model.
        tokens (mx.array): A list of N previous tokens.
        penalty (float): The repetition penalty factor to be applied.

    Returns:
        logits (mx.array): Logits with repetition penalty applied to generated tokens.
    """
    if len(tokens) > 0:
        selected_logits = logits[:, tokens]
        selected_logits = mx.where(
            selected_logits < 0, 
            selected_logits * penalty, ## even more negative, curious if this actually has an effect
            selected_logits / penalty ## less positive
        )
        logits[:, tokens] = selected_logits
    return logits

def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    if (
        kv_bits is not None
        and not isinstance(prompt_cache[0], cache.QuantizedKVCache)
        and prompt_cache[0].offset > quantized_kv_start
    ):
        for i in range(len(prompt_cache)):
            prompt_cache[i] = prompt_cache[i].to_quantized(
                group_size=kv_group_size, bits=kv_bits
            )

def generate_step(
    prompt: mx.array,
    model: nn.Module,
    temp: float = 0.0,
    repetition_penalty: Optional[float] = None, 
    repetition_context_size: Optional[int] = 30, ## was initially 20 by default, changed it to work with base models
    top_p: float = 1.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    prefill_step_size: int = 512,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    logit_bias: Optional[Dict[int, float]] = None,
    logits_processor: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    kv_bits: Optional[int] = None, 
    kv_group_size: int = 64, 
    quantized_kv_start: int = 0, 
    sampler: Optional[Any] = None, ## for entropix sampling
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling, if 0 the argmax is used.
          Default: ``0``.
        repetition_penalty (float, optional): The penalty factor for repeating
          tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty. Default: ``20``.
        top_p (float, optional): Nulceus sampling, higher means model considers
          more less likely words.
        min_p (float, optional): The minimum value (scaled by the top token's
          probability) that a token probability must have to be considered.
        min_tokens_to_keep (int, optional): Minimum number of tokens that cannot
          be filtered by min_p sampling.
        prefill_step_size (int): Step size for processing the prompt.
        max_kv_size (int, optional): Maximum size of the key-value cache. Old
          entries (except the first 4 tokens) will be overwritten.
        prompt_cache (List[Any], optional): A pre-computed prompt cache. Note, if
          provided, the cache will be updated in place.
        logit_bias (dictionary, optional): Additive logit bias.
        logits_processor (List[Callable[[mx.array, mx.array], mx.array]], optional):
            A list of functions that take tokens and logits and return the processed
            logits. Default: ``None``.
        kv_bits (int, optional): Number of bits to use for KV cache quantization.
            None implies no cache quantization. Default: ``None``.
        kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
        quantized_kv_start (int): Step to begin using a quantized KV cache.
            when ``kv_bits`` is non-None. Default: ``0``.

    Yields:
        Generator[Tuple[mx.array, mx.array], None, None]: A generator producing
          one token and a vector of log probabilities.
    """

    ### default sampler
    def sampling(logits: mx.array) -> mx.array:
        # logprobs = logits - mx.logsumexp(logits) ### removing logprobs from sample function

        if temp == 0:
            token = mx.argmax(logits, axis=-1)
        else:
            if top_p > 0 and top_p < 1.0:
                token = top_p_sampling(logits, top_p, temp)
            elif min_p != 0.0:
                token = min_p_sampling(logits, min_p, min_tokens_to_keep, temp)
            else:
                token = categorical_sampling(logits, temp)

        return token #, logprobs
    
    if isinstance(sampler, EntropixSamplerConfig):
        from entropix.config import EntropixConfig
        entropix_cfg = EntropixConfig()

    if repetition_penalty and (
        repetition_penalty < 0 or not isinstance(repetition_penalty, float)
    ):
        raise ValueError(
            f"repetition_penalty must be a non-negative float, got {repetition_penalty}"
        )

    logits_processor = logits_processor or []

    if repetition_penalty:

        def repetition_penalty_processor(tokens, logits):
            return apply_repetition_penalty(
                logits, tokens[-repetition_context_size:], repetition_penalty
            )

        logits_processor.append(repetition_penalty_processor)

    if logit_bias:
        indices = mx.array(list(logit_bias.keys()))
        values = mx.array(list(logit_bias.values()))

        def logit_bias_processor(_, logits): # using _ as tokens is not used
            logits[:, indices] += values
            return logits

        logits_processor.append(logit_bias_processor)

    y = prompt
    tokens = None
    seq_length = y.shape[0]

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model,
            max_kv_size=max_kv_size,
        )
    elif len(prompt_cache) != len(model.layers):
        raise ValueError("Wrong number of layers in the prompt cache.")

    def _step(y:mx.array, n:int) -> Tuple[mx.array, mx.array, Optional[Dict[str, mx.array]]]:

        current_pos=n ### keeping track of n for entropix
        logits, scores = model(y[None], cache=prompt_cache) ### adding scores for entropix
        logits = logits[:, -1, :] ### no keepdims so becomes batch x vocab

        ### keeping track of tokens for entropix
        nonlocal tokens
        tokens = mx.concat([tokens, y]) if tokens is not None else y

        if logits_processor:
            for processor in logits_processor: ### how does order affect the output?
                logits = processor(tokens, logits) ### careful as it impacts entropix
                ### maybe entropix should just be another processor?

        maybe_quantize_kv_cache(
            prompt_cache, quantized_kv_start, kv_group_size, kv_bits
        )

        logprobs = logits - mx.logsumexp(logits) ### adding logprobs and removing from sample function

        ### sampler selection
        if isinstance(sampler, EntropixSamplerConfig):
            y, metrics = entropix_sampling(
                tokens, 
                logits, 
                scores, 
                sampler, 
                entropix_cfg,
                current_pos, 
                # clarifying_question_token ### now passed as part of sampler config
                # rng_key ### no need for key anymore as it is handled via the partial decorator
            )
            y=y.squeeze(0) ### removing batch dimension that is returned by entropix sampler
        else: ### TODO : let user chose the mlx sampler or the _entropix_sampler
            metrics = None
            ### default sampler
            y = sampling(logits)

            ### entropix _sampler for debugging
            # y = _entropix_sampling(
            #     logits,
            #     temp,
            #     top_p,
            #     50, ### top_k TODO : make this a parameter
            #     min_p,
            # )
            # y = y.squeeze(0) ### removing batch dimension that is returned by entropix sampler

        return y, logprobs.squeeze(0), metrics ### added metrics for entropix viz  

    while y.size > prefill_step_size:
        model(y[:prefill_step_size][None], cache=prompt_cache)
        mx.eval([c.state for c in prompt_cache])
        y = y[prefill_step_size:]
        mx.metal.clear_cache()

    y, logprobs, metrics = _step(y, seq_length) ### added metrics for entropix sampler viz. 

    mx.async_eval(y, logprobs) ### no eval metrics unless entropix sampler is used?
    n = 0
    while True:
        yield y.item(), logprobs, metrics  ### added metrix for entropix sampler.  
        next_y, next_logprobs, next_metrics = _step(y, seq_length + n + 1) ### keeping track of n for entropix
        mx.async_eval(next_y, next_logprobs)
        if n % 256 == 0:
            mx.metal.clear_cache()
        n += 1
        y, logprobs, metrics = next_y, next_logprobs, next_metrics 


def stream_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: str,
    max_tokens: int = 100,
    **kwargs,
) -> Union[str, Generator[str, None, None]]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        max_tokens (int): The ma
        kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.

    Yields:
        Generator[Tuple[mx.array, mx.array]]: A generator producing text.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    ### TODO : implement entropix sampling

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    detokenizer = tokenizer.detokenizer

    detokenizer.reset()
    for n, (token, _, metrics) in zip(
        range(max_tokens),
        generate_step(prompt_tokens, model, **kwargs),
    ):
        if token == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token)

        # Yield the last segment if streaming
        yield detokenizer.last_segment

    detokenizer.finalize()
    yield detokenizer.last_segment


def generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: str,
    max_tokens: int = 100,
    verbose: bool = False,
    formatter: Optional[Callable] = None,
    **kwargs,
) -> Union[str, Generator[str, None, None]]:
    """
    Generate a complete response from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       max_tokens (int): The maximum number of tokens. Default: ``100``.
       verbose (bool): If ``True``, print tokens and timing information.
           Default: ``False``.
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    sampler = kwargs.get("sampler", None )
    entropix = isinstance(sampler, EntropixSamplerConfig) 

    if entropix:
        ### Initialize lists to store metrics (for entropix sampling)
        metrics_data = {
            'logits_entropy': [],
            'logits_varentropy': [],
            'attn_entropy': [],
            'attn_varentropy': [],
            'sampler_state':[] ### moving sampler state here
        }
        # sampler_states = []
        # generated_tokens = []

    if verbose:
        print("=" * 10)
        print("Prompt:", prompt)

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    detokenizer = tokenizer.detokenizer

    with wired_limit(model):
        tic = time.perf_counter()
        detokenizer.reset()
        for n, (token, logprobs, metrics) in zip( ### passing sampling metrics would avoid a second calculation
            range(max_tokens),
            generate_step(prompt_tokens, model, **kwargs),  ### metrics now retruned
        ):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                tic = time.perf_counter()
            if token == tokenizer.eos_token_id: ### could be a list of stop tokens
                break
            detokenizer.add_token(token)

            if verbose:
                if formatter:
                    # We have to finalize so that the prob corresponds to the last segment
                    detokenizer.finalize()
                    with mx.stream(mx.cpu):
                        prob = mx.exp(logprobs[token]).item()
                    formatter(detokenizer.last_segment, prob)
                else:
                    print(detokenizer.last_segment, end="", flush=True)

            if entropix:
                # metrics = calculate_metrics(logits, scores, n) ### get metrics directly from the sampling
                for key in metrics_data.keys():
                    if key in metrics:
                        if key == "sampler_state":
                            metrics_data[key].append(metrics[key])
                        else:
                            metrics_data[key].append(metrics[key].item())

        token_count = n + 1
        detokenizer.finalize()

        if verbose:
            gen_time = time.perf_counter() - tic
            print(detokenizer.last_segment, flush=True)
            print("=" * 10)
            if token_count == 0:
                print("No tokens generated for this prompt")
                return
            prompt_tps = prompt_tokens.size / prompt_time
            gen_tps = (token_count - 1) / gen_time
            print(
                f"Prompt: {prompt_tokens.size} tokens, {prompt_tps:.3f} tokens-per-sec"
            )
            print(f"Generation: {token_count} tokens, {gen_tps:.3f} tokens-per-sec")
            peak_mem = mx.metal.get_peak_memory() / 2**30
            print(f"Peak memory: {peak_mem:.3f} GB")

        if entropix:
            print("=" * 10)
            print('Starting data viz')
            visualize_sampler_metrics(
                metrics_data['logits_entropy'], 
                metrics_data['logits_varentropy'],
                metrics_data['sampler_state'], ### now as part of metrics
                detokenizer.tokens, 
                tokenizer
            )
            visualize_token_entropy_varentropy(
                metrics_data, 
                detokenizer.tokens, 
                tokenizer, 
                sampler
            )
        else:
            print('not entropix')

        return detokenizer.text


def load_config(model_path: Path) -> dict:
    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found in {model_path}")
        raise
    return config


def load_model(
    model_path: Path,
    lazy: bool = False,
    model_config: dict = {},
    get_model_classes: Callable[[dict], Tuple[Type[nn.Module], Type]] = _get_classes,
) -> nn.Module:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        model_config (dict, optional): Configuration parameters for the model.
            Defaults to an empty dictionary.
        get_model_classes (Callable[[dict], Tuple[Type[nn.Module], Type]], optional):
            A function that returns the model class and model args class given a config.
            Defaults to the _get_classes function.

    Returns:
        nn.Module: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """

    config = load_config(model_path)
    config.update(model_config)

    weight_files = glob.glob(str(model_path / "model*.safetensors"))

    if not weight_files:
        # Try weight for back-compat
        weight_files = glob.glob(str(model_path / "weight*.safetensors"))

    if not weight_files:
        logging.error(f"No safetensors found in {model_path}")
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model_class, model_args_class = get_model_classes(config=config)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    if (quantization := config.get("quantization", None)) is not None:
        # Handle legacy models which may not have everything quantized
        def class_predicate(p, m):
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            **quantization,
            class_predicate=class_predicate,
        )

    model.load_weights(list(weights.items()))

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model


def load(
    path_or_hf_repo: str,
    tokenizer_config={},
    model_config={},
    adapter_path: Optional[str] = None, ## for now, disabling adapter loading
    lazy: bool = False,
) -> Tuple[nn.Module, TokenizerWrapper]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        model_config(dict, optional): Configuration parameters specifically for the model.
            Defaults to an empty dictionary.
        adapter_path (str, optional): Path to the LoRA adapters. If provided, applies LoRA layers
            to the model. Default: ``None``.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
    Returns:
        Tuple[nn.Module, TokenizerWrapper]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    model_path = get_model_path(path_or_hf_repo)

    model = load_model(model_path, lazy, model_config)
    ## for now, disabling adapter loading
    # if adapter_path is not None:
    #     model = load_adapters(model, adapter_path)
    #     model.eval()
    tokenizer = load_tokenizer(model_path, tokenizer_config)

    return model, tokenizer

### fetching fully covered by load function
# def fetch_from_hub(
#     model_path: Path, lazy: bool = False
# ) -> Tuple[nn.Module, dict, PreTrainedTokenizer]:
#     model = load_model(model_path, lazy)
#     config = load_config(model_path)
#     tokenizer = load_tokenizer(model_path)
#     return model, config, tokenizer

### for now, commenting out all quantization and upload functions
# def make_shards(weights: dict, max_file_size_gb: int = MAX_FILE_SIZE_GB) -> list:
#     """
#     Splits the weights into smaller shards.

#     Args:
#         weights (dict): Model weights.
#         max_file_size_gb (int): Maximum size of each shard in gigabytes.

#     Returns:
#         list: List of weight shards.
#     """
#     max_file_size_bytes = max_file_size_gb << 30
#     shards = []
#     shard, shard_size = {}, 0
#     for k, v in weights.items():
#         if shard_size + v.nbytes > max_file_size_bytes:
#             shards.append(shard)
#             shard, shard_size = {}, 0
#         shard[k] = v
#         shard_size += v.nbytes
#     shards.append(shard)
#     return shards


# def upload_to_hub(path: str, upload_repo: str, hf_path: str):
#     """
#     Uploads the model to Hugging Face hub.

#     Args:
#         path (str): Local path to the model.
#         upload_repo (str): Name of the HF repo to upload to.
#         hf_path (str): Path to the original Hugging Face model.
#     """
#     import os

#     from huggingface_hub import HfApi, ModelCard, logging

#     from . import __version__

#     card = ModelCard.load(hf_path)
#     card.data.tags = ["mlx"] if card.data.tags is None else card.data.tags + ["mlx"]
#     card.data.base_model = hf_path
#     card.text = dedent(
#         f"""
#         # {upload_repo}

#         The Model [{upload_repo}](https://huggingface.co/{upload_repo}) was converted to MLX format from [{hf_path}](https://huggingface.co/{hf_path}) using mlx-lm version **{__version__}**.

#         ## Use with mlx

#         ```bash
#         pip install mlx-lm
#         ```

#         ```python
#         from mlx_lm import load, generate

#         model, tokenizer = load("{upload_repo}")

#         prompt="hello"

#         if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
#             messages = [{{"role": "user", "content": prompt}}]
#             prompt = tokenizer.apply_chat_template(
#                 messages, tokenize=False, add_generation_prompt=True
#             )

#         response = generate(model, tokenizer, prompt=prompt, verbose=True)
#         ```
#         """
#     )
#     card.save(os.path.join(path, "README.md"))

#     logging.set_verbosity_info()

#     api = HfApi()
#     api.create_repo(repo_id=upload_repo, exist_ok=True)
#     api.upload_folder(
#         folder_path=path,
#         repo_id=upload_repo,
#         repo_type="model",
#         multi_commits=True,
#         multi_commits_verbose=True,
#     )
#     print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")


# def save_weights(
#     save_path: Union[str, Path],
#     weights: Dict[str, Any],
#     *,
#     donate_weights: bool = False,
# ) -> None:
#     """Save model weights into specified directory."""
#     if isinstance(save_path, str):
#         save_path = Path(save_path)
#     save_path.mkdir(parents=True, exist_ok=True)

#     shards = make_shards(weights)
#     shards_count = len(shards)
#     shard_file_format = (
#         "model-{:05d}-of-{:05d}.safetensors"
#         if shards_count > 1
#         else "model.safetensors"
#     )

#     total_size = sum(v.nbytes for v in weights.values())
#     index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

#     # Write the weights and make sure no references are kept other than the
#     # necessary ones
#     if donate_weights:
#         weights.clear()
#         del weights

#     for i in range(len(shards)):
#         shard = shards[i]
#         shards[i] = None
#         shard_name = shard_file_format.format(i + 1, shards_count)
#         shard_path = save_path / shard_name

#         mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})

#         for weight_name in shard.keys():
#             index_data["weight_map"][weight_name] = shard_name
#         del shard

#     index_data["weight_map"] = {
#         k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
#     }

#     with open(save_path / "model.safetensors.index.json", "w") as f:
#         json.dump(
#             index_data,
#             f,
#             indent=4,
#         )


# def quantize_model(
#     model: nn.Module, config: dict, q_group_size: int, q_bits: int
# ) -> Tuple:
#     """
#     Applies quantization to the model weights.

#     Args:
#         model (nn.Module): The model to be quantized.
#         config (dict): Model configuration.
#         q_group_size (int): Group size for quantization.
#         q_bits (int): Bits per weight for quantization.

#     Returns:
#         Tuple: Tuple containing quantized weights and config.
#     """
#     quantized_config = copy.deepcopy(config)
#     nn.quantize(model, q_group_size, q_bits)
#     quantized_config["quantization"] = {"group_size": q_group_size, "bits": q_bits}
#     # support hf model tree #957
#     quantized_config["quantization_config"] = quantized_config["quantization"]
#     quantized_weights = dict(tree_flatten(model.parameters()))

#     return quantized_weights, quantized_config


# def save_config(
#     config: dict,
#     config_path: Union[str, Path],
# ) -> None:
#     """Save the model configuration to the ``config_path``.

#     The final configuration will be sorted before saving for better readability.

#     Args:
#         config (dict): The model configuration.
#         config_path (Union[str, Path]): Model configuration file path.
#     """
#     # Clean unused keys
#     config.pop("_name_or_path", None)

#     # sort the config for better readability
#     config = dict(sorted(config.items()))

#     # write the updated config to the config_path (if provided)
#     with open(config_path, "w") as fid:
#         json.dump(config, fid, indent=4)


# def convert(
#     hf_path: str,
#     mlx_path: str = "mlx_model",
#     quantize: bool = False,
#     q_group_size: int = 64,
#     q_bits: int = 4,
#     dtype: str = "float16",
#     upload_repo: str = None,
#     revision: Optional[str] = None,
#     dequantize: bool = False,
# ):
#     # Check the save path is empty
#     if isinstance(mlx_path, str):
#         mlx_path = Path(mlx_path)

#     if mlx_path.exists():
#         raise ValueError(
#             f"Cannot save to the path {mlx_path} as it already exists."
#             " Please delete the file/directory or specify a new path to save to."
#         )

#     print("[INFO] Loading")
#     model_path = get_model_path(hf_path, revision=revision)
#     model, config, tokenizer = fetch_from_hub(model_path, lazy=True)

#     weights = dict(tree_flatten(model.parameters()))
#     dtype = getattr(mx, dtype)
#     weights = {k: v.astype(dtype) for k, v in weights.items()}

#     if quantize and dequantize:
#         raise ValueError("Choose either quantize or dequantize, not both.")

#     if quantize:
#         print("[INFO] Quantizing")
#         model.load_weights(list(weights.items()))
#         weights, config = quantize_model(model, config, q_group_size, q_bits)

#     if dequantize:
#         print("[INFO] Dequantizing")
#         model = dequantize_model(model)
#         weights = dict(tree_flatten(model.parameters()))

#     del model
#     save_weights(mlx_path, weights, donate_weights=True)

#     py_files = glob.glob(str(model_path / "*.py"))
#     for file in py_files:
#         shutil.copy(file, mlx_path)

#     tokenizer.save_pretrained(mlx_path)

#     save_config(config, config_path=mlx_path / "config.json")

#     if upload_repo is not None:
#         upload_to_hub(mlx_path, upload_repo, hf_path)