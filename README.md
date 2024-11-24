<!-- TODO:
- python 13?
- should entropix be just another processor before sampling rather than a separate sampler?
- speculative decoding for llama, gemma, qwen
-->

# FAMLX
This project is a playground built on MLX, a great framework for tinkering with small LLMs. It's all about learning by doing, breaking things, and (sometimes) making them work again.    

The philosophy is simple: keeping things as straightforward as possible, trying to minimize abstractions and dependencies, while staying true to the awesome projects that introduced or inspired the showcased methods (proper credits and licenses below).  

FAMLX focuses purely on local inference - no training, adapters are disabled and quantization is untested. The generate.py file constitutes an appropriate entry point as it's where the whole process of turning a prompt into words begins. From there you should be able to track everything happening under the hood.   

## Quick start

uv is the quickest way to start. If you haven't done it already, [download uv here](https://docs.astral.sh/uv/getting-started/installation/) 

```bash
git clone https://github.com/pappitti/mlx-llms.git
```

```bash
uv sync
```

To generate text with an LLM use:

```bash
uv run generate.py --prompt "Tell me a story about a dog that loves ice"
```

The default model is defined in constants.py. To specify a model:
```bash
uv run generate.py --model mlx-community/Qwen2.5-0.5B-bf16 --prompt "Tell me a story about a dog that loves ice"
```

## Sources and Licences
This project leverages the work of amazing open-source projects listed below. Special thanks to them for making such valuable resources available. The different sources explain the different coding styles in different files.  
I removed a lot of features that I did not need for this project but I tried not to delete anything from the original files when I simplified functions or objects. I commented out unnecessary parts with '###' so my comments can be more easily distinguished from those in the original project. 

Licence : it is my understanding that the Apple Licence (MIT) is the most restrictive and therefore applies to this project. See more details about the individual parts below.
 
1. mlx-lm : [mlx-example-repo](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- root files
- models/*  
see [licence](https://github.com/ml-explore/mlx-examples/blob/main/LICENSE)

2. transformers: [HuggingFace transformers](https://github.com/huggingface/transformers/tree/main/src/transformers)
- tokenization/* :substantially all files were copied from the HF transformers library. The objective was to use PretrrainedTokenizers from transformers but not the rest since it's covered by mlx-examples. Pytorch and Tensorflow utils were removed as well as all upload utils. I tried to consolidate files when it made sense to me (initial source in the transformers library is clearly identified in the files)  
see [licence](https://github.com/huggingface/transformers/blob/main/LICENSE)

3. entropix : [entropix](https://github.com/xjdr-alt/entropix)  
More specifically [entropix-local mlx](https://github.com/xjdr-alt/entropix-local/tree/research/entropix/local/mlx) but also heavily inspired by [SmolLM x Entropix](https://github.com/SinatrasC/entropix-smollm)   
- entropix/* : Most of the code was lifted from the entropix-local repo (research branch) and adapted to this project. See notes for more information on what part of the project were implemented here.  
see [licence](https://github.com/xjdr-alt/entropix-local/blob/main/LICENSE)
  
  
## Models
VRAM will be your constraint : as a rule of thumb, 2GB of VRAM for each 1B parameters in bf/fp16, so choose your model accordingly. You can use quantized models to reduce VRAM requirements.  

Tokenizers and models architectures are available in this repo for the models or model families described below. Any model that matches a PreTrainedTokenizer and a model in this repo should work : for example, SmolLM2 works with a GTP2 tokenizer and Lllama model.  

Note: for base model, you may want to ignore chat templates `--ignore-chat-template` and increase repetition penalty, in particular for smaller models (unless you are prompting for a math question, in which case repeating tokens is necessary). If you use the entropix sampler, repetition penalty at 1 is better.  

### Qwen2.5
- mlx-community/Qwen2.5-0.5B-bf16  
ignoring chat template, forcing eos_token and higher repetition penalty: 
```bash
uv run generate.py --model mlx-community/Qwen2.5-0.5B-bf16 --ignore-chat-template --eos-token "<|endoftext|>" --repetition-penalty 1.5 --prompt "This is the story of a dog that loves ice. On a cold Winter day"
```
```bash
uv run generate.py --model mlx-community/Qwen2.5-0.5B-bf16 --ignore-chat-template --eos-token "<|endoftext|>" --repetition-penalty 1.5 --prompt "Which is the larger number between 9.11 and 9.8? Thinking step-by-step,"
```
- mlx-community/Qwen2.5-Coder-32B-Instruct-bf16 
```bash
uv run generate.py --model mlx-community/Qwen2.5-Coder-32B-Instruct-bf16 --eos-token "<|endoftext|>"  --prompt "Please a functional implementation of transformers, i.e not the usual object-oriented implementation"
```

### Llama3.2
- mlx-community/Llama-3.2-1B-Instruct-bf16 (currently set as default model in constants.py)  
```bash
uv run generate.py --prompt "Tell me a story about a dog that loves ice"
```
```bash
uv run generate.py --prompt "what is larger between 9.11 and 9.8? think step by step"
```

### Llama3.1
- mlx-community/Meta-Llama-3.1-70B-bf16 
70B in bf16 can run on a MacStudio with M2 Ultra and 192GB of RAM but will likely be very slow (see note on large models in the MLX section below)
```bash
uv run generate.py --model mlx-community/Meta-Llama-3.1-70B-bf16 --ignore-chat-template --prompt "To address the strengths and weaknesses of the business model of hairdressers, we must primarily focus on the route to market and cost base :"
```

### Gemma2  
- mlx-community/gemma-2-2b-it-fp16  
```bash
uv run generate.py --model mlx-community/gemma-2-2b-it-fp16 --prompt "Please write a the story about a dog that loves ice"
```
- mlx-community/gemma-2-2b-fp16  
```bash
uv run generate.py --model mlx-community/gemma-2-2b-fp16 --ignore-chat-template --prompt "This is the story of a dog that loves ice. On a cold Winter day"
```
- mlx-community/gemma-2-27b-fp16
```bash
uv run generate.py --model mlx-community/gemma-2-27b-fp16 --ignore-chat-template --prompt "To address the strengths and weaknesses of the business model of hairdressers, we must primarily focus on the route to market and cost base :"
```

### SmolLM2  
- mlx-community/SmolLM2-360M-Instruct  
suggesting to increase repetition penalty for smaller models  
```bash
uv run generate.py --model mlx-community/SmolLM2-360M-Instruct --repetition-penalty 1.5 --prompt "Please write a the story about a dog that loves ice"
```
```bash
uv run generate.py --model mlx-community/SmolLM2-360M-Instruct --repetition-penalty 1.5 --prompt "what is larger between 9.11 and 9.8? think step by step" --sampler "entropix"
```

## Entropix 
EXPERIMENTAL : A new argument was added to select the entropix sampler instead of the default sampler. use `--sampler "entropix"` to use this sampler.  

Note that this sampler is only the implementation of the original entropix project and does not reflect the current research branch. Adaptive sampling thresholds for Llama3.2-1B and SMOLLM2-360M are pulled from the the entropix-local and entropix-smollm respectively. See entropix/config.py for more details.  

Further work on the entropix methods would require to add jax to the project, which contradicts the overarching philosophy of FAMLX. Let's see what the original entropix project yields and if it's so compelling that a change in philosophy is mandated.            
  

## MLX
from the [mlx-example-repo](https://github.com/ml-explore/mlx-examples/blob/main/llms/README.md?plain=1)  
Use `-h` to see a list of available options for a command, e.g.:  
```bash
mlx_lm.generate -h
```

### Chat
Chat and streaming still WIP 
<!-- To chat with an LLM use:

```bash
mlx_lm.chat
```

This will give you a chat REPL that you can use to interact with the LLM. The
chat context is preserved during the lifetime of the REPL. 

Commands in `mlx-lm` typically take command line options which let you specify
the model, sampling parameters, and more. -->

### Large Models
This requires macOS 15.0 or higher to work.

Models which are large relative to the total RAM available on the machine can be slow. mlx-lm will attempt to make them faster by wiring the memory occupied by the model and cache. This requires macOS 15 or higher to work.

If you see the following warning message:

[WARNING] Generating with a model that requires ...

then the model will likely be slow on the given machine. If the model fits in RAM then it can often be sped up by increasing the system wired memory limit. To increase the limit, set the following sysctl:

sudo sysctl iogpu.wired_limit_mb=N
The value N should be larger than the size of the model in megabytes but smaller than the memory size of the machine.
<!-- 

### Python API

You can use `mlx-lm` as a module:

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

prompt = "Write a story about Einstein"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```

To see a description of all the arguments you can do:

```
>>> help(generate)
```

#### Streaming

For streaming generation, use the `stream_generate` function. This returns a
generator object which streams the output text. For example,

```python
from mlx_lm import load, stream_generate

repo = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
model, tokenizer = load(repo)

prompt = "Write a story about Einstein"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

for t in stream_generate(model, tokenizer, prompt, max_tokens=512):
    print(t, end="", flush=True)
print()
```

### Command Line

You can also use `mlx-lm` from the command line with:

```
mlx_lm.generate --model mistralai/Mistral-7B-Instruct-v0.3 --prompt "hello"
```

This will download a Mistral 7B model from the Hugging Face Hub and generate
text using the given prompt.

For a full list of options run:

```
mlx_lm.generate --help
```


Large Models
Note

This requires macOS 15.0 or higher to work.

Models which are large relative to the total RAM available on the machine can be slow. mlx-lm will attempt to make them faster by wiring the memory occupied by the model and cache. This requires macOS 15 or higher to work.

If you see the following warning message:

[WARNING] Generating with a model that requires ...

then the model will likely be slow on the given machine. If the model fits in RAM then it can often be sped up by increasing the system wired memory limit. To increase the limit, set the following sysctl:

sudo sysctl iogpu.wired_limit_mb=N
The value N should be larger than the size of the model in megabytes but smaller than the memory size of the machine. -->