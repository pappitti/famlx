
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration base class and utilities."""

import copy
import json
import importlib
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, OrderedDict

from packaging import version

###from .dynamic_module_utils import custom_object_save, get_class_from_dynamic_module, resolve_trust_remote_code ### custom_object_save moved to utils and trust remote code should not be needed in this project
### from .modeling_gguf_pytorch_utils import load_gguf_checkpoint
from .utils import (
    CONFIG_NAME,
    PushToHubMixin,
    add_model_info_to_auto_map,
    add_model_info_to_custom_pipelines,
    cached_file,
    copy_func,
    download_url,
    extract_commit_hash,
    is_remote_url,
    ### is_torch_available,
    get_logger,
    ### find_adapter_config_file,
    ### is_peft_available,
    ### requires_backends,
    custom_object_save,
    __version__
)

### from .configuration_auto import AutoConfig, model_type_to_module_name, replace_list_option_in_docstrings ## added directly to this file

### should not happen with MLX
# if is_torch_available():
#     from ...generation import GenerationMixin

logger = get_logger(__name__)

### pasted from transformers/models/auto/configuration_auto.py for simplicity
CONFIG_MAPPING_NAMES = OrderedDict(
    [
        # Add configs here
        ("albert", "AlbertConfig"),
        ("align", "AlignConfig"),
        ("altclip", "AltCLIPConfig"),
        ("audio-spectrogram-transformer", "ASTConfig"),
        ("autoformer", "AutoformerConfig"),
        ("bark", "BarkConfig"),
        ("bart", "BartConfig"),
        ("beit", "BeitConfig"),
        ("bert", "BertConfig"),
        ("bert-generation", "BertGenerationConfig"),
        ("big_bird", "BigBirdConfig"),
        ("bigbird_pegasus", "BigBirdPegasusConfig"),
        ("biogpt", "BioGptConfig"),
        ("bit", "BitConfig"),
        ("blenderbot", "BlenderbotConfig"),
        ("blenderbot-small", "BlenderbotSmallConfig"),
        ("blip", "BlipConfig"),
        ("blip-2", "Blip2Config"),
        ("bloom", "BloomConfig"),
        ("bridgetower", "BridgeTowerConfig"),
        ("bros", "BrosConfig"),
        ("camembert", "CamembertConfig"),
        ("canine", "CanineConfig"),
        ("chameleon", "ChameleonConfig"),
        ("chinese_clip", "ChineseCLIPConfig"),
        ("chinese_clip_vision_model", "ChineseCLIPVisionConfig"),
        ("clap", "ClapConfig"),
        ("clip", "CLIPConfig"),
        ("clip_text_model", "CLIPTextConfig"),
        ("clip_vision_model", "CLIPVisionConfig"),
        ("clipseg", "CLIPSegConfig"),
        ("clvp", "ClvpConfig"),
        ("code_llama", "LlamaConfig"),
        ("codegen", "CodeGenConfig"),
        ("cohere", "CohereConfig"),
        ("conditional_detr", "ConditionalDetrConfig"),
        ("convbert", "ConvBertConfig"),
        ("convnext", "ConvNextConfig"),
        ("convnextv2", "ConvNextV2Config"),
        ("cpmant", "CpmAntConfig"),
        ("ctrl", "CTRLConfig"),
        ("cvt", "CvtConfig"),
        ("dac", "DacConfig"),
        ("data2vec-audio", "Data2VecAudioConfig"),
        ("data2vec-text", "Data2VecTextConfig"),
        ("data2vec-vision", "Data2VecVisionConfig"),
        ("dbrx", "DbrxConfig"),
        ("deberta", "DebertaConfig"),
        ("deberta-v2", "DebertaV2Config"),
        ("decision_transformer", "DecisionTransformerConfig"),
        ("deformable_detr", "DeformableDetrConfig"),
        ("deit", "DeiTConfig"),
        ("depth_anything", "DepthAnythingConfig"),
        ("deta", "DetaConfig"),
        ("detr", "DetrConfig"),
        ("dinat", "DinatConfig"),
        ("dinov2", "Dinov2Config"),
        ("distilbert", "DistilBertConfig"),
        ("donut-swin", "DonutSwinConfig"),
        ("dpr", "DPRConfig"),
        ("dpt", "DPTConfig"),
        ("efficientformer", "EfficientFormerConfig"),
        ("efficientnet", "EfficientNetConfig"),
        ("electra", "ElectraConfig"),
        ("encodec", "EncodecConfig"),
        ("encoder-decoder", "EncoderDecoderConfig"),
        ("ernie", "ErnieConfig"),
        ("ernie_m", "ErnieMConfig"),
        ("esm", "EsmConfig"),
        ("falcon", "FalconConfig"),
        ("falcon_mamba", "FalconMambaConfig"),
        ("fastspeech2_conformer", "FastSpeech2ConformerConfig"),
        ("flaubert", "FlaubertConfig"),
        ("flava", "FlavaConfig"),
        ("fnet", "FNetConfig"),
        ("focalnet", "FocalNetConfig"),
        ("fsmt", "FSMTConfig"),
        ("funnel", "FunnelConfig"),
        ("fuyu", "FuyuConfig"),
        ("gemma", "GemmaConfig"),
        ("gemma2", "Gemma2Config"),
        ("git", "GitConfig"),
        ("glm", "GlmConfig"),
        ("glpn", "GLPNConfig"),
        ("gpt-sw3", "GPT2Config"),
        ("gpt2", "GPT2Config"),
        ("gpt_bigcode", "GPTBigCodeConfig"),
        ("gpt_neo", "GPTNeoConfig"),
        ("gpt_neox", "GPTNeoXConfig"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseConfig"),
        ("gptj", "GPTJConfig"),
        ("gptsan-japanese", "GPTSanJapaneseConfig"),
        ("granite", "GraniteConfig"),
        ("granitemoe", "GraniteMoeConfig"),
        ("graphormer", "GraphormerConfig"),
        ("grounding-dino", "GroundingDinoConfig"),
        ("groupvit", "GroupViTConfig"),
        ("hiera", "HieraConfig"),
        ("hubert", "HubertConfig"),
        ("ibert", "IBertConfig"),
        ("idefics", "IdeficsConfig"),
        ("idefics2", "Idefics2Config"),
        ("idefics3", "Idefics3Config"),
        ("imagegpt", "ImageGPTConfig"),
        ("informer", "InformerConfig"),
        ("instructblip", "InstructBlipConfig"),
        ("instructblipvideo", "InstructBlipVideoConfig"),
        ("jamba", "JambaConfig"),
        ("jetmoe", "JetMoeConfig"),
        ("jukebox", "JukeboxConfig"),
        ("kosmos-2", "Kosmos2Config"),
        ("layoutlm", "LayoutLMConfig"),
        ("layoutlmv2", "LayoutLMv2Config"),
        ("layoutlmv3", "LayoutLMv3Config"),
        ("led", "LEDConfig"),
        ("levit", "LevitConfig"),
        ("lilt", "LiltConfig"),
        ("llama", "LlamaConfig"),
        ("llava", "LlavaConfig"),
        ("llava_next", "LlavaNextConfig"),
        ("llava_next_video", "LlavaNextVideoConfig"),
        ("llava_onevision", "LlavaOnevisionConfig"),
        ("longformer", "LongformerConfig"),
        ("longt5", "LongT5Config"),
        ("luke", "LukeConfig"),
        ("lxmert", "LxmertConfig"),
        ("m2m_100", "M2M100Config"),
        ("mamba", "MambaConfig"),
        ("mamba2", "Mamba2Config"),
        ("marian", "MarianConfig"),
        ("markuplm", "MarkupLMConfig"),
        ("mask2former", "Mask2FormerConfig"),
        ("maskformer", "MaskFormerConfig"),
        ("maskformer-swin", "MaskFormerSwinConfig"),
        ("mbart", "MBartConfig"),
        ("mctct", "MCTCTConfig"),
        ("mega", "MegaConfig"),
        ("megatron-bert", "MegatronBertConfig"),
        ("mgp-str", "MgpstrConfig"),
        ("mimi", "MimiConfig"),
        ("mistral", "MistralConfig"),
        ("mixtral", "MixtralConfig"),
        ("mllama", "MllamaConfig"),
        ("mobilebert", "MobileBertConfig"),
        ("mobilenet_v1", "MobileNetV1Config"),
        ("mobilenet_v2", "MobileNetV2Config"),
        ("mobilevit", "MobileViTConfig"),
        ("mobilevitv2", "MobileViTV2Config"),
        ("moshi", "MoshiConfig"),
        ("mpnet", "MPNetConfig"),
        ("mpt", "MptConfig"),
        ("mra", "MraConfig"),
        ("mt5", "MT5Config"),
        ("musicgen", "MusicgenConfig"),
        ("musicgen_melody", "MusicgenMelodyConfig"),
        ("mvp", "MvpConfig"),
        ("nat", "NatConfig"),
        ("nemotron", "NemotronConfig"),
        ("nezha", "NezhaConfig"),
        ("nllb-moe", "NllbMoeConfig"),
        ("nougat", "VisionEncoderDecoderConfig"),
        ("nystromformer", "NystromformerConfig"),
        ("olmo", "OlmoConfig"),
        ("olmoe", "OlmoeConfig"),
        ("omdet-turbo", "OmDetTurboConfig"),
        ("oneformer", "OneFormerConfig"),
        ("open-llama", "OpenLlamaConfig"),
        ("openai-gpt", "OpenAIGPTConfig"),
        ("opt", "OPTConfig"),
        ("owlv2", "Owlv2Config"),
        ("owlvit", "OwlViTConfig"),
        ("paligemma", "PaliGemmaConfig"),
        ("patchtsmixer", "PatchTSMixerConfig"),
        ("patchtst", "PatchTSTConfig"),
        ("pegasus", "PegasusConfig"),
        ("pegasus_x", "PegasusXConfig"),
        ("perceiver", "PerceiverConfig"),
        ("persimmon", "PersimmonConfig"),
        ("phi", "PhiConfig"),
        ("phi3", "Phi3Config"),
        ("phimoe", "PhimoeConfig"),
        ("pix2struct", "Pix2StructConfig"),
        ("pixtral", "PixtralVisionConfig"),
        ("plbart", "PLBartConfig"),
        ("poolformer", "PoolFormerConfig"),
        ("pop2piano", "Pop2PianoConfig"),
        ("prophetnet", "ProphetNetConfig"),
        ("pvt", "PvtConfig"),
        ("pvt_v2", "PvtV2Config"),
        ("qdqbert", "QDQBertConfig"),
        ("qwen2", "Qwen2Config"),
        ("qwen2_audio", "Qwen2AudioConfig"),
        ("qwen2_audio_encoder", "Qwen2AudioEncoderConfig"),
        ("qwen2_moe", "Qwen2MoeConfig"),
        ("qwen2_vl", "Qwen2VLConfig"),
        ("rag", "RagConfig"),
        ("realm", "RealmConfig"),
        ("recurrent_gemma", "RecurrentGemmaConfig"),
        ("reformer", "ReformerConfig"),
        ("regnet", "RegNetConfig"),
        ("rembert", "RemBertConfig"),
        ("resnet", "ResNetConfig"),
        ("retribert", "RetriBertConfig"),
        ("roberta", "RobertaConfig"),
        ("roberta-prelayernorm", "RobertaPreLayerNormConfig"),
        ("roc_bert", "RoCBertConfig"),
        ("roformer", "RoFormerConfig"),
        ("rt_detr", "RTDetrConfig"),
        ("rt_detr_resnet", "RTDetrResNetConfig"),
        ("rwkv", "RwkvConfig"),
        ("sam", "SamConfig"),
        ("seamless_m4t", "SeamlessM4TConfig"),
        ("seamless_m4t_v2", "SeamlessM4Tv2Config"),
        ("segformer", "SegformerConfig"),
        ("seggpt", "SegGptConfig"),
        ("sew", "SEWConfig"),
        ("sew-d", "SEWDConfig"),
        ("siglip", "SiglipConfig"),
        ("siglip_vision_model", "SiglipVisionConfig"),
        ("speech-encoder-decoder", "SpeechEncoderDecoderConfig"),
        ("speech_to_text", "Speech2TextConfig"),
        ("speech_to_text_2", "Speech2Text2Config"),
        ("speecht5", "SpeechT5Config"),
        ("splinter", "SplinterConfig"),
        ("squeezebert", "SqueezeBertConfig"),
        ("stablelm", "StableLmConfig"),
        ("starcoder2", "Starcoder2Config"),
        ("superpoint", "SuperPointConfig"),
        ("swiftformer", "SwiftFormerConfig"),
        ("swin", "SwinConfig"),
        ("swin2sr", "Swin2SRConfig"),
        ("swinv2", "Swinv2Config"),
        ("switch_transformers", "SwitchTransformersConfig"),
        ("t5", "T5Config"),
        ("table-transformer", "TableTransformerConfig"),
        ("tapas", "TapasConfig"),
        ("time_series_transformer", "TimeSeriesTransformerConfig"),
        ("timesformer", "TimesformerConfig"),
        ("timm_backbone", "TimmBackboneConfig"),
        ("trajectory_transformer", "TrajectoryTransformerConfig"),
        ("transfo-xl", "TransfoXLConfig"),
        ("trocr", "TrOCRConfig"),
        ("tvlt", "TvltConfig"),
        ("tvp", "TvpConfig"),
        ("udop", "UdopConfig"),
        ("umt5", "UMT5Config"),
        ("unispeech", "UniSpeechConfig"),
        ("unispeech-sat", "UniSpeechSatConfig"),
        ("univnet", "UnivNetConfig"),
        ("upernet", "UperNetConfig"),
        ("van", "VanConfig"),
        ("video_llava", "VideoLlavaConfig"),
        ("videomae", "VideoMAEConfig"),
        ("vilt", "ViltConfig"),
        ("vipllava", "VipLlavaConfig"),
        ("vision-encoder-decoder", "VisionEncoderDecoderConfig"),
        ("vision-text-dual-encoder", "VisionTextDualEncoderConfig"),
        ("visual_bert", "VisualBertConfig"),
        ("vit", "ViTConfig"),
        ("vit_hybrid", "ViTHybridConfig"),
        ("vit_mae", "ViTMAEConfig"),
        ("vit_msn", "ViTMSNConfig"),
        ("vitdet", "VitDetConfig"),
        ("vitmatte", "VitMatteConfig"),
        ("vits", "VitsConfig"),
        ("vivit", "VivitConfig"),
        ("wav2vec2", "Wav2Vec2Config"),
        ("wav2vec2-bert", "Wav2Vec2BertConfig"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerConfig"),
        ("wavlm", "WavLMConfig"),
        ("whisper", "WhisperConfig"),
        ("xclip", "XCLIPConfig"),
        ("xglm", "XGLMConfig"),
        ("xlm", "XLMConfig"),
        ("xlm-prophetnet", "XLMProphetNetConfig"),
        ("xlm-roberta", "XLMRobertaConfig"),
        ("xlm-roberta-xl", "XLMRobertaXLConfig"),
        ("xlnet", "XLNetConfig"),
        ("xmod", "XmodConfig"),
        ("yolos", "YolosConfig"),
        ("yoso", "YosoConfig"),
        ("zamba", "ZambaConfig"),
        ("zoedepth", "ZoeDepthConfig"),
    ]
)


MODEL_NAMES_MAPPING = OrderedDict(
    [
        # Add full (and cased) model names here
        ("albert", "ALBERT"),
        ("align", "ALIGN"),
        ("altclip", "AltCLIP"),
        ("audio-spectrogram-transformer", "Audio Spectrogram Transformer"),
        ("autoformer", "Autoformer"),
        ("bark", "Bark"),
        ("bart", "BART"),
        ("barthez", "BARThez"),
        ("bartpho", "BARTpho"),
        ("beit", "BEiT"),
        ("bert", "BERT"),
        ("bert-generation", "Bert Generation"),
        ("bert-japanese", "BertJapanese"),
        ("bertweet", "BERTweet"),
        ("big_bird", "BigBird"),
        ("bigbird_pegasus", "BigBird-Pegasus"),
        ("biogpt", "BioGpt"),
        ("bit", "BiT"),
        ("blenderbot", "Blenderbot"),
        ("blenderbot-small", "BlenderbotSmall"),
        ("blip", "BLIP"),
        ("blip-2", "BLIP-2"),
        ("bloom", "BLOOM"),
        ("bort", "BORT"),
        ("bridgetower", "BridgeTower"),
        ("bros", "BROS"),
        ("byt5", "ByT5"),
        ("camembert", "CamemBERT"),
        ("canine", "CANINE"),
        ("chameleon", "Chameleon"),
        ("chinese_clip", "Chinese-CLIP"),
        ("chinese_clip_vision_model", "ChineseCLIPVisionModel"),
        ("clap", "CLAP"),
        ("clip", "CLIP"),
        ("clip_text_model", "CLIPTextModel"),
        ("clip_vision_model", "CLIPVisionModel"),
        ("clipseg", "CLIPSeg"),
        ("clvp", "CLVP"),
        ("code_llama", "CodeLlama"),
        ("codegen", "CodeGen"),
        ("cohere", "Cohere"),
        ("conditional_detr", "Conditional DETR"),
        ("convbert", "ConvBERT"),
        ("convnext", "ConvNeXT"),
        ("convnextv2", "ConvNeXTV2"),
        ("cpm", "CPM"),
        ("cpmant", "CPM-Ant"),
        ("ctrl", "CTRL"),
        ("cvt", "CvT"),
        ("dac", "DAC"),
        ("data2vec-audio", "Data2VecAudio"),
        ("data2vec-text", "Data2VecText"),
        ("data2vec-vision", "Data2VecVision"),
        ("dbrx", "DBRX"),
        ("deberta", "DeBERTa"),
        ("deberta-v2", "DeBERTa-v2"),
        ("decision_transformer", "Decision Transformer"),
        ("deformable_detr", "Deformable DETR"),
        ("deit", "DeiT"),
        ("deplot", "DePlot"),
        ("depth_anything", "Depth Anything"),
        ("depth_anything_v2", "Depth Anything V2"),
        ("deta", "DETA"),
        ("detr", "DETR"),
        ("dialogpt", "DialoGPT"),
        ("dinat", "DiNAT"),
        ("dinov2", "DINOv2"),
        ("distilbert", "DistilBERT"),
        ("dit", "DiT"),
        ("donut-swin", "DonutSwin"),
        ("dpr", "DPR"),
        ("dpt", "DPT"),
        ("efficientformer", "EfficientFormer"),
        ("efficientnet", "EfficientNet"),
        ("electra", "ELECTRA"),
        ("encodec", "EnCodec"),
        ("encoder-decoder", "Encoder decoder"),
        ("ernie", "ERNIE"),
        ("ernie_m", "ErnieM"),
        ("esm", "ESM"),
        ("falcon", "Falcon"),
        ("falcon_mamba", "FalconMamba"),
        ("fastspeech2_conformer", "FastSpeech2Conformer"),
        ("flan-t5", "FLAN-T5"),
        ("flan-ul2", "FLAN-UL2"),
        ("flaubert", "FlauBERT"),
        ("flava", "FLAVA"),
        ("fnet", "FNet"),
        ("focalnet", "FocalNet"),
        ("fsmt", "FairSeq Machine-Translation"),
        ("funnel", "Funnel Transformer"),
        ("fuyu", "Fuyu"),
        ("gemma", "Gemma"),
        ("gemma2", "Gemma2"),
        ("git", "GIT"),
        ("glm", "GLM"),
        ("glpn", "GLPN"),
        ("gpt-sw3", "GPT-Sw3"),
        ("gpt2", "OpenAI GPT-2"),
        ("gpt_bigcode", "GPTBigCode"),
        ("gpt_neo", "GPT Neo"),
        ("gpt_neox", "GPT NeoX"),
        ("gpt_neox_japanese", "GPT NeoX Japanese"),
        ("gptj", "GPT-J"),
        ("gptsan-japanese", "GPTSAN-japanese"),
        ("granite", "Granite"),
        ("granitemoe", "GraniteMoeMoe"),
        ("graphormer", "Graphormer"),
        ("grounding-dino", "Grounding DINO"),
        ("groupvit", "GroupViT"),
        ("herbert", "HerBERT"),
        ("hiera", "Hiera"),
        ("hubert", "Hubert"),
        ("ibert", "I-BERT"),
        ("idefics", "IDEFICS"),
        ("idefics2", "Idefics2"),
        ("idefics3", "Idefics3"),
        ("imagegpt", "ImageGPT"),
        ("informer", "Informer"),
        ("instructblip", "InstructBLIP"),
        ("instructblipvideo", "InstructBlipVideo"),
        ("jamba", "Jamba"),
        ("jetmoe", "JetMoe"),
        ("jukebox", "Jukebox"),
        ("kosmos-2", "KOSMOS-2"),
        ("layoutlm", "LayoutLM"),
        ("layoutlmv2", "LayoutLMv2"),
        ("layoutlmv3", "LayoutLMv3"),
        ("layoutxlm", "LayoutXLM"),
        ("led", "LED"),
        ("levit", "LeViT"),
        ("lilt", "LiLT"),
        ("llama", "LLaMA"),
        ("llama2", "Llama2"),
        ("llama3", "Llama3"),
        ("llava", "LLaVa"),
        ("llava_next", "LLaVA-NeXT"),
        ("llava_next_video", "LLaVa-NeXT-Video"),
        ("llava_onevision", "LLaVA-Onevision"),
        ("longformer", "Longformer"),
        ("longt5", "LongT5"),
        ("luke", "LUKE"),
        ("lxmert", "LXMERT"),
        ("m2m_100", "M2M100"),
        ("madlad-400", "MADLAD-400"),
        ("mamba", "Mamba"),
        ("mamba2", "mamba2"),
        ("marian", "Marian"),
        ("markuplm", "MarkupLM"),
        ("mask2former", "Mask2Former"),
        ("maskformer", "MaskFormer"),
        ("maskformer-swin", "MaskFormerSwin"),
        ("matcha", "MatCha"),
        ("mbart", "mBART"),
        ("mbart50", "mBART-50"),
        ("mctct", "M-CTC-T"),
        ("mega", "MEGA"),
        ("megatron-bert", "Megatron-BERT"),
        ("megatron_gpt2", "Megatron-GPT2"),
        ("mgp-str", "MGP-STR"),
        ("mimi", "Mimi"),
        ("mistral", "Mistral"),
        ("mixtral", "Mixtral"),
        ("mllama", "Mllama"),
        ("mluke", "mLUKE"),
        ("mms", "MMS"),
        ("mobilebert", "MobileBERT"),
        ("mobilenet_v1", "MobileNetV1"),
        ("mobilenet_v2", "MobileNetV2"),
        ("mobilevit", "MobileViT"),
        ("mobilevitv2", "MobileViTV2"),
        ("moshi", "Moshi"),
        ("mpnet", "MPNet"),
        ("mpt", "MPT"),
        ("mra", "MRA"),
        ("mt5", "MT5"),
        ("musicgen", "MusicGen"),
        ("musicgen_melody", "MusicGen Melody"),
        ("mvp", "MVP"),
        ("myt5", "myt5"),
        ("nat", "NAT"),
        ("nemotron", "Nemotron"),
        ("nezha", "Nezha"),
        ("nllb", "NLLB"),
        ("nllb-moe", "NLLB-MOE"),
        ("nougat", "Nougat"),
        ("nystromformer", "Nyströmformer"),
        ("olmo", "OLMo"),
        ("olmoe", "OLMoE"),
        ("omdet-turbo", "OmDet-Turbo"),
        ("oneformer", "OneFormer"),
        ("open-llama", "OpenLlama"),
        ("openai-gpt", "OpenAI GPT"),
        ("opt", "OPT"),
        ("owlv2", "OWLv2"),
        ("owlvit", "OWL-ViT"),
        ("paligemma", "PaliGemma"),
        ("patchtsmixer", "PatchTSMixer"),
        ("patchtst", "PatchTST"),
        ("pegasus", "Pegasus"),
        ("pegasus_x", "PEGASUS-X"),
        ("perceiver", "Perceiver"),
        ("persimmon", "Persimmon"),
        ("phi", "Phi"),
        ("phi3", "Phi3"),
        ("phimoe", "Phimoe"),
        ("phobert", "PhoBERT"),
        ("pix2struct", "Pix2Struct"),
        ("pixtral", "Pixtral"),
        ("plbart", "PLBart"),
        ("poolformer", "PoolFormer"),
        ("pop2piano", "Pop2Piano"),
        ("prophetnet", "ProphetNet"),
        ("pvt", "PVT"),
        ("pvt_v2", "PVTv2"),
        ("qdqbert", "QDQBert"),
        ("qwen2", "Qwen2"),
        ("qwen2_audio", "Qwen2Audio"),
        ("qwen2_audio_encoder", "Qwen2AudioEncoder"),
        ("qwen2_moe", "Qwen2MoE"),
        ("qwen2_vl", "Qwen2VL"),
        ("rag", "RAG"),
        ("realm", "REALM"),
        ("recurrent_gemma", "RecurrentGemma"),
        ("reformer", "Reformer"),
        ("regnet", "RegNet"),
        ("rembert", "RemBERT"),
        ("resnet", "ResNet"),
        ("retribert", "RetriBERT"),
        ("roberta", "RoBERTa"),
        ("roberta-prelayernorm", "RoBERTa-PreLayerNorm"),
        ("roc_bert", "RoCBert"),
        ("roformer", "RoFormer"),
        ("rt_detr", "RT-DETR"),
        ("rt_detr_resnet", "RT-DETR-ResNet"),
        ("rwkv", "RWKV"),
        ("sam", "SAM"),
        ("seamless_m4t", "SeamlessM4T"),
        ("seamless_m4t_v2", "SeamlessM4Tv2"),
        ("segformer", "SegFormer"),
        ("seggpt", "SegGPT"),
        ("sew", "SEW"),
        ("sew-d", "SEW-D"),
        ("siglip", "SigLIP"),
        ("siglip_vision_model", "SiglipVisionModel"),
        ("speech-encoder-decoder", "Speech Encoder decoder"),
        ("speech_to_text", "Speech2Text"),
        ("speech_to_text_2", "Speech2Text2"),
        ("speecht5", "SpeechT5"),
        ("splinter", "Splinter"),
        ("squeezebert", "SqueezeBERT"),
        ("stablelm", "StableLm"),
        ("starcoder2", "Starcoder2"),
        ("superpoint", "SuperPoint"),
        ("swiftformer", "SwiftFormer"),
        ("swin", "Swin Transformer"),
        ("swin2sr", "Swin2SR"),
        ("swinv2", "Swin Transformer V2"),
        ("switch_transformers", "SwitchTransformers"),
        ("t5", "T5"),
        ("t5v1.1", "T5v1.1"),
        ("table-transformer", "Table Transformer"),
        ("tapas", "TAPAS"),
        ("tapex", "TAPEX"),
        ("time_series_transformer", "Time Series Transformer"),
        ("timesformer", "TimeSformer"),
        ("timm_backbone", "TimmBackbone"),
        ("trajectory_transformer", "Trajectory Transformer"),
        ("transfo-xl", "Transformer-XL"),
        ("trocr", "TrOCR"),
        ("tvlt", "TVLT"),
        ("tvp", "TVP"),
        ("udop", "UDOP"),
        ("ul2", "UL2"),
        ("umt5", "UMT5"),
        ("unispeech", "UniSpeech"),
        ("unispeech-sat", "UniSpeechSat"),
        ("univnet", "UnivNet"),
        ("upernet", "UPerNet"),
        ("van", "VAN"),
        ("video_llava", "VideoLlava"),
        ("videomae", "VideoMAE"),
        ("vilt", "ViLT"),
        ("vipllava", "VipLlava"),
        ("vision-encoder-decoder", "Vision Encoder decoder"),
        ("vision-text-dual-encoder", "VisionTextDualEncoder"),
        ("visual_bert", "VisualBERT"),
        ("vit", "ViT"),
        ("vit_hybrid", "ViT Hybrid"),
        ("vit_mae", "ViTMAE"),
        ("vit_msn", "ViTMSN"),
        ("vitdet", "VitDet"),
        ("vitmatte", "ViTMatte"),
        ("vits", "VITS"),
        ("vivit", "ViViT"),
        ("wav2vec2", "Wav2Vec2"),
        ("wav2vec2-bert", "Wav2Vec2-BERT"),
        ("wav2vec2-conformer", "Wav2Vec2-Conformer"),
        ("wav2vec2_phoneme", "Wav2Vec2Phoneme"),
        ("wavlm", "WavLM"),
        ("whisper", "Whisper"),
        ("xclip", "X-CLIP"),
        ("xglm", "XGLM"),
        ("xlm", "XLM"),
        ("xlm-prophetnet", "XLM-ProphetNet"),
        ("xlm-roberta", "XLM-RoBERTa"),
        ("xlm-roberta-xl", "XLM-RoBERTa-XL"),
        ("xlm-v", "XLM-V"),
        ("xlnet", "XLNet"),
        ("xls_r", "XLS-R"),
        ("xlsr_wav2vec2", "XLSR-Wav2Vec2"),
        ("xmod", "X-MOD"),
        ("yolos", "YOLOS"),
        ("yoso", "YOSO"),
        ("zamba", "Zamba"),
        ("zoedepth", "ZoeDepth"),
    ]
)

# This is tied to the processing `-` -> `_` in `model_type_to_module_name`. For example, instead of putting
# `transfo-xl` (as in `CONFIG_MAPPING_NAMES`), we should use `transfo_xl`.
DEPRECATED_MODELS = [
    "bort",
    "deta",
    "efficientformer",
    "ernie_m",
    "gptsan_japanese",
    "graphormer",
    "jukebox",
    "mctct",
    "mega",
    "mmbt",
    "nat",
    "nezha",
    "open_llama",
    "qdqbert",
    "realm",
    "retribert",
    "speech_to_text_2",
    "tapex",
    "trajectory_transformer",
    "transfo_xl",
    "tvlt",
    "van",
    "vit_hybrid",
    "xlm_prophetnet",
]

SPECIAL_MODEL_TYPE_TO_MODULE_NAME = OrderedDict(
    [
        ("openai-gpt", "openai"),
        ("data2vec-audio", "data2vec"),
        ("data2vec-text", "data2vec"),
        ("data2vec-vision", "data2vec"),
        ("donut-swin", "donut"),
        ("kosmos-2", "kosmos2"),
        ("maskformer-swin", "maskformer"),
        ("xclip", "x_clip"),
        ("clip_vision_model", "clip"),
        ("qwen2_audio_encoder", "qwen2_audio"),
        ("clip_text_model", "clip"),
        ("siglip_vision_model", "siglip"),
        ("chinese_clip_vision_model", "chinese_clip"),
        ("rt_detr_resnet", "rt_detr"),
    ]
)


def model_type_to_module_name(key):
    """Converts a config key to the corresponding module."""
    # Special treatment
    if key in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:
        key = SPECIAL_MODEL_TYPE_TO_MODULE_NAME[key]

        if key in DEPRECATED_MODELS:
            key = f"deprecated.{key}"
        return key

    key = key.replace("-", "_")
    if key in DEPRECATED_MODELS:
        key = f"deprecated.{key}"

    return key

class _LazyConfigMapping(OrderedDict):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, mapping):
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]
        if key not in self._mapping:
            raise KeyError(key)
        value = self._mapping[key]
        module_name = model_type_to_module_name(key)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        if hasattr(self._modules[module_name], value):
            return getattr(self._modules[module_name], value)

        # Some of the mappings have entries model_type -> config of another model type. In that case we try to grab the
        # object at the top level.
        transformers_module = importlib.import_module("transformers")
        return getattr(transformers_module, value)

    def keys(self):
        return list(self._mapping.keys()) + list(self._extra_content.keys())

    def values(self):
        return [self[k] for k in self._mapping.keys()] + list(self._extra_content.values())

    def items(self):
        return [(k, self[k]) for k in self._mapping.keys()] + list(self._extra_content.items())

    def __iter__(self):
        return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

    def __contains__(self, item):
        return item in self._mapping or item in self._extra_content

    def register(self, key, value, exist_ok=False):
        """
        Register a new configuration in this mapping.
        """
        if key in self._mapping.keys() and not exist_ok:
            raise ValueError(f"'{key}' is already used by a Transformers config, pick another name.")
        self._extra_content[key] = value

CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)

def _get_class_name(model_class: Union[str, List[str]]):
    if isinstance(model_class, (list, tuple)):
        return " or ".join([f"[`{c}`]" for c in model_class if c is not None])
    return f"[`{model_class}`]"

def _list_model_options(indent, config_to_class=None, use_model_types=True):
    if config_to_class is None and not use_model_types:
        raise ValueError("Using `use_model_types=False` requires a `config_to_class` dictionary.")
    if use_model_types:
        if config_to_class is None:
            model_type_to_name = {model_type: f"[`{config}`]" for model_type, config in CONFIG_MAPPING_NAMES.items()}
        else:
            model_type_to_name = {
                model_type: _get_class_name(model_class)
                for model_type, model_class in config_to_class.items()
                if model_type in MODEL_NAMES_MAPPING
            }
        lines = [
            f"{indent}- **{model_type}** -- {model_type_to_name[model_type]} ({MODEL_NAMES_MAPPING[model_type]} model)"
            for model_type in sorted(model_type_to_name.keys())
        ]
    else:
        config_to_name = {
            CONFIG_MAPPING_NAMES[config]: _get_class_name(clas)
            for config, clas in config_to_class.items()
            if config in CONFIG_MAPPING_NAMES
        }
        config_to_model_name = {
            config: MODEL_NAMES_MAPPING[model_type] for model_type, config in CONFIG_MAPPING_NAMES.items()
        }
        lines = [
            f"{indent}- [`{config_name}`] configuration class:"
            f" {config_to_name[config_name]} ({config_to_model_name[config_name]} model)"
            for config_name in sorted(config_to_name.keys())
        ]
    return "\n".join(lines)

def replace_list_option_in_docstrings(config_to_class=None, use_model_types=True):
    def docstring_decorator(fn):
        docstrings = fn.__doc__
        if docstrings is None:
            # Example: -OO
            return fn
        lines = docstrings.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^(\s*)List options\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = re.search(r"^(\s*)List options\s*$", lines[i]).groups()[0]
            if use_model_types:
                indent = f"{indent}    "
            lines[i] = _list_model_options(indent, config_to_class=config_to_class, use_model_types=use_model_types)
            docstrings = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'List options' in its docstring as placeholder, current"
                f" docstring is:\n{docstrings}"
            )
        fn.__doc__ = docstrings
        return fn

    return docstring_decorator


class AutoConfig:
    r"""
    This is a generic configuration class that will be instantiated as one of the configuration classes of the library
    when created with the [`~AutoConfig.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def for_model(cls, model_type: str, *args, **kwargs):
        if model_type in CONFIG_MAPPING:
            config_class = CONFIG_MAPPING[model_type]
            return config_class(*args, **kwargs)
        raise ValueError(
            f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(CONFIG_MAPPING.keys())}"
        )

    @classmethod
    @replace_list_option_in_docstrings()
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate one of the configuration classes of the library from a pretrained model configuration.

        The configuration class to instantiate is selected based on the `model_type` property of the config object that
        is loaded, or when it's missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* of a pretrained model configuration hosted inside a model repo on
                      huggingface.co.
                    - A path to a *directory* containing a configuration file saved using the
                      [`~PretrainedConfig.save_pretrained`] method, or the [`~PreTrainedModel.save_pretrained`] method,
                      e.g., `./my_model_directory/`.
                    - A path or url to a saved configuration JSON *file*, e.g.,
                      `./my_model_directory/configuration.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.

                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs(additional keyword arguments, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Examples:

        ```python
        >>> from transformers import AutoConfig

        >>> # Download configuration from huggingface.co and cache.
        >>> config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")

        >>> # Download configuration from huggingface.co (user-uploaded) and cache.
        >>> config = AutoConfig.from_pretrained("dbmdz/bert-base-german-cased")

        >>> # If configuration file is in a directory (e.g., was saved using *save_pretrained('./test/saved_model/')*).
        >>> config = AutoConfig.from_pretrained("./test/bert_saved_model/")

        >>> # Load a specific configuration file.
        >>> config = AutoConfig.from_pretrained("./test/bert_saved_model/my_configuration.json")

        >>> # Change some config attributes when loading a pretrained config.
        >>> config = AutoConfig.from_pretrained("google-bert/bert-base-uncased", output_attentions=True, foo=False)
        >>> config.output_attentions
        True

        >>> config, unused_kwargs = AutoConfig.from_pretrained(
        ...     "google-bert/bert-base-uncased", output_attentions=True, foo=False, return_unused_kwargs=True
        ... )
        >>> config.output_attentions
        True

        >>> unused_kwargs
        {'foo': False}
        ```"""
        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        kwargs["_from_auto"] = True
        kwargs["name_or_path"] = pretrained_model_name_or_path
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        code_revision = kwargs.pop("code_revision", None)

        config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
        has_local_code = "model_type" in config_dict and config_dict["model_type"] in CONFIG_MAPPING
        ### trust_remote_code should not be True in the scope of this project (removed for dependancies)
        trust_remote_code = False ### replacing the commented code below
        # trust_remote_code = resolve_trust_remote_code(
        #     trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        # )

        # if has_remote_code and trust_remote_code:
        #     class_ref = config_dict["auto_map"]["AutoConfig"]
        #     config_class = get_class_from_dynamic_module(
        #         class_ref, pretrained_model_name_or_path, code_revision=code_revision, **kwargs
        #     )
        #     if os.path.isdir(pretrained_model_name_or_path):
        #         config_class.register_for_auto_class()
        #     return config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        # el
        if "model_type" in config_dict:
            try:
                config_class = CONFIG_MAPPING[config_dict["model_type"]]
            except KeyError:
                raise ValueError(
                    f"The checkpoint you are trying to load has model type `{config_dict['model_type']}` "
                    "but Transformers does not recognize this architecture. This could be because of an "
                    "issue with the checkpoint, or because your version of Transformers is out of date."
                )
            return config_class.from_dict(config_dict, **unused_kwargs)
        else:
            # Fallback: use pattern matching on the string.
            # We go from longer names to shorter names to catch roberta before bert (for instance)
            for pattern in sorted(CONFIG_MAPPING.keys(), key=len, reverse=True):
                if pattern in str(pretrained_model_name_or_path):
                    return CONFIG_MAPPING[pattern].from_dict(config_dict, **unused_kwargs)

        raise ValueError(
            f"Unrecognized model in {pretrained_model_name_or_path}. "
            f"Should have a `model_type` key in its {CONFIG_NAME}, or contain one of the following strings "
            f"in its name: {', '.join(CONFIG_MAPPING.keys())}"
        )

    @staticmethod
    def register(model_type, config, exist_ok=False):
        """
        Register a new configuration for this class.

        Args:
            model_type (`str`): The model type like "bert" or "gpt".
            config ([`PretrainedConfig`]): The config to register.
        """
        if issubclass(config, PretrainedConfig) and config.model_type != model_type:
            raise ValueError(
                "The config you are passing has a `model_type` attribute that is not consistent with the model type "
                f"you passed (config has {config.model_type} and you passed {model_type}. Fix one of those so they "
                "match!"
            )
        CONFIG_MAPPING.register(model_type, config, exist_ok=exist_ok)


### actual configuration_utils.py code starts here
_re_configuration_file = re.compile(r"config\.(.*)\.json")

class PretrainedConfig(PushToHubMixin):
    # no-format
    r"""
    Base class for all configuration classes. Handles a few parameters common to all models' configurations as well as
    methods for loading/downloading/saving configurations.

    <Tip>

    A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to
    initialize a model does **not** load the model weights. It only affects the model's configuration.

    </Tip>

    Class attributes (overridden by derived classes):

    - **model_type** (`str`) -- An identifier for the model type, serialized into the JSON file, and used to recreate
      the correct object in [`~transformers.AutoConfig`].
    - **is_composition** (`bool`) -- Whether the config class is composed of multiple sub-configs. In this case the
      config has to be initialized from two or more configs of type [`~transformers.PretrainedConfig`] like:
      [`~transformers.EncoderDecoderConfig`] or [`~RagConfig`].
    - **keys_to_ignore_at_inference** (`List[str]`) -- A list of keys to ignore by default when looking at dictionary
      outputs of the model during inference.
    - **attribute_map** (`Dict[str, str]`) -- A dict that maps model specific attribute names to the standardized
      naming of attributes.

    Common attributes (present in all subclasses):

    - **vocab_size** (`int`) -- The number of tokens in the vocabulary, which is also the first dimension of the
      embeddings matrix (this attribute may be missing for models that don't have a text modality like ViT).
    - **hidden_size** (`int`) -- The hidden size of the model.
    - **num_attention_heads** (`int`) -- The number of attention heads used in the multi-head attention layers of the
      model.
    - **num_hidden_layers** (`int`) -- The number of blocks in the model.

    <Tip warning={true}>

    Setting parameters for sequence generation in the model config is deprecated. For backward compatibility, loading
    some of them will still be possible, but attempting to overwrite them will throw an exception -- you should set
    them in a [~transformers.GenerationConfig]. Check the documentation of [~transformers.GenerationConfig] for more
    information about the individual parameters.

    </Tip>

    Arg:
        name_or_path (`str`, *optional*, defaults to `""`):
            Store the string that was passed to [`PreTrainedModel.from_pretrained`] or
            [`TFPreTrainedModel.from_pretrained`] as `pretrained_model_name_or_path` if the configuration was created
            with such a method.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return all hidden-states.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not the model should returns all attentions.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return a [`~transformers.utils.ModelOutput`] instead of a plain tuple.
        is_encoder_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as an encoder/decoder or not.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as decoder or not (in which case it's used as an encoder).
        cross_attention_hidden_size** (`bool`, *optional*):
            The hidden size of the cross-attention layer in case the model is used as a decoder in an encoder-decoder
            setting and the cross-attention hidden dimension differs from `self.config.hidden_size`.
        add_cross_attention (`bool`, *optional*, defaults to `False`):
            Whether cross-attention layers should be added to the model. Note, this option is only relevant for models
            that can be used as decoder models within the [`EncoderDecoderModel`] class, which consists of all models
            in `AUTO_MODELS_FOR_CAUSAL_LM`.
        tie_encoder_decoder (`bool`, *optional*, defaults to `False`):
            Whether all encoder weights should be tied to their equivalent decoder weights. This requires the encoder
            and decoder model to have the exact same parameter names.
        prune_heads (`Dict[int, List[int]]`, *optional*, defaults to `{}`):
            Pruned heads of the model. The keys are the selected layer indices and the associated values, the list of
            heads to prune in said layer.

            For instance `{1: [0, 2], 2: [2, 3]}` will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        chunk_size_feed_forward (`int`, *optional*, defaults to `0`):
            The chunk size of all feed forward layers in the residual attention blocks. A chunk size of `0` means that
            the feed forward layer is not chunked. A chunk size of n means that the feed forward layer processes `n` <
            sequence_length embeddings at a time. For more information on feed forward chunking, see [How does Feed
            Forward Chunking work?](../glossary.html#feed-forward-chunking).

        > Parameters for fine-tuning tasks

        architectures (`List[str]`, *optional*):
            Model architectures that can be used with the model pretrained weights.
        finetuning_task (`str`, *optional*):
            Name of the task used to fine-tune the model. This can be used when converting from an original (TensorFlow
            or PyTorch) checkpoint.
        id2label (`Dict[int, str]`, *optional*):
            A map from index (for instance prediction index, or target index) to label.
        label2id (`Dict[str, int]`, *optional*): A map from label to index for the model.
        num_labels (`int`, *optional*):
            Number of labels to use in the last layer added to the model, typically for a classification task.
        task_specific_params (`Dict[str, Any]`, *optional*):
            Additional keyword arguments to store for the current task.
        problem_type (`str`, *optional*):
            Problem type for `XxxForSequenceClassification` models. Can be one of `"regression"`,
            `"single_label_classification"` or `"multi_label_classification"`.

        > Parameters linked to the tokenizer

        tokenizer_class (`str`, *optional*):
            The name of the associated tokenizer class to use (if none is set, will use the tokenizer associated to the
            model by default).
        prefix (`str`, *optional*):
            A specific prompt that should be added at the beginning of each text before calling the model.
        bos_token_id (`int`, *optional*): The id of the _beginning-of-stream_ token.
        pad_token_id (`int`, *optional*): The id of the _padding_ token.
        eos_token_id (`int`, *optional*): The id of the _end-of-stream_ token.
        decoder_start_token_id (`int`, *optional*):
            If an encoder-decoder model starts decoding with a different token than _bos_, the id of that token.
        sep_token_id (`int`, *optional*): The id of the _separation_ token.

        > PyTorch specific parameters

        torchscript (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be used with Torchscript.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer.
        torch_dtype (`str`, *optional*):
            The `dtype` of the weights. This attribute can be used to initialize the model to a non-default `dtype`
            (which is normally `float32`) and thus allow for optimal storage allocation. For example, if the saved
            model is `float16`, ideally we want to load it back using the minimal amount of memory needed to load
            `float16` weights. Since the config object is stored in plain text, this attribute contains just the
            floating type string without the `torch.` prefix. For example, for `torch.float16` ``torch_dtype` is the
            `"float16"` string.

            This attribute is currently not being used during model loading time, but this may change in the future
            versions. But we can already start preparing for the future by saving the dtype with save_pretrained.

        > TensorFlow specific parameters

        use_bfloat16 (`bool`, *optional*, defaults to `False`):
            Whether or not the model should use BFloat16 scalars (only used by some TensorFlow models).
        tf_legacy_loss (`bool`, *optional*, defaults to `False`):
            Whether the model should use legacy TensorFlow losses. Legacy losses have variable output shapes and may
            not be XLA-compatible. This option is here for backward compatibility and will be removed in Transformers
            v5.
        loss_type (`str`, *optional*):
            The type of loss that the model should use. It should be in `LOSS_MAPPING`'s keys, otherwise the loss will
            be automatically infered from the model architecture.
    """

    model_type: str = ""
    base_config_key: str = ""
    sub_configs: Dict[str, "PretrainedConfig"] = {}
    is_composition: bool = False
    attribute_map: Dict[str, str] = {}
    _auto_class: Optional[str] = None

    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

    def __init__(self, **kwargs):
        # Attributes with defaults
        self.return_dict = kwargs.pop("return_dict", True)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.torchscript = kwargs.pop("torchscript", False)  # Only used by PyTorch models
        self.torch_dtype = kwargs.pop("torch_dtype", None)  # Only used by PyTorch models
        self.use_bfloat16 = kwargs.pop("use_bfloat16", False)
        self.tf_legacy_loss = kwargs.pop("tf_legacy_loss", False)  # Only used by TensorFlow models
        self.pruned_heads = kwargs.pop("pruned_heads", {})
        self.tie_word_embeddings = kwargs.pop(
            "tie_word_embeddings", True
        )  # Whether input and output word embeddings should be tied for all MLM, LM and Seq2Seq models.
        self.chunk_size_feed_forward = kwargs.pop("chunk_size_feed_forward", 0)

        # Is decoder is used in encoder-decoder models to differentiate encoder from decoder
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        self.is_decoder = kwargs.pop("is_decoder", False)
        self.cross_attention_hidden_size = kwargs.pop("cross_attention_hidden_size", None)
        self.add_cross_attention = kwargs.pop("add_cross_attention", False)
        self.tie_encoder_decoder = kwargs.pop("tie_encoder_decoder", False)

        # Retrocompatibility: Parameters for sequence generation. While we will keep the ability to load these
        # parameters, saving them will be deprecated. In a distant future, we won't need to load them.
        for parameter_name, default_value in self._get_global_generation_defaults().items():
            setattr(self, parameter_name, kwargs.pop(parameter_name, default_value))

        # Fine-tuning task arguments
        self.architectures = kwargs.pop("architectures", None)
        self.finetuning_task = kwargs.pop("finetuning_task", None)
        self.id2label = kwargs.pop("id2label", None)
        self.label2id = kwargs.pop("label2id", None)
        if self.label2id is not None and not isinstance(self.label2id, dict):
            raise ValueError("Argument label2id should be a dictionary.")
        if self.id2label is not None:
            if not isinstance(self.id2label, dict):
                raise ValueError("Argument id2label should be a dictionary.")
            num_labels = kwargs.pop("num_labels", None)
            if num_labels is not None and len(self.id2label) != num_labels:
                logger.warning(
                    f"You passed along `num_labels={num_labels}` with an incompatible id to label map: "
                    f"{self.id2label}. The number of labels wil be overwritten to {self.num_labels}."
                )
            self.id2label = {int(key): value for key, value in self.id2label.items()}
            # Keys are always strings in JSON so convert ids to int here.
        else:
            self.num_labels = kwargs.pop("num_labels", 2)

        ### should not happen with MLX
        # if self.torch_dtype is not None and isinstance(self.torch_dtype, str):
        #     # we will start using self.torch_dtype in v5, but to be consistent with
        #     # from_pretrained's torch_dtype arg convert it to an actual torch.dtype object
        #     if is_torch_available():
        #         import torch

        #         self.torch_dtype = getattr(torch, self.torch_dtype)

        # Tokenizer arguments TODO: eventually tokenizer and models should share the same config
        self.tokenizer_class = kwargs.pop("tokenizer_class", None)
        self.prefix = kwargs.pop("prefix", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.sep_token_id = kwargs.pop("sep_token_id", None)

        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # task specific arguments
        self.task_specific_params = kwargs.pop("task_specific_params", None)

        # regression / multi-label classification
        self.problem_type = kwargs.pop("problem_type", None)
        allowed_problem_types = ("regression", "single_label_classification", "multi_label_classification")
        if self.problem_type is not None and self.problem_type not in allowed_problem_types:
            raise ValueError(
                f"The config parameter `problem_type` was not understood: received {self.problem_type} "
                "but only 'regression', 'single_label_classification' and 'multi_label_classification' are valid."
            )

        # TPU arguments
        if kwargs.pop("xla_device", None) is not None:
            logger.warning(
                "The `xla_device` argument has been deprecated in v4.4.0 of Transformers. It is ignored and you can "
                "safely remove it from your `config.json` file."
            )

        # Name or path to the pretrained checkpoint
        self._name_or_path = str(kwargs.pop("name_or_path", ""))
        # Config hash
        self._commit_hash = kwargs.pop("_commit_hash", None)

        # Attention implementation to use, if relevant.
        self._attn_implementation_internal = kwargs.pop("attn_implementation", None)
        self._attn_implementation_autoset = False

        # Drop the transformers version info
        self.transformers_version = kwargs.pop("transformers_version", None)

        # Deal with gradient checkpointing
        if kwargs.get("gradient_checkpointing", False):
            warnings.warn(
                "Passing `gradient_checkpointing` to a config initialization is deprecated and will be removed in v5 "
                "Transformers. Using `model.gradient_checkpointing_enable()` instead, or if you are using the "
                "`Trainer` API, pass `gradient_checkpointing=True` in your `TrainingArguments`."
            )

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    @property
    def name_or_path(self) -> str:
        return getattr(self, "_name_or_path", None)

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(value)  # Make sure that name_or_path is a string (for JSON encoding)

    @property
    def use_return_dict(self) -> bool:
        """
        `bool`: Whether or not return [`~utils.ModelOutput`] instead of tuples.
        """
        # If torchscript is set, force `return_dict=False` to avoid jit errors
        return self.return_dict and not self.torchscript

    @property
    def num_labels(self) -> int:
        """
        `int`: The number of labels for classification models.
        """
        return len(self.id2label)

    @num_labels.setter
    def num_labels(self, num_labels: int):
        if not hasattr(self, "id2label") or self.id2label is None or len(self.id2label) != num_labels:
            self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
            self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))

    @property
    def _attn_implementation(self):
        # This property is made private for now (as it cannot be changed and a PreTrainedModel.use_attn_implementation method needs to be implemented.)
        if hasattr(self, "_attn_implementation_internal"):
            if self._attn_implementation_internal is None:
                # `config.attn_implementation` should never be None, for backward compatibility.
                return "eager"
            else:
                return self._attn_implementation_internal
        else:
            return "eager"

    @_attn_implementation.setter
    def _attn_implementation(self, value):
        self._attn_implementation_internal = value

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        self._set_token_in_kwargs(kwargs)

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        non_default_generation_parameters = self._get_non_default_generation_parameters()
        if len(non_default_generation_parameters) > 0:
            # TODO (joao): this should be an exception if the user has modified the loaded config. See #33886
            warnings.warn(
                "Some non-default generation parameters are set in the model config. These should go into either a) "
                "`model.generation_config` (as opposed to `model.config`); OR b) a GenerationConfig file "
                "(https://huggingface.co/docs/transformers/generation_strategies#save-a-custom-decoding-strategy-with-your-model)."
                "This warning will become an exception in the future."
                f"\nNon-default generation parameters: {str(non_default_generation_parameters)}",
                UserWarning,
            )

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file, use_diff=True)
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    @staticmethod
    def _set_token_in_kwargs(kwargs, token=None):
        """Temporary method to deal with `token` and `use_auth_token`.

        This method is to avoid apply the same changes in all model config classes that overwrite `from_pretrained`.

        Need to clean up `use_auth_token` in a follow PR.
        """
        # Some model config classes like CLIP define their own `from_pretrained` without the new argument `token` yet.
        if token is None:
            token = kwargs.pop("token", None)
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ) -> "PretrainedConfig":
        r"""
        Instantiate a [`PretrainedConfig`] (or a derived class) from a pretrained model configuration.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a configuration file saved using the
                  [`~PretrainedConfig.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved configuration JSON *file*, e.g., `./my_model_directory/configuration.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>

            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.

                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from this pretrained model.

        Examples:

        ```python
        # We can't instantiate directly the base class *PretrainedConfig* so let's show the examples on a
        # derived class: BertConfig
        config = BertConfig.from_pretrained(
            "google-bert/bert-base-uncased"
        )  # Download configuration from huggingface.co and cache.
        config = BertConfig.from_pretrained(
            "./test/saved_model/"
        )  # E.g. config (or model) was saved using *save_pretrained('./test/saved_model/')*
        config = BertConfig.from_pretrained("./test/saved_model/my_configuration.json")
        config = BertConfig.from_pretrained("google-bert/bert-base-uncased", output_attentions=True, foo=False)
        assert config.output_attentions == True
        config, unused_kwargs = BertConfig.from_pretrained(
            "google-bert/bert-base-uncased", output_attentions=True, foo=False, return_unused_kwargs=True
        )
        assert config.output_attentions == True
        assert unused_kwargs == {"foo": False}
        ```"""
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        cls._set_token_in_kwargs(kwargs, token)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if cls.base_config_key and cls.base_config_key in config_dict:
            config_dict = config_dict[cls.base_config_key]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            # sometimes the config has no `base_config_key` if the config is used in several composite models
            # e.g. LlamaConfig. In that case we try to see if there is match in `model_type` before raising a warning
            for k, v in config_dict.items():
                if isinstance(v, dict) and v.get("model_type") == cls.model_type:
                    config_dict = v

            # raise warning only if we still can't see a match in `model_type`
            if config_dict["model_type"] != cls.model_type:
                logger.warning(
                    f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                    f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
                )

        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        [`PretrainedConfig`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.

        """
        cls._set_token_in_kwargs(kwargs)

        original_kwargs = copy.deepcopy(kwargs)
        # Get config dict associated with the base config file
        config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict is None:
            return {}, kwargs
        if "_commit_hash" in config_dict:
            original_kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # That config file may point us toward another config file to use.
        if "configuration_files" in config_dict:
            configuration_file = get_configuration_file(config_dict["configuration_files"])
            config_dict, kwargs = cls._get_config_dict(
                pretrained_model_name_or_path, _configuration_file=configuration_file, **original_kwargs
            )

        return config_dict, kwargs

    @classmethod
    def _get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        subfolder = kwargs.pop("subfolder", "")
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        commit_hash = kwargs.pop("_commit_hash", None)

        gguf_file = kwargs.get("gguf_file", None)

        if trust_remote_code is True:
            logger.warning(
                "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
                " ignored."
            )

        user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
            # Special case when pretrained_model_name_or_path is a local file
            resolved_config_file = pretrained_model_name_or_path
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            configuration_file = pretrained_model_name_or_path if gguf_file is None else gguf_file
            resolved_config_file = download_url(pretrained_model_name_or_path)
        else:
            configuration_file = kwargs.pop("_configuration_file", CONFIG_NAME) if gguf_file is None else gguf_file

            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    configuration_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _commit_hash=commit_hash,
                )
                if resolved_config_file is None:
                    return None, kwargs
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the configuration of '{pretrained_model_name_or_path}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"
                    f" name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory"
                    f" containing a {configuration_file} file"
                )

        try:
            ### should not happen through the MLX code
            if gguf_file:
                ### should not be used through the MLX code but want to see error if it does
                raise ValueError("gguf_file should not be set")
                # config_dict = load_gguf_checkpoint(resolved_config_file, return_tensors=False)["config"]
            else:
                # Load config dict
                config_dict = cls._dict_from_json_file(resolved_config_file)

            config_dict["_commit_hash"] = commit_hash
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_config_file}' is not a valid JSON file."
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_config_file}")
        else:
            logger.info(f"loading configuration file {configuration_file} from cache at {resolved_config_file}")

        if "auto_map" in config_dict and not is_local:
            config_dict["auto_map"] = add_model_info_to_auto_map(
                config_dict["auto_map"], pretrained_model_name_or_path
            )
        if "custom_pipelines" in config_dict and not is_local:
            config_dict["custom_pipelines"] = add_model_info_to_custom_pipelines(
                config_dict["custom_pipelines"], pretrained_model_name_or_path
            )
        return config_dict, kwargs

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
        """
        Instantiates a [`PretrainedConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the [`~PretrainedConfig.get_config_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        # Those arguments may be passed along for our internal telemetry.
        # We remove them so they don't appear in `return_unused_kwargs`.
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)
        # The commit hash might have been updated in the `config_dict`, we don't want the kwargs to erase that update.
        if "_commit_hash" in kwargs and "_commit_hash" in config_dict:
            kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # We remove it from kwargs so that it does not appear in `return_unused_kwargs`.
        config_dict["attn_implementation"] = kwargs.pop("attn_implementation", None)

        config = cls(**config_dict)

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = {int(key): value for key, value in config.pruned_heads.items()}

        # Update config with kwargs if needed
        if "num_labels" in kwargs and "id2label" in kwargs:
            num_labels = kwargs["num_labels"]
            id2label = kwargs["id2label"] if kwargs["id2label"] is not None else []
            if len(id2label) != num_labels:
                raise ValueError(
                    f"You passed along `num_labels={num_labels }` with an incompatible id to label map: "
                    f"{kwargs['id2label']}. Since those arguments are inconsistent with each other, you should remove "
                    "one of them."
                )
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                current_attr = getattr(config, key)
                # To authorize passing a custom subconfig as kwarg in models that have nested configs.
                if isinstance(current_attr, PretrainedConfig) and isinstance(value, dict):
                    value = current_attr.__class__(**value)
                setattr(config, key, value)
                if key != "torch_dtype":
                    to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Model config {config}")
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> "PretrainedConfig":
        """
        Instantiates a [`PretrainedConfig`] from the path to a JSON file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from that JSON file.

        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __eq__(self, other):
        return isinstance(other, PretrainedConfig) and (self.__dict__ == other.__dict__)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def __iter__(self):
        for attr in self.__dict__:
            yield attr

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = PretrainedConfig().to_dict()

        # get class specific config dict
        class_config_dict = self.__class__().to_dict() if not self.is_composition else {}

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if (
                isinstance(getattr(self, key, None), PretrainedConfig)
                and key in class_config_dict
                and isinstance(class_config_dict[key], dict)
            ):
                # For nested configs we need to clean the diff recursively
                diff = recursive_diff_dict(value, class_config_dict[key], config_obj=getattr(self, key, None))
                if "model_type" in value:
                    # Needs to be set even if it's not in the diff
                    diff["model_type"] = value["model_type"]
                if len(diff) > 0:
                    serializable_config_dict[key] = diff
            elif (
                key not in default_config_dict
                or key == "transformers_version"
                or value != default_config_dict[key]
                or (key in class_config_dict and value != class_config_dict[key])
            ):
                serializable_config_dict[key] = value

        if hasattr(self, "quantization_config"):
            serializable_config_dict["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(self.quantization_config, dict)
                else self.quantization_config
            )

            # pop the `_pre_quantization_dtype` as torch.dtypes are not serializable.
            _ = serializable_config_dict.pop("_pre_quantization_dtype", None)

        self.dict_torch_dtype_to_str(serializable_config_dict)

        if "_attn_implementation_internal" in serializable_config_dict:
            del serializable_config_dict["_attn_implementation_internal"]

        return serializable_config_dict

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        if "_auto_class" in output:
            del output["_auto_class"]
        if "_commit_hash" in output:
            del output["_commit_hash"]
        if "_attn_implementation_internal" in output:
            del output["_attn_implementation_internal"]

        # Transformers version when serializing the model
        output["transformers_version"] = __version__

        for key, value in output.items():
            # Deal with nested configs like CLIP
            if isinstance(value, PretrainedConfig):
                value = value.to_dict()
                del value["transformers_version"]

            output[key] = value

        if hasattr(self, "quantization_config"):
            output["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(self.quantization_config, dict)
                else self.quantization_config
            )

            # pop the `_pre_quantization_dtype` as torch.dtypes are not serializable.
            _ = output.pop("_pre_quantization_dtype", None)

        self.dict_torch_dtype_to_str(output)

        return output

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def update(self, config_dict: Dict[str, Any]):
        """
        Updates attributes of this class with attributes from `config_dict`.

        Args:
            config_dict (`Dict[str, Any]`): Dictionary of attributes that should be updated for this class.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)

    def update_from_string(self, update_str: str):
        """
        Updates attributes of this class with attributes from `update_str`.

        The expected format is ints, floats and strings as is, and for booleans use `true` or `false`. For example:
        "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"

        The keys to change have to already exist in the config object.

        Args:
            update_str (`str`): String with attributes that should be updated for this class.

        """

        d = dict(x.split("=") for x in update_str.split(","))
        for k, v in d.items():
            if not hasattr(self, k):
                raise ValueError(f"key {k} isn't in the original config dict")

            old_v = getattr(self, k)
            if isinstance(old_v, bool):
                if v.lower() in ["true", "1", "y", "yes"]:
                    v = True
                elif v.lower() in ["false", "0", "n", "no"]:
                    v = False
                else:
                    raise ValueError(f"can't derive true or false from {v} (key {k})")
            elif isinstance(old_v, int):
                v = int(v)
            elif isinstance(old_v, float):
                v = float(v)
            elif not isinstance(old_v, str):
                raise TypeError(
                    f"You can only update int, float, bool or string values in the config, got {v} for key {k}"
                )

            setattr(self, k, v)

    def dict_torch_dtype_to_str(self, d: Dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *torch_dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
            d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self.dict_torch_dtype_to_str(value)

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoConfig"):
        """
        Register this class with a given auto class. This should only be used for custom configurations as the ones in
        the library are already mapped with `AutoConfig`.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoConfig"`):
                The auto class to register this new configuration with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        ### voluntarily removed for debugging (should not be used but want to see Error if it is)
        # import transformers.models.auto as auto_module

        # if not hasattr(auto_module, auto_class):
        raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    @staticmethod
    def _get_global_generation_defaults() -> Dict[str, Any]:
        return {
            "max_length": 20,
            "min_length": 0,
            "do_sample": False,
            "early_stopping": False,
            "num_beams": 1,
            "num_beam_groups": 1,
            "diversity_penalty": 0.0,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "typical_p": 1.0,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "encoder_no_repeat_ngram_size": 0,
            "bad_words_ids": None,
            "num_return_sequences": 1,
            "output_scores": False,
            "return_dict_in_generate": False,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "remove_invalid_values": False,
            "exponential_decay_length_penalty": None,
            "suppress_tokens": None,
            "begin_suppress_tokens": None,
        }

    def _get_non_default_generation_parameters(self) -> Dict[str, Any]:
        """
        Gets the non-default generation parameters on the PretrainedConfig instance
        """
        non_default_generation_parameters = {}
        decoder_attribute_name = None

        # Composite models don't have a default config, use their decoder config as a fallback for default values
        # If no known pattern is matched, then `default_config = None` -> check against the global generation defaults
        try:
            default_config = self.__class__()
        except ValueError:
            decoder_config = self.get_text_config(decoder=True)
            if decoder_config is not self:
                default_config = decoder_config.__class__()
            else:
                default_config = None

        # If it is a composite model, we want to check the subconfig that will be used for generation
        self_decoder_config = self if decoder_attribute_name is None else getattr(self, decoder_attribute_name)

        for parameter_name, default_global_value in self._get_global_generation_defaults().items():
            if hasattr(self_decoder_config, parameter_name):
                is_default_in_config = is_default_generation_value = None
                parameter_value = getattr(self_decoder_config, parameter_name)
                # Three cases in which is okay for the model config to hold generation config parameters:
                # 1. The parameter is set to `None`, effectivelly delegating its value to the generation config
                if parameter_value is None:
                    continue
                # 2. If we have a default config, then the instance should hold the same generation defaults
                if default_config is not None:
                    is_default_in_config = parameter_value == getattr(default_config, parameter_name)
                # 3. if we don't have a default config, then the instance should hold the global generation defaults
                else:
                    is_default_generation_value = parameter_value == default_global_value

                is_non_default = (is_default_in_config is False) or (
                    is_default_in_config is None and is_default_generation_value is False
                )
                if is_non_default:
                    non_default_generation_parameters[parameter_name] = getattr(self_decoder_config, parameter_name)

        return non_default_generation_parameters

    def get_text_config(self, decoder=False) -> "PretrainedConfig":
        """
        Returns the config that is meant to be used with text IO. On most models, it is the original config instance
        itself. On specific composite models, it is under a set of valid names.

        If `decoder` is set to `True`, then only search for decoder config names.
        """
        decoder_possible_text_config_names = ("decoder", "generator", "text_config")
        encoder_possible_text_config_names = ("text_encoder",)
        if decoder:
            possible_text_config_names = decoder_possible_text_config_names
        else:
            possible_text_config_names = encoder_possible_text_config_names + decoder_possible_text_config_names

        valid_text_config_names = []
        for text_config_name in possible_text_config_names:
            if hasattr(self, text_config_name):
                text_config = getattr(self, text_config_name, None)
                if text_config is not None:
                    valid_text_config_names += [text_config_name]

        if len(valid_text_config_names) > 1:
            raise ValueError(
                f"Multiple valid text configs were found in the model config: {valid_text_config_names}. In this "
                "case, using `get_text_config()` would be ambiguous. Please specify the desied text config directly."
            )
        elif len(valid_text_config_names) == 1:
            return getattr(self, valid_text_config_names[0])
        return self


def get_configuration_file(configuration_files: List[str]) -> str:
    """
    Get the configuration file to use for this version of transformers.

    Args:
        configuration_files (`List[str]`): The list of available configuration files.

    Returns:
        `str`: The configuration file to use.
    """
    configuration_files_map = {}
    for file_name in configuration_files:
        search = _re_configuration_file.search(file_name)
        if search is not None:
            v = search.groups()[0]
            configuration_files_map[v] = file_name
    available_versions = sorted(configuration_files_map.keys())

    # Defaults to FULL_CONFIGURATION_FILE and then try to look at some newer versions.
    configuration_file = CONFIG_NAME
    transformers_version = version.parse(__version__)
    for v in available_versions:
        if version.parse(v) <= transformers_version:
            configuration_file = configuration_files_map[v]
        else:
            # No point going further since the versions are sorted.
            break

    return configuration_file


def recursive_diff_dict(dict_a, dict_b, config_obj=None):
    """
    Helper function to recursively take the diff between two nested dictionaries. The resulting diff only contains the
    values from `dict_a` that are different from values in `dict_b`.
    """
    diff = {}
    default = config_obj.__class__().to_dict() if config_obj is not None else {}
    for key, value in dict_a.items():
        obj_value = getattr(config_obj, str(key), None)
        if isinstance(obj_value, PretrainedConfig) and key in dict_b and isinstance(dict_b[key], dict):
            diff_value = recursive_diff_dict(value, dict_b[key], config_obj=obj_value)
            if len(diff_value) > 0:
                diff[key] = diff_value
        elif key not in dict_b or value != dict_b[key] or key not in default or value != default[key]:
            diff[key] = value
    return diff


PretrainedConfig.push_to_hub = copy_func(PretrainedConfig.push_to_hub)
if PretrainedConfig.push_to_hub.__doc__ is not None:
    PretrainedConfig.push_to_hub.__doc__ = PretrainedConfig.push_to_hub.__doc__.format(
        object="config", object_class="AutoConfig", object_files="configuration file"
    )


CLASS_DOCSTRING = """
    This is a generic model class that will be instantiated as one of the model classes of the library when created
    with the [`~BaseAutoModelClass.from_pretrained`] class method or the [`~BaseAutoModelClass.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
"""

FROM_CONFIG_DOCSTRING = """
        Instantiates one of the model classes of the library from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights. It only affects the
            model's configuration. Use [`~BaseAutoModelClass.from_pretrained`] to load the model weights.

        Args:
            config ([`PretrainedConfig`]):
                The model class to instantiate is selected based on the configuration class:

                List options
            attn_implementation (`str`, *optional*):
                The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

        Examples:

        ```python
        >>> from transformers import AutoConfig, BaseAutoModelClass

        >>> # Download configuration from huggingface.co and cache.
        >>> config = AutoConfig.from_pretrained("checkpoint_placeholder")
        >>> model = BaseAutoModelClass.from_config(config)
        ```
"""

FROM_PRETRAINED_TORCH_DOCSTRING = """
        Instantiate one of the model classes of the library from a pretrained model.

        The model class to instantiate is selected based on the `model_type` property of the config object (either
        passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it's missing, by
        falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
        deactivated). To train the model, you should first set it back in training mode with `model.train()`

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
            model_args (additional positional arguments, *optional*):
                Will be passed along to the underlying model `__init__()` method.
            config ([`PretrainedConfig`], *optional*):
                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            state_dict (*Dict[str, torch.Tensor]*, *optional*):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.

                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and
                [`~PreTrainedModel.from_pretrained`] is not a simpler option.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_tf (`bool`, *optional*, defaults to `False`):
                Load the model weights from a TensorFlow checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (e.g., not try downloading the model).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            code_revision (`str`, *optional*, defaults to `"main"`):
                The specific revision to use for the code on the Hub, if the code leaves in a different repository than
                the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
                system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
                allowed by git.
            kwargs (additional keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
                      corresponds to a configuration attribute will be used to override said attribute with the
                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                      will be passed to the underlying model's `__init__` function.

        Examples:

        ```python
        >>> from transformers import AutoConfig, BaseAutoModelClass

        >>> # Download model and configuration from huggingface.co and cache.
        >>> model = BaseAutoModelClass.from_pretrained("checkpoint_placeholder")

        >>> # Update configuration during loading
        >>> model = BaseAutoModelClass.from_pretrained("checkpoint_placeholder", output_attentions=True)
        >>> model.config.output_attentions
        True

        >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
        >>> config = AutoConfig.from_pretrained("./tf_model/shortcut_placeholder_tf_model_config.json")
        >>> model = BaseAutoModelClass.from_pretrained(
        ...     "./tf_model/shortcut_placeholder_tf_checkpoint.ckpt.index", from_tf=True, config=config
        ... )
        ```
"""

FROM_PRETRAINED_TF_DOCSTRING = """
        Instantiate one of the model classes of the library from a pretrained model.

        The model class to instantiate is selected based on the `model_type` property of the config object (either
        passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it's missing, by
        falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *PyTorch state_dict save file* (e.g, `./pt_model/pytorch_model.bin`). In this
                      case, `from_pt` should be set to `True` and a configuration object should be provided as `config`
                      argument. This loading path is slower than converting the PyTorch model in a TensorFlow model
                      using the provided conversion scripts and loading the TensorFlow model afterwards.
            model_args (additional positional arguments, *optional*):
                Will be passed along to the underlying model `__init__()` method.
            config ([`PretrainedConfig`], *optional*):
                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_pt (`bool`, *optional*, defaults to `False`):
                Load the model weights from a PyTorch checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (e.g., not try downloading the model).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            code_revision (`str`, *optional*, defaults to `"main"`):
                The specific revision to use for the code on the Hub, if the code leaves in a different repository than
                the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
                system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
                allowed by git.
            kwargs (additional keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
                      corresponds to a configuration attribute will be used to override said attribute with the
                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                      will be passed to the underlying model's `__init__` function.

        Examples:

        ```python
        >>> from transformers import AutoConfig, BaseAutoModelClass

        >>> # Download model and configuration from huggingface.co and cache.
        >>> model = BaseAutoModelClass.from_pretrained("checkpoint_placeholder")

        >>> # Update configuration during loading
        >>> model = BaseAutoModelClass.from_pretrained("checkpoint_placeholder", output_attentions=True)
        >>> model.config.output_attentions
        True

        >>> # Loading from a PyTorch checkpoint file instead of a TensorFlow model (slower)
        >>> config = AutoConfig.from_pretrained("./pt_model/shortcut_placeholder_pt_model_config.json")
        >>> model = BaseAutoModelClass.from_pretrained(
        ...     "./pt_model/shortcut_placeholder_pytorch_model.bin", from_pt=True, config=config
        ... )
        ```
"""

FROM_PRETRAINED_FLAX_DOCSTRING = """
        Instantiate one of the model classes of the library from a pretrained model.

        The model class to instantiate is selected based on the `model_type` property of the config object (either
        passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it's missing, by
        falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *PyTorch state_dict save file* (e.g, `./pt_model/pytorch_model.bin`). In this
                      case, `from_pt` should be set to `True` and a configuration object should be provided as `config`
                      argument. This loading path is slower than converting the PyTorch model in a TensorFlow model
                      using the provided conversion scripts and loading the TensorFlow model afterwards.
            model_args (additional positional arguments, *optional*):
                Will be passed along to the underlying model `__init__()` method.
            config ([`PretrainedConfig`], *optional*):
                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_pt (`bool`, *optional*, defaults to `False`):
                Load the model weights from a PyTorch checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (e.g., not try downloading the model).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            code_revision (`str`, *optional*, defaults to `"main"`):
                The specific revision to use for the code on the Hub, if the code leaves in a different repository than
                the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
                system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
                allowed by git.
            kwargs (additional keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
                      corresponds to a configuration attribute will be used to override said attribute with the
                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                      will be passed to the underlying model's `__init__` function.

        Examples:

        ```python
        >>> from transformers import AutoConfig, BaseAutoModelClass

        >>> # Download model and configuration from huggingface.co and cache.
        >>> model = BaseAutoModelClass.from_pretrained("checkpoint_placeholder")

        >>> # Update configuration during loading
        >>> model = BaseAutoModelClass.from_pretrained("checkpoint_placeholder", output_attentions=True)
        >>> model.config.output_attentions
        True

        >>> # Loading from a PyTorch checkpoint file instead of a TensorFlow model (slower)
        >>> config = AutoConfig.from_pretrained("./pt_model/shortcut_placeholder_pt_model_config.json")
        >>> model = BaseAutoModelClass.from_pretrained(
        ...     "./pt_model/shortcut_placeholder_pytorch_model.bin", from_pt=True, config=config
        ... )
        ```
"""


def _get_model_class(config, model_mapping):
    supported_models = model_mapping[type(config)]
    if not isinstance(supported_models, (list, tuple)):
        return supported_models

    name_to_model = {model.__name__: model for model in supported_models}
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in name_to_model:
            return name_to_model[arch]
        elif f"TF{arch}" in name_to_model:
            return name_to_model[f"TF{arch}"]
        elif f"Flax{arch}" in name_to_model:
            return name_to_model[f"Flax{arch}"]

    # If not architecture is set in the config or match the supported models, the first element of the tuple is the
    # defaults.
    return supported_models[0]


class _BaseAutoModelClass:
    # Base class for auto models.
    _model_mapping = None

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
        has_local_code = type(config) in cls._model_mapping.keys()
        ### trust_remote_code should not be True in the scope of this project (removed for dependancies)
        trust_remote_code = False ### replacing the commented code below
        # trust_remote_code = resolve_trust_remote_code(
        #     trust_remote_code, config._name_or_path, has_local_code, has_remote_code
        # )

        # if has_remote_code and trust_remote_code:
        #     class_ref = config.auto_map[cls.__name__]
        #     if "--" in class_ref:
        #         repo_id, class_ref = class_ref.split("--")
        #     else:
        #         repo_id = config.name_or_path
        #     model_class = get_class_from_dynamic_module(class_ref, repo_id, **kwargs)
        #     cls.register(config.__class__, model_class, exist_ok=True)
        #     _ = kwargs.pop("code_revision", None)
        #     model_class = add_generation_mixin_to_remote_model(model_class)
        #     return model_class._from_config(config, **kwargs)
        # el
        if type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class._from_config(config, **kwargs)

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        kwargs["_from_auto"] = True
        hub_kwargs_names = [
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "resume_download",
            "revision",
            "subfolder",
            "use_auth_token",
            "token",
        ]
        hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}
        code_revision = kwargs.pop("code_revision", None)
        commit_hash = kwargs.pop("_commit_hash", None)
        adapter_kwargs = kwargs.pop("adapter_kwargs", None)

        token = hub_kwargs.pop("token", None)
        use_auth_token = hub_kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            hub_kwargs["token"] = token

        if commit_hash is None:
            if not isinstance(config, PretrainedConfig):
                # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    CONFIG_NAME,
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                    **hub_kwargs,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            else:
                commit_hash = getattr(config, "_commit_hash", None)

        ### out of scope for this project
        # if is_peft_available():
        #     if adapter_kwargs is None:
        #         adapter_kwargs = {}
        #         if token is not None:
        #             adapter_kwargs["token"] = token

        #     maybe_adapter_path = find_adapter_config_file(
        #         pretrained_model_name_or_path, _commit_hash=commit_hash, **adapter_kwargs
        #     )

        #     if maybe_adapter_path is not None:
        #         with open(maybe_adapter_path, "r", encoding="utf-8") as f:
        #             adapter_config = json.load(f)

        #             adapter_kwargs["_adapter_model_path"] = pretrained_model_name_or_path
        #             pretrained_model_name_or_path = adapter_config["base_model_name_or_path"]

        if not isinstance(config, PretrainedConfig):
            kwargs_orig = copy.deepcopy(kwargs)
            # ensure not to pollute the config object with torch_dtype="auto" - since it's
            # meaningless in the context of the config object - torch.dtype values are acceptable
            if kwargs.get("torch_dtype", None) == "auto":
                _ = kwargs.pop("torch_dtype")
            # to not overwrite the quantization_config if config has a quantization_config
            if kwargs.get("quantization_config", None) is not None:
                _ = kwargs.pop("quantization_config")

            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                trust_remote_code=trust_remote_code,
                code_revision=code_revision,
                _commit_hash=commit_hash,
                **hub_kwargs,
                **kwargs,
            )

            # if torch_dtype=auto was passed here, ensure to pass it on
            if kwargs_orig.get("torch_dtype", None) == "auto":
                kwargs["torch_dtype"] = "auto"
            if kwargs_orig.get("quantization_config", None) is not None:
                kwargs["quantization_config"] = kwargs_orig["quantization_config"]

        has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
        has_local_code = type(config) in cls._model_mapping.keys()
        ### trust_remote_code should not be True in the scope of this project (removed for dependancies)
        trust_remote_code = False ### replacing the commented code below
        # trust_remote_code = resolve_trust_remote_code(
        #     trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        # )

        # Set the adapter kwargs
        kwargs["adapter_kwargs"] = adapter_kwargs
        ### should not happen though the MLX code in the scope of this project (see above)
        # if has_remote_code and trust_remote_code:
        #     class_ref = config.auto_map[cls.__name__]
        #     model_class = get_class_from_dynamic_module(
        #         class_ref, pretrained_model_name_or_path, code_revision=code_revision, **hub_kwargs, **kwargs
        #     )
        #     _ = hub_kwargs.pop("code_revision", None)
        #     cls.register(config.__class__, model_class, exist_ok=True)
        #     model_class = add_generation_mixin_to_remote_model(model_class)
        #     return model_class.from_pretrained(
        #         pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
        #     )
        # el
        if type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    @classmethod
    def register(cls, config_class, model_class, exist_ok=False):
        """
        Register a new model for this class.

        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            model_class ([`PreTrainedModel`]):
                The model to register.
        """
        if hasattr(model_class, "config_class") and str(model_class.config_class) != str(config_class):
            raise ValueError(
                "The model class you are passing has a `config_class` attribute that is not consistent with the "
                f"config class you passed (model has {model_class.config_class} and you passed {config_class}. Fix "
                "one of those so they match!"
            )
        cls._model_mapping.register(config_class, model_class, exist_ok=exist_ok)


class _BaseAutoBackboneClass(_BaseAutoModelClass):
    # Base class for auto backbone models.
    _model_mapping = None

    @classmethod
    def _load_timm_backbone_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        ### no reason to happen in the current codebase but want to see the error message
        raise ValueError("This method should not be called directly.")
    
        # requires_backends(cls, ["vision", "timm"])
        # from ...models.timm_backbone import TimmBackboneConfig

        # config = kwargs.pop("config", TimmBackboneConfig())

        # if kwargs.get("out_features", None) is not None:
        #     raise ValueError("Cannot specify `out_features` for timm backbones")

        # if kwargs.get("output_loading_info", False):
        #     raise ValueError("Cannot specify `output_loading_info=True` when loading from timm")

        # num_channels = kwargs.pop("num_channels", config.num_channels)
        # features_only = kwargs.pop("features_only", config.features_only)
        # use_pretrained_backbone = kwargs.pop("use_pretrained_backbone", config.use_pretrained_backbone)
        # out_indices = kwargs.pop("out_indices", config.out_indices)
        # config = TimmBackboneConfig(
        #     backbone=pretrained_model_name_or_path,
        #     num_channels=num_channels,
        #     features_only=features_only,
        #     use_pretrained_backbone=use_pretrained_backbone,
        #     out_indices=out_indices,
        # )
        # return super().from_config(config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        use_timm_backbone = kwargs.pop("use_timm_backbone", False)
        if use_timm_backbone:
            return cls._load_timm_backbone_from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


def insert_head_doc(docstring, head_doc=""):
    if len(head_doc) > 0:
        return docstring.replace(
            "one of the model classes of the library ",
            f"one of the model classes of the library (with a {head_doc} head) ",
        )
    return docstring.replace(
        "one of the model classes of the library ", "one of the base model classes of the library "
    )


def auto_class_update(cls, checkpoint_for_example="google-bert/bert-base-cased", head_doc=""):
    # Create a new class with the right name from the base class
    model_mapping = cls._model_mapping
    name = cls.__name__
    class_docstring = insert_head_doc(CLASS_DOCSTRING, head_doc=head_doc)
    cls.__doc__ = class_docstring.replace("BaseAutoModelClass", name)

    # Now we need to copy and re-register `from_config` and `from_pretrained` as class methods otherwise we can't
    # have a specific docstrings for them.
    from_config = copy_func(_BaseAutoModelClass.from_config)
    from_config_docstring = insert_head_doc(FROM_CONFIG_DOCSTRING, head_doc=head_doc)
    from_config_docstring = from_config_docstring.replace("BaseAutoModelClass", name)
    from_config_docstring = from_config_docstring.replace("checkpoint_placeholder", checkpoint_for_example)
    from_config.__doc__ = from_config_docstring
    from_config = replace_list_option_in_docstrings(model_mapping._model_mapping, use_model_types=False)(from_config)
    cls.from_config = classmethod(from_config)

    if name.startswith("TF"):
        from_pretrained_docstring = FROM_PRETRAINED_TF_DOCSTRING
    elif name.startswith("Flax"):
        from_pretrained_docstring = FROM_PRETRAINED_FLAX_DOCSTRING
    else:
        from_pretrained_docstring = FROM_PRETRAINED_TORCH_DOCSTRING
    from_pretrained = copy_func(_BaseAutoModelClass.from_pretrained)
    from_pretrained_docstring = insert_head_doc(from_pretrained_docstring, head_doc=head_doc)
    from_pretrained_docstring = from_pretrained_docstring.replace("BaseAutoModelClass", name)
    from_pretrained_docstring = from_pretrained_docstring.replace("checkpoint_placeholder", checkpoint_for_example)
    shortcut = checkpoint_for_example.split("/")[-1].split("-")[0]
    from_pretrained_docstring = from_pretrained_docstring.replace("shortcut_placeholder", shortcut)
    from_pretrained.__doc__ = from_pretrained_docstring
    from_pretrained = replace_list_option_in_docstrings(model_mapping._model_mapping)(from_pretrained)
    cls.from_pretrained = classmethod(from_pretrained)
    return cls


def get_values(model_mapping):
    result = []
    for model in model_mapping.values():
        if isinstance(model, (list, tuple)):
            result += list(model)
        else:
            result.append(model)

    return result


def getattribute_from_module(module, attr):
    if attr is None:
        return None
    if isinstance(attr, tuple):
        return tuple(getattribute_from_module(module, a) for a in attr)
    if hasattr(module, attr):
        return getattr(module, attr)
    # Some of the mappings have entries model_type -> object of another model type. In that case we try to grab the
    # object at the top level.

    ### no reason to happen in the current codebase but want to see the error message
    # transformers_module = importlib.import_module("transformers")

    # if module != transformers_module:
    #     try:
    #         return getattribute_from_module(transformers_module, attr)
    #     except ValueError:
    #         raise ValueError(f"Could not find {attr} neither in {module} nor in {transformers_module}!")
    # else:
    raise ValueError(f"Could not find {attr} !")


def add_generation_mixin_to_remote_model(model_class):
    """
    Adds `GenerationMixin` to the inheritance of `model_class`, if `model_class` is a PyTorch model.

    This function is used for backwards compatibility purposes: in v4.45, we've started a deprecation cycle to make
    `PreTrainedModel` stop inheriting from `GenerationMixin`. Without this function, older models dynamically loaded
    from the Hub may not have the `generate` method after we remove the inheritance.
    """
    # 1. If it is not a PT model (i.e. doesn't inherit Module), do nothing
    if "torch.nn.modules.module.Module" not in str(model_class.__mro__):
        return model_class

    # 2. If it already **directly** inherits from GenerationMixin, do nothing
    if "GenerationMixin" in str(model_class.__bases__):
        return model_class

    # 3. Prior to v4.45, we could detect whether a model was `generate`-compatible if it had its own `generate` and/or
    # `prepare_inputs_for_generation` method.
    has_custom_generate = "GenerationMixin" not in str(getattr(model_class, "generate"))
    has_custom_prepare_inputs = "GenerationMixin" not in str(getattr(model_class, "prepare_inputs_for_generation"))
    if has_custom_generate or has_custom_prepare_inputs:
        ### no reason to happen in the current codebase but want to see the error message
        raise ValueError(
            f"Model {model_class.__name__} has its own `generate` and/or `prepare_inputs_for_generation` method. "
            "This method is used to add the `generate` method to the model. Please remove these methods from your model."
        )
        # model_class_with_generation_mixin = type(
        #     model_class.__name__, (model_class, GenerationMixin), {**model_class.__dict__}
        # )
        # return model_class_with_generation_mixin
    return model_class

### pasted here from transformers/models/auto/auto_factory.py
class _LazyAutoMapping(OrderedDict):
    """
    " A mapping config to object (model or tokenizer for instance) that will load keys and values when it is accessed.

    Args:
        - config_mapping: The map model type to config class
        - model_mapping: The map model type to model (or tokenizer) class
    """

    def __init__(self, config_mapping, model_mapping):
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._model_mapping = model_mapping
        self._model_mapping._model_mapping = self
        self._extra_content = {}
        self._modules = {}

    def __len__(self):
        common_keys = set(self._config_mapping.keys()).intersection(self._model_mapping.keys())
        return len(common_keys) + len(self._extra_content)

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]
        model_type = self._reverse_config_mapping[key.__name__]
        if model_type in self._model_mapping:
            model_name = self._model_mapping[model_type]
            return self._load_attr_from_module(model_type, model_name)

        # Maybe there was several model types associated with this config.
        model_types = [k for k, v in self._config_mapping.items() if v == key.__name__]
        for mtype in model_types:
            if mtype in self._model_mapping:
                model_name = self._model_mapping[mtype]
                return self._load_attr_from_module(mtype, model_name)
        raise KeyError(key)

    def _load_attr_from_module(self, model_type, attr):
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        return getattribute_from_module(self._modules[module_name], attr)

    def keys(self):
        mapping_keys = [
            self._load_attr_from_module(key, name)
            for key, name in self._config_mapping.items()
            if key in self._model_mapping.keys()
        ]
        return mapping_keys + list(self._extra_content.keys())

    def get(self, key, default):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __bool__(self):
        return bool(self.keys())

    def values(self):
        mapping_values = [
            self._load_attr_from_module(key, name)
            for key, name in self._model_mapping.items()
            if key in self._config_mapping.keys()
        ]
        return mapping_values + list(self._extra_content.values())

    def items(self):
        mapping_items = [
            (
                self._load_attr_from_module(key, self._config_mapping[key]),
                self._load_attr_from_module(key, self._model_mapping[key]),
            )
            for key in self._model_mapping.keys()
            if key in self._config_mapping.keys()
        ]
        return mapping_items + list(self._extra_content.items())

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item):
        if item in self._extra_content:
            return True
        if not hasattr(item, "__name__") or item.__name__ not in self._reverse_config_mapping:
            return False
        model_type = self._reverse_config_mapping[item.__name__]
        return model_type in self._model_mapping

    def register(self, key, value, exist_ok=False):
        """
        Register a new model in this mapping.
        """
        if hasattr(key, "__name__") and key.__name__ in self._reverse_config_mapping:
            model_type = self._reverse_config_mapping[key.__name__]
            if model_type in self._model_mapping.keys() and not exist_ok:
                raise ValueError(f"'{key}' is already used by a Transformers model.")

        self._extra_content[key] = value