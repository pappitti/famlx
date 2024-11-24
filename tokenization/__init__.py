# Copyright 2020 The HuggingFace Team. All rights reserved.
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

# When adding a new object to this init, remember to add it twice: once inside the `_import_structure` dictionary and
# once inside the `if TYPE_CHECKING` branch. The `TYPE_CHECKING` should have import statements as usual, but they are
# only there for type checking. The `_import_structure` is a dictionary submodule to list of object names, and is used
# to defer the actual importing for when the objects are requested. This way `import transformers` provides the names
# in the namespace without actually importing anything (and especially none of the backends).

from typing import TYPE_CHECKING
from .utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
)

_import_structure = {
    "tokenization_auto": ["TOKENIZER_MAPPING", "AutoTokenizer"],
    "tokenization_utils": ["PreTrainedTokenizer", "PreTrainedTokenizerFast"],
    "tokenization_utils_base": [
        "AddedToken",
        "BatchEncoding",
        "CharSpan",
        "PreTrainedTokenizerBase",
        "SpecialTokensMixin",
        "TokenSpan",
    ],
    "utils": [
        "CONFIG_NAME",
        "TRANSFORMERS_CACHE",
        "TensorType",
        "add_end_docstrings",
        "is_sentencepiece_available",
        "is_tokenizers_available",
        "get_logger",
    ],
    "models.qwen2": ["Qwen2Tokenizer"],
    "models.gpt2": ["GPT2Tokenizer"],
}  

# sentencepiece-backed objects
try:
    if not is_sentencepiece_available():  # should always be true using uv sync
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["models.llama"] = ["LlamaTokenizer"]
    _import_structure["models.gemma2"] = ["GemmaTokenizer"]
    _import_structure["models.deberta_v2"] = ["DebertaV2Tokenizer"]

# tokenizers-backed objects
try:
    if not is_tokenizers_available(): # should always be true using uv sync
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["models.llama"].append("LlamaTokenizerFast")
    _import_structure["models.gemma2"].append("GemmaTokenizerFast")
    _import_structure["models.gpt2"].append("GPT2TokenizerFast")
    _import_structure["models.qwen2"].append("Qwen2TokenizerFast")
    _import_structure["models.deberta_v2"].append("DebertaV2TokenizerFast")

try:
    if not (is_sentencepiece_available() and is_tokenizers_available()): # should always be true using uv sync
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["utils"].append("SLOW_TO_FAST_CONVERTERS")
    _import_structure["utils"].append("convert_slow_tokenizer")

# Direct imports for type-checking
if TYPE_CHECKING:
    from .tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
    from .tokenization_utils import PreTrainedTokenizer, PreTrainedTokenizerFast
    from .tokenization_utils_base import (
        AddedToken,
        BatchEncoding,
        CharSpan,
        PreTrainedTokenizerBase,
        SpecialTokensMixin,
        TokenSpan,
    )
    from .utils import (
        CONFIG_NAME,
        TRANSFORMERS_CACHE,
        TensorType,
        add_end_docstrings,
        is_sentencepiece_available,
        is_tokenizers_available,
        get_logger,
    )
    from .models.qwen2 import Qwen2Tokenizer
    from .models.gpt2 import GPT2Tokenizer

    try:
        if not is_sentencepiece_available():  # should always be true using uv sync
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .models.llama import LlamaTokenizer
        from .models.gemma2 import GemmaTokenizer
        from .models.deberta_v2 import DebertaV2Tokenizer
    
    try:
        if not is_tokenizers_available():  # should always be true using uv sync
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .models.llama import LlamaTokenizerFast
        from .models.gemma2 import GemmaTokenizerFast
        from .models.gpt2 import GPT2TokenizerFast
        from .models.qwen2 import Qwen2TokenizerFast
        from .models.deberta_v2 import DebertaV2TokenizerFast

    try:
        if not (is_sentencepiece_available() and is_tokenizers_available()):  # should always be true using uv sync
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .utils import SLOW_TO_FAST_CONVERTERS, convert_slow_tokenizer

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)