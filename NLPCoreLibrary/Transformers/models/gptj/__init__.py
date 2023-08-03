#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor,
#  Boston, MA  02110-1301, USA.

# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
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
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)


_import_structure = {"configuration_gptj": ["GPTJ_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTJConfig", "GPTJOnnxConfig"]}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_gptj"] = [
        "GPTJ_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GPTJForCausalLM",
        "GPTJForQuestionAnswering",
        "GPTJForSequenceClassification",
        "GPTJModel",
        "GPTJPreTrainedModel",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_gptj"] = [
        "TFGPTJForCausalLM",
        "TFGPTJForQuestionAnswering",
        "TFGPTJForSequenceClassification",
        "TFGPTJModel",
        "TFGPTJPreTrainedModel",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_gptj"] = [
        "FlaxGPTJForCausalLM",
        "FlaxGPTJModel",
        "FlaxGPTJPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_gptj import GPTJ_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTJConfig, GPTJOnnxConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_gptj import (
            GPTJ_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPTJForCausalLM,
            GPTJForQuestionAnswering,
            GPTJForSequenceClassification,
            GPTJModel,
            GPTJPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_gptj import (
            TFGPTJForCausalLM,
            TFGPTJForQuestionAnswering,
            TFGPTJForSequenceClassification,
            TFGPTJModel,
            TFGPTJPreTrainedModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_gptj import FlaxGPTJForCausalLM, FlaxGPTJModel, FlaxGPTJPreTrainedModel

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
