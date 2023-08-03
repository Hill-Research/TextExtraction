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

# Copyright 2022 The HuggingFace Team. All rights reserved.
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

# rely on isort to merge the imports
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_torch_available,
    is_vision_available,
)


_import_structure = {
    "configuration_convnext": ["CONVNEXT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ConvNextConfig", "ConvNextOnnxConfig"]
}

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_convnext"] = ["ConvNextFeatureExtractor"]
    _import_structure["image_processing_convnext"] = ["ConvNextImageProcessor"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_convnext"] = [
        "CONVNEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ConvNextForImageClassification",
        "ConvNextModel",
        "ConvNextPreTrainedModel",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_convnext"] = [
        "TFConvNextForImageClassification",
        "TFConvNextModel",
        "TFConvNextPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_convnext import CONVNEXT_PRETRAINED_CONFIG_ARCHIVE_MAP, ConvNextConfig, ConvNextOnnxConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_convnext import ConvNextFeatureExtractor
        from .image_processing_convnext import ConvNextImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_convnext import (
            CONVNEXT_PRETRAINED_MODEL_ARCHIVE_LIST,
            ConvNextForImageClassification,
            ConvNextModel,
            ConvNextPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_convnext import TFConvNextForImageClassification, TFConvNextModel, TFConvNextPreTrainedModel


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
