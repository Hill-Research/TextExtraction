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

from typing import Any, Dict, List, Union

from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
from .base import PIPELINE_INIT_ARGS, ChunkPipeline


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_torch_available():
    import torch

    from transformers.modeling_outputs import BaseModelOutput

    from ..models.auto.modeling_auto import MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING

logger = logging.get_logger(__name__)


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ZeroShotObjectDetectionPipeline(ChunkPipeline):
    """
    Zero shot object detection pipeline using `OwlViTForObjectDetection`. This pipeline predicts bounding boxes of
    objects when you provide an image and a set of `candidate_labels`.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")
    >>> detector(
    ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
    ...     candidate_labels=["cat", "couch"],
    ... )
    [{'score': 0.287, 'label': 'cat', 'box': {'xmin': 324, 'ymin': 20, 'xmax': 640, 'ymax': 373}}, {'score': 0.254, 'label': 'cat', 'box': {'xmin': 1, 'ymin': 55, 'xmax': 315, 'ymax': 472}}, {'score': 0.121, 'label': 'couch', 'box': {'xmin': 4, 'ymin': 0, 'xmax': 642, 'ymax': 476}}]

    >>> detector(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    ...     candidate_labels=["head", "bird"],
    ... )
    [{'score': 0.119, 'label': 'bird', 'box': {'xmin': 71, 'ymin': 170, 'xmax': 410, 'ymax': 508}}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This object detection pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"zero-shot-object-detection"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=zero-shot-object-detection).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        requires_backends(self, "vision")
        self.check_model_type(MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING)

    def __call__(
        self,
        image: Union[str, "Image.Image", List[Dict[str, Any]]],
        candidate_labels: Union[str, List[str]] = None,
        **kwargs
    ):
        """
        Detect objects (bounding boxes & classes) in the image(s) passed as inputs.

        Args:
            image (`str`, `PIL.Image` or `List[Dict[str, Any]]`):
                The pipeline handles three types of images:

                - A string containing an http url pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                You can use this parameter to send directly a list of images, or a dataset or a generator like so:

                ```python
                >>> from transformers import pipeline

                >>> detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")
                >>> detector(
                ...     [
                ...         {
                ...             "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
                ...             "candidate_labels": ["cat", "couch"],
                ...         },
                ...         {
                ...             "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
                ...             "candidate_labels": ["cat", "couch"],
                ...         },
                ...     ]
                ... )
                [[{'score': 0.286811888217926, 'label': 'cat', 'box': {'xmin': 324, 'ymin': 20, 'xmax': 640, 'ymax': 373}}, {'score': 0.2537279725074768, 'label': 'cat', 'box': {'xmin': 1, 'ymin': 55, 'xmax': 315, 'ymax': 472}}, {'score': 0.12082888185977936, 'label': 'couch', 'box': {'xmin': 4, 'ymin': 0, 'xmax': 642, 'ymax': 476}}], [{'score': 0.286811888217926, 'label': 'cat', 'box': {'xmin': 324, 'ymin': 20, 'xmax': 640, 'ymax': 373}}, {'score': 0.2537279725074768, 'label': 'cat', 'box': {'xmin': 1, 'ymin': 55, 'xmax': 315, 'ymax': 472}}, {'score': 0.12082888185977936, 'label': 'couch', 'box': {'xmin': 4, 'ymin': 0, 'xmax': 642, 'ymax': 476}}]]
                ```


            candidate_labels (`str` or `List[str]` or `List[List[str]]`):
                What the model should recognize in the image.

            threshold (`float`, *optional*, defaults to 0.1):
                The probability necessary to make a prediction.

            top_k (`int`, *optional*, defaults to None):
                The number of top predictions that will be returned by the pipeline. If the provided number is `None`
                or higher than the number of predictions available, it will default to the number of predictions.


        Return:
            A list of lists containing prediction results, one list per input image. Each list contains dictionaries
            with the following keys:

            - **label** (`str`) -- Text query corresponding to the found object.
            - **score** (`float`) -- Score corresponding to the object (between 0 and 1).
            - **box** (`Dict[str,int]`) -- Bounding box of the detected object in image's original size. It is a
              dictionary with `x_min`, `x_max`, `y_min`, `y_max` keys.
        """
        if "text_queries" in kwargs:
            candidate_labels = kwargs.pop("text_queries")

        if isinstance(image, (str, Image.Image)):
            inputs = {"image": image, "candidate_labels": candidate_labels}
        else:
            inputs = image
        results = super().__call__(inputs, **kwargs)
        return results

    def _sanitize_parameters(self, **kwargs):
        postprocess_params = {}
        if "threshold" in kwargs:
            postprocess_params["threshold"] = kwargs["threshold"]
        if "top_k" in kwargs:
            postprocess_params["top_k"] = kwargs["top_k"]
        return {}, {}, postprocess_params

    def preprocess(self, inputs):
        image = load_image(inputs["image"])
        candidate_labels = inputs["candidate_labels"]
        if isinstance(candidate_labels, str):
            candidate_labels = candidate_labels.split(",")

        target_size = torch.tensor([[image.height, image.width]], dtype=torch.int32)
        for i, candidate_label in enumerate(candidate_labels):
            text_inputs = self.tokenizer(candidate_label, return_tensors=self.framework)
            image_features = self.feature_extractor(image, return_tensors=self.framework)
            yield {
                "is_last": i == len(candidate_labels) - 1,
                "target_size": target_size,
                "candidate_label": candidate_label,
                **text_inputs,
                **image_features,
            }

    def _forward(self, model_inputs):
        target_size = model_inputs.pop("target_size")
        candidate_label = model_inputs.pop("candidate_label")
        is_last = model_inputs.pop("is_last")

        outputs = self.model(**model_inputs)

        model_outputs = {"target_size": target_size, "candidate_label": candidate_label, "is_last": is_last, **outputs}
        return model_outputs

    def postprocess(self, model_outputs, threshold=0.1, top_k=None):

        results = []
        for model_output in model_outputs:
            label = model_output["candidate_label"]
            model_output = BaseModelOutput(model_output)
            outputs = self.feature_extractor.post_process(
                outputs=model_output, target_sizes=model_output["target_size"]
            )[0]
            keep = outputs["scores"] >= threshold

            for index in keep.nonzero():
                score = outputs["scores"][index].item()
                box = self._get_bounding_box(outputs["boxes"][index][0])

                result = {"score": score, "label": label, "box": box}
                results.append(result)

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        if top_k:
            results = results[:top_k]

        return results

    def _get_bounding_box(self, box: "torch.Tensor") -> Dict[str, int]:
        """
        Turns list [xmin, xmax, ymin, ymax] into dict { "xmin": xmin, ... }

        Args:
            box (`torch.Tensor`): Tensor containing the coordinates in corners format.

        Returns:
            bbox (`Dict[str, int]`): Dict containing the coordinates in corners format.
        """
        if self.framework != "pt":
            raise ValueError("The ZeroShotObjectDetectionPipeline is only available in PyTorch.")
        xmin, ymin, xmax, ymax = box.int().tolist()
        bbox = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }
        return bbox
