
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

from PIL import Image, UnidentifiedImageError


class QualityPipeline:

    def __init__(self, config_path="config.yaml", session=None):

        self.config_path = Path(config_path).resolve()

        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        self.base_dir = self.config_path.parent

        model_relative_path = Path(
            self.config["model"]["path"]
        )

        self.model_path = self.base_dir / model_relative_path

        if session is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {self.model_path}"
                )

            options = ort.SessionOptions()

            options.intra_op_num_threads = self.config[
                "deployment"
            ]["intra_op_threads"]

            options.inter_op_num_threads = self.config[
                "deployment"
            ]["inter_op_threads"]

            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=[
                    self.config["deployment"]["provider"]
                ]
            )
        else:
            self.session = session

        self.input_name = self.config["model"]["input_name"]
        self.output_name = self.config["model"]["output_name"]

        self.threshold = float(
            self.config["classification"]["threshold"]
        )

        self.image_size = self.config["preprocessing"]["image_size"]

        self.mean = np.array(
            self.config["preprocessing"]["mean"],
            dtype=np.float32
        ).reshape(1, 1, 3)

        self.std = np.array(
            self.config["preprocessing"]["std"],
            dtype=np.float32
        ).reshape(1, 1, 3)

        if not 0 < self.threshold < 1:
            raise ValueError("Threshold must be between 0 and 1.")

    def preprocess(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            image = image.resize(
                (self.image_size, self.image_size),
                Image.Resampling.BILINEAR
            )

            image_array = np.asarray(
                image,
                dtype=np.float32
            ) / 255.0

        image_array = (image_array - self.mean) / self.std

        # HWC -> CHW
        image_array = np.transpose(
            image_array,
            (2, 0, 1)
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        return image_array.astype(np.float32)

    def predict_probability(self, image_array):

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: image_array}
        )

        logit = float(
            np.asarray(outputs[0]).reshape(-1)[0]
        )

        # Numerically stable sigmoid
        if logit >= 0:
            probability_bad = 1.0 / (
                1.0 + np.exp(-logit)
            )
        else:
            exp_value = np.exp(logit)
            probability_bad = exp_value / (1.0 + exp_value)

        return float(probability_bad)

    def error_response(self, file_path, reason, latency_ms=0):

        return {
            "file": str(file_path),
            "status": "error",
            "quality": None,
            "quality_confidence": None,
            "document_type": None,
            "document_type_confidence": None,
            "reason": str(reason),
            "model_versions": {
                "quality": self.config["model"]["version"],
                "document_type": None
            },
            "latency_ms": round(float(latency_ms), 3),
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat()
        }

    def predict(self, image_path):

        start = time.perf_counter()

        image_path = Path(image_path)

        try:
            # Reject non-image extensions
            allowed_extensions = {
                ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
            }

            if image_path.suffix.lower() not in allowed_extensions:
                raise ValueError("Unsupported image file format.")

            image_array = self.preprocess(image_path)

            probability_bad = self.predict_probability(
                image_array
            )

            is_bad = probability_bad >= self.threshold

            quality = "bad" if is_bad else "good"

            confidence = (
                probability_bad if is_bad
                else 1.0 - probability_bad
            )

            status = "rejected" if is_bad else "accepted"

            reason = (
                f"Bad-quality probability ({probability_bad:.4f}) "
                f"is at or above threshold ({self.threshold:.4f})."
                if is_bad
                else
                f"Bad-quality probability ({probability_bad:.4f}) "
                f"is below threshold ({self.threshold:.4f})."
            )

            latency_ms = (
                time.perf_counter() - start
            ) * 1000

            return {
                "file": str(image_path),
                "status": status,
                "quality": quality,
                "quality_confidence": round(
                    float(confidence), 6
                ),
                "document_type": None,
                "document_type_confidence": None,
                "reason": reason,
                "model_versions": {
                    "quality": self.config["model"]["version"],
                    "document_type": None
                },
                "latency_ms": round(float(latency_ms), 3),
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat()
            }

        except (
            FileNotFoundError,
            ValueError,
            UnidentifiedImageError,
            OSError,
            RuntimeError
        ) as error:

            latency_ms = (
                time.perf_counter() - start
            ) * 1000

            return self.error_response(
                image_path,
                error,
                latency_ms
            )

        except Exception as error:

            latency_ms = (
                time.perf_counter() - start
            ) * 1000

            return self.error_response(
                image_path,
                f"Unexpected pipeline error: {error}",
                latency_ms
            )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("image")

    parser.add_argument(
        "--config",
        default="config.yaml"
    )

    args = parser.parse_args()

    pipeline = QualityPipeline(args.config)

    result = pipeline.predict(args.image)

    print(json.dumps(result, indent=4))
