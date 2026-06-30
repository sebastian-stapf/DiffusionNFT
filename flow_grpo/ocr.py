import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image

try:
    from Levenshtein import distance as _levenshtein_distance
except ImportError:
    _levenshtein_distance = None


def distance(left: str, right: str) -> int:
    if _levenshtein_distance is not None:
        return int(_levenshtein_distance(left, right))
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insert = current[right_index - 1] + 1
            delete = previous[right_index] + 1
            replace = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def _target_text(prompt: str) -> str:
    parts = prompt.split('"')
    if len(parts) >= 3:
        return parts[1]
    return prompt


def _normalize_text(text: str) -> str:
    return text.replace(" ", "").lower()


def _reward_from_text(recognized_text: str, prompt: str) -> float:
    target = _normalize_text(_target_text(prompt))
    if not target:
        return 0.0
    recognized = _normalize_text(recognized_text)
    if target in recognized:
        dist = 0
    else:
        dist = min(distance(recognized, target), len(target))
    return 1.0 - dist / len(target)


def _recognized_text_from_result(result) -> str:
    if not result or not result[0]:
        return ""
    tokens = []
    for row in result[0]:
        try:
            text, confidence = row[1]
        except (TypeError, IndexError, ValueError):
            continue
        if confidence > 0:
            tokens.append(text)
    return "".join(tokens)


def _images_to_numpy(images: Union[List[Image.Image], List[np.ndarray]]) -> list[np.ndarray]:
    arrays = []
    for image in images:
        if isinstance(image, Image.Image):
            arrays.append(np.array(image.convert("RGB")))
        else:
            arrays.append(np.asarray(image))
    return arrays


def _default_backend() -> str:
    requested = os.environ.get("DIFFUSIONNFT_OCR_BACKEND", "auto").lower()
    if requested != "auto":
        return requested
    if sys.version_info >= (3, 12) or platform.machine().lower() in {"aarch64", "arm64"}:
        return "isolated"
    return "paddle"


class OcrScorer:
    def __init__(
        self, use_gpu: bool = False, backend: str | None = None, timeout_s: int | None = None
    ):
        self.use_gpu = use_gpu
        self.backend = (backend or _default_backend()).lower()
        self.timeout_s = int(os.environ.get("DIFFUSIONNFT_OCR_TIMEOUT_S", timeout_s or 180))
        self._ocr = None

    def _ensure_paddle_ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                use_angle_cls=False,
                lang="en",
                use_gpu=self.use_gpu,
                show_log=False,
            )
        return self._ocr

    def _score_in_process(
        self, images: Union[List[Image.Image], List[np.ndarray]], prompts: List[str]
    ) -> list[float]:
        ocr = self._ensure_paddle_ocr()
        rewards = []
        for image, prompt in zip(_images_to_numpy(images), prompts):
            try:
                result = ocr.ocr(image, cls=False)
                recognized_text = _recognized_text_from_result(result)
                rewards.append(_reward_from_text(recognized_text, prompt))
            except Exception as exc:
                print(f"OCR processing failed: {exc}", flush=True)
                rewards.append(0.0)
        return rewards

    def _score_isolated(
        self, images: Union[List[Image.Image], List[np.ndarray]], prompts: List[str]
    ) -> list[float]:
        with tempfile.TemporaryDirectory(prefix="diffusionnft_ocr_") as tmpdir:
            tmp = Path(tmpdir)
            image_paths = []
            for index, image in enumerate(_images_to_numpy(images)):
                path = tmp / f"{index:04d}.png"
                Image.fromarray(image).save(path)
                image_paths.append(str(path))
            request_path = tmp / "request.json"
            response_path = tmp / "response.json"
            request_path.write_text(
                json.dumps(
                    {
                        "image_paths": image_paths,
                        "prompts": prompts,
                        "use_gpu": self.use_gpu,
                    }
                ),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("FLAGS_allocator_strategy", "auto_growth")
            cmd = [
                sys.executable,
                "-m",
                "flow_grpo.ocr",
                "--worker-request",
                str(request_path),
                "--worker-response",
                str(response_path),
            ]
            try:
                completed = subprocess.run(
                    cmd,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                print("Isolated OCR worker timed out; returning zero rewards.", flush=True)
                return [0.0 for _ in prompts]
            if completed.returncode != 0:
                print(
                    "Isolated OCR worker failed; returning zero rewards. "
                    f"returncode={completed.returncode} stderr={completed.stderr[-1000:]}",
                    flush=True,
                )
                return [0.0 for _ in prompts]
            try:
                response = json.loads(response_path.read_text(encoding="utf-8"))
                scores = [float(value) for value in response["scores"]]
            except Exception as exc:
                print(f"Could not parse isolated OCR response: {exc}", flush=True)
                return [0.0 for _ in prompts]
            if len(scores) != len(prompts):
                print("Isolated OCR response length mismatch; returning zero rewards.", flush=True)
                return [0.0 for _ in prompts]
            return scores

    @torch.no_grad()
    def __call__(
        self, images: Union[List[Image.Image], List[np.ndarray]], prompts: List[str]
    ) -> torch.Tensor:
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        if self.backend in {"isolated", "subprocess"}:
            return self._score_isolated(images, prompts)
        if self.backend in {"paddle", "inprocess", "in_process"}:
            return self._score_in_process(images, prompts)
        raise ValueError(f"Unknown OCR backend: {self.backend}")


def _worker_main(request_path: Path, response_path: Path) -> None:
    from paddleocr import PaddleOCR

    request = json.loads(request_path.read_text(encoding="utf-8"))
    ocr = PaddleOCR(
        use_angle_cls=False,
        lang="en",
        use_gpu=bool(request.get("use_gpu", False)),
        show_log=False,
    )
    scores = []
    for image_path, prompt in zip(request["image_paths"], request["prompts"]):
        try:
            image = np.array(Image.open(image_path).convert("RGB"))
            result = ocr.ocr(image, cls=False)
            recognized_text = _recognized_text_from_result(result)
            scores.append(_reward_from_text(recognized_text, prompt))
        except Exception as exc:
            print(f"OCR worker item failed: {exc}", flush=True)
            scores.append(0.0)
    response_path.write_text(json.dumps({"scores": scores}), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-request", type=Path)
    parser.add_argument("--worker-response", type=Path)
    args = parser.parse_args()
    if args.worker_request and args.worker_response:
        _worker_main(args.worker_request, args.worker_response)
        return

    example_image_path = "test_cases/hello world.jpg"
    example_image = Image.open(example_image_path)
    example_prompt = 'New York Skyline with "Hello World" written with fireworks on the sky'
    scorer = OcrScorer(use_gpu=False)
    reward = scorer([example_image], [example_prompt])
    print(f"OCR Reward: {reward}")


if __name__ == "__main__":
    main()
