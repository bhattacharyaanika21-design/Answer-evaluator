"""Handwritten answer recognition with TrOCR (optional module).

TrOCR recognises one line of text at a time, so a full answer-sheet photo is first
split into text lines with a horizontal projection profile, and each line is
transcribed separately.

Requires: torch, transformers, pillow  (pip install -r requirements-ocr.txt)
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

DEFAULT_OCR_MODEL = "microsoft/trocr-base-handwritten"


def preprocess(image: Image.Image) -> Image.Image:
    """Greyscale, fix orientation from EXIF and stretch contrast."""
    image = ImageOps.exif_transpose(image).convert("L")
    return ImageOps.autocontrast(image, cutoff=1)


def segment_lines(image: Image.Image, min_line_height: int = 12, pad: int = 6) -> list[Image.Image]:
    """Split a page image into horizontal text-line crops using ink density per row."""
    gray = np.asarray(preprocess(image), dtype=np.float32)
    ink = gray < (gray.mean() - 0.5 * gray.std())          # dark pixels = ink
    row_ink = ink.sum(axis=1)
    threshold = max(2, 0.02 * ink.shape[1])
    is_text = row_ink > threshold

    lines, start = [], None
    for y, flag in enumerate(np.append(is_text, False)):
        if flag and start is None:
            start = y
        elif not flag and start is not None:
            if y - start >= min_line_height:
                top, bottom = max(0, start - pad), min(gray.shape[0], y + pad)
                cols = np.where(ink[top:bottom].any(axis=0))[0]
                left, right = (max(0, cols[0] - pad), min(gray.shape[1], cols[-1] + pad)) if len(cols) else (0, gray.shape[1])
                lines.append(image.convert("RGB").crop((left, top, right, bottom)))
            start = None
    return lines or [image.convert("RGB")]


class HandwritingOCR:
    def __init__(self, model_name: str = DEFAULT_OCR_MODEL, device: str | None = None):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device).eval()
        self._torch = torch

    def transcribe_line(self, line: Image.Image) -> str:
        pixel_values = self.processor(images=line.convert("RGB"), return_tensors="pt").pixel_values
        with self._torch.no_grad():
            ids = self.model.generate(pixel_values.to(self.device), max_new_tokens=64)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def transcribe(self, image: Image.Image) -> tuple[str, list[str]]:
        """Return the full transcription and the per-line texts."""
        lines = [self.transcribe_line(crop) for crop in segment_lines(image)]
        lines = [t for t in lines if t]
        return " ".join(lines), lines
