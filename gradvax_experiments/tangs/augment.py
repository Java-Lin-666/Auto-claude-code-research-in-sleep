"""Vendored RandAugment API used by CDMAD.

This follows the public pytorch-randaugment implementation imported by the
released CDMAD loaders, avoiding an unpinned server-side dependency.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps


def _random_sign(value: float) -> float:
    return -value if random.random() > 0.5 else value


def shear_x(img: Image.Image, value: float) -> Image.Image:
    value = _random_sign(value)
    return img.transform(
        img.size,
        Image.AFFINE,
        (1, value, 0, 0, 1, 0),
        Image.BICUBIC,
        fillcolor=(128, 128, 128),
    )


def shear_y(img: Image.Image, value: float) -> Image.Image:
    value = _random_sign(value)
    return img.transform(
        img.size,
        Image.AFFINE,
        (1, 0, 0, value, 1, 0),
        Image.BICUBIC,
        fillcolor=(128, 128, 128),
    )


def translate_x(img: Image.Image, value: float) -> Image.Image:
    value = _random_sign(value)
    return img.transform(
        img.size,
        Image.AFFINE,
        (1, 0, value * img.size[0], 0, 1, 0),
        fillcolor=(128, 128, 128),
    )


def translate_y(img: Image.Image, value: float) -> Image.Image:
    value = _random_sign(value)
    return img.transform(
        img.size,
        Image.AFFINE,
        (1, 0, 0, 0, 1, value * img.size[1]),
        fillcolor=(128, 128, 128),
    )


def rotate(img: Image.Image, value: float) -> Image.Image:
    return img.rotate(_random_sign(value))


def color(img: Image.Image, value: float) -> Image.Image:
    return ImageEnhance.Color(img).enhance(value)


def posterize(img: Image.Image, value: float) -> Image.Image:
    return ImageOps.posterize(img, int(value))


def solarize(img: Image.Image, value: float) -> Image.Image:
    return ImageOps.solarize(img, int(value))


def contrast(img: Image.Image, value: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(value)


def sharpness(img: Image.Image, value: float) -> Image.Image:
    return ImageEnhance.Sharpness(img).enhance(value)


def brightness(img: Image.Image, value: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(value)


def autocontrast(img: Image.Image, _: float) -> Image.Image:
    return ImageOps.autocontrast(img)


def equalize(img: Image.Image, _: float) -> Image.Image:
    return ImageOps.equalize(img)


def invert(img: Image.Image, _: float) -> Image.Image:
    return ImageOps.invert(img)


Augmentation = tuple[Callable[[Image.Image, float], Image.Image], float, float]


def augment_list() -> list[Augmentation]:
    return [
        (autocontrast, 0, 1),
        (equalize, 0, 1),
        (invert, 0, 1),
        (rotate, 0, 30),
        (posterize, 0, 4),
        (solarize, 0, 256),
        (color, 0.1, 1.9),
        (contrast, 0.1, 1.9),
        (brightness, 0.1, 1.9),
        (sharpness, 0.1, 1.9),
        (shear_x, 0.0, 0.3),
        (shear_y, 0.0, 0.3),
        (translate_x, 0.0, 0.33),
        (translate_y, 0.0, 0.33),
    ]


class RandAugment:
    def __init__(self, n: int, m: int):
        if n < 0 or not 0 <= m <= 10:
            raise ValueError("RandAugment requires n >= 0 and m in [0, 10].")
        self.n = n
        self.m = m
        self.augment_list = augment_list()

    def __call__(self, img: Image.Image) -> Image.Image:
        for operation, minimum, maximum in random.choices(
            self.augment_list, k=self.n
        ):
            value = (float(self.m) / 10) * (maximum - minimum) + minimum
            img = operation(img, value)
        return img


class CutoutDefault:
    def __init__(self, length: int):
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        height, width = img.size(1), img.size(2)
        mask = np.ones((height, width), np.float32)
        y = np.random.randint(height)
        x = np.random.randint(width)
        y1 = np.clip(y - self.length // 2, 0, height)
        y2 = np.clip(y + self.length // 2, 0, height)
        x1 = np.clip(x - self.length // 2, 0, width)
        x2 = np.clip(x + self.length // 2, 0, width)
        mask[y1:y2, x1:x2] = 0.0
        mask_tensor = torch.from_numpy(mask).expand_as(img)
        return img * mask_tensor
