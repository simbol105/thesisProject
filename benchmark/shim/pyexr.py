"""Minimal stand-in for `pyexr` (not installed in the trellis2 env), backed by OpenCV's EXR codec."""
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np


class _ExrFile:
    def __init__(self, path):
        self.path = path

    def get(self, *args, **kwargs):
        img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"could not read {self.path}")
        if img.ndim == 2:
            img = img[..., None]
        elif img.shape[2] >= 3:
            img = np.concatenate([img[..., 2::-1], img[..., 3:]], axis=-1)
        return img.astype(np.float32)


def open(path):  # noqa: A001 -- same name as pyexr's API
    return _ExrFile(path)


def read(path):
    return open(path).get()
