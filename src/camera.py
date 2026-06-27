"""
Encapsulation de la webcam du PC via OpenCV.

Gere l'ouverture, la lecture des images et la liberation de la camera.
La camera est miroir-inversee (effet selfie) pour une experience plus
naturelle a l'ecran.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config


class Camera:
    """Acces simple a la webcam du PC."""

    def __init__(self, index: int = config.CAMERA_INDEX) -> None:
        self._index = index
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Ouvre la camera et configure la resolution."""
        capture = cv2.VideoCapture(self._index)
        if not capture.isOpened():
            raise RuntimeError(
                f"Impossible d'ouvrir la camera (index {self._index}). "
                "Verifiez qu'aucune autre application ne l'utilise et que "
                "l'autorisation camera est accordee."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self._capture = capture

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def read(self) -> np.ndarray | None:
        """
        Lit une image de la camera. Renvoie l'image (BGR, effet miroir) ou
        None si la lecture echoue.
        """
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return cv2.flip(frame, 1)  # effet miroir horizontal

    def release(self) -> None:
        """Libere la camera."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
