"""
Moteur de detection + reconnaissance des visages (apprentissage profond).

Deux implementations partagent la meme interface :

  - InsightFaceEngine : SCRFD (detection) + ArcFace (embedding 512-d).
                        Precision maximale (~99.8% sur LFW).
  - OpenCVEngine      : YuNet (detection) + SFace (embedding 128-d).
                        Toujours disponible avec opencv-contrib-python.

Chaque moteur transforme une image en une liste de visages, ou chaque visage
porte sa position (bbox) et son "empreinte" (embedding) normalisee L2. La
comparaison de deux visages se fait alors par simple similarite cosinus.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class DetectedFace:
    """Un visage detecte : position + empreinte normalisee."""

    bbox: tuple[int, int, int, int]   # (x, y, largeur, hauteur)
    embedding: np.ndarray             # vecteur L2-normalise (float32)
    score: float                      # confiance de detection


def insightface_model_dir() -> str:
    """Dossier ou est (ou sera) stocke le modele InsightFace dans le projet."""
    return os.path.join(config.INSIGHTFACE_ROOT, "models", config.INSIGHTFACE_MODEL)


def insightface_model_present() -> bool:
    """True si le modele InsightFace est deja telecharge localement."""
    directory = insightface_model_dir()
    return os.path.isdir(directory) and bool(glob.glob(os.path.join(directory, "*.onnx")))


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Normalise un vecteur (norme L2 = 1) pour que cosinus = produit scalaire."""
    norm = float(np.linalg.norm(vector))
    return (vector / norm).astype(np.float32) if norm > 1e-9 else vector.astype(np.float32)


# --------------------------------------------------------------------------- #
#  Moteur InsightFace (ArcFace) — precision maximale                          #
# --------------------------------------------------------------------------- #
class InsightFaceEngine:
    """Detection SCRFD + reconnaissance ArcFace via la librairie InsightFace."""

    def __init__(self) -> None:
        from insightface.app import FaceAnalysis

        # Cache LOCAL fixe -> telechargement unique (voir config.INSIGHTFACE_ROOT).
        os.makedirs(config.INSIGHTFACE_ROOT, exist_ok=True)
        if insightface_model_present():
            print(f"[Moteur] Modele InsightFace en cache : {insightface_model_dir()}")
        else:
            print("[Moteur] Telechargement unique du modele InsightFace "
                  f"(buffalo_l, ~326 Mo) vers {config.INSIGHTFACE_ROOT} ...")

        # On limite aux modules utiles (detection + reconnaissance) pour la vitesse.
        self._app = FaceAnalysis(
            name=config.INSIGHTFACE_MODEL,
            root=config.INSIGHTFACE_ROOT,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=config.INSIGHTFACE_DET_SIZE)
        self.name = f"InsightFace ArcFace ({config.INSIGHTFACE_MODEL})"
        self.cosine_threshold = config.COSINE_THRESHOLD_INSIGHTFACE
        self.embedding_dim = 512

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        results: list[DetectedFace] = []
        for face in self._app.get(frame_bgr):
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            bbox = (x1, y1, max(0, x2 - x1), max(0, y2 - y1))
            embedding = _l2_normalize(np.asarray(face.normed_embedding))
            results.append(DetectedFace(bbox, embedding, float(face.det_score)))
        return results


# --------------------------------------------------------------------------- #
#  Moteur OpenCV (YuNet + SFace) — toujours disponible                        #
# --------------------------------------------------------------------------- #
class OpenCVEngine:
    """Detection YuNet + reconnaissance SFace, integres a OpenCV (cv2)."""

    def __init__(self) -> None:
        import cv2

        self._cv2 = cv2
        self._detector = cv2.FaceDetectorYN.create(
            config.YUNET_PATH, "", (320, 320),
            score_threshold=config.YUNET_SCORE_THRESHOLD,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(config.SFACE_PATH, "")
        self.name = "OpenCV (YuNet + SFace)"
        self.cosine_threshold = config.COSINE_THRESHOLD_SFACE
        self.embedding_dim = 128

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        height, width = frame_bgr.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(frame_bgr)
        if faces is None:
            return []

        results: list[DetectedFace] = []
        for row in faces:
            # row = [x, y, w, h, 5 points (10 valeurs), score]
            aligned = self._recognizer.alignCrop(frame_bgr, row)
            feature = self._recognizer.feature(aligned).flatten()
            embedding = _l2_normalize(np.asarray(feature))
            x, y, w, h = (int(v) for v in row[:4])
            results.append(DetectedFace((x, y, w, h), embedding, float(row[-1])))
        return results


def create_engine(preference: str = config.ENGINE_PREFERENCE):
    """
    Cree le meilleur moteur disponible selon la preference.

    "auto"        -> InsightFace si installe, sinon OpenCV.
    "insightface" -> force InsightFace (erreur si indisponible).
    "opencv"      -> force OpenCV.

    Renvoie l'instance du moteur (qui possede .name, .detect, .cosine_threshold,
    .embedding_dim).
    """
    if preference in ("auto", "insightface"):
        try:
            return InsightFaceEngine()
        except Exception as error:  # noqa: BLE001 - on bascule proprement
            if preference == "insightface":
                raise RuntimeError(
                    f"InsightFace indisponible : {error}"
                ) from error
            print(f"[Moteur] InsightFace indisponible ({error}). "
                  "Bascule sur OpenCV YuNet + SFace.")

    return OpenCVEngine()
