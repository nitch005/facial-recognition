"""
Memoire des personnes connues ("galerie").

Pour chaque personne on memorise :
  - des metadonnees (nom, date, nombre d'empreintes) dans gallery.json ;
  - une matrice d'empreintes (embeddings) dans data/embeddings/<label>.npy.

La reconnaissance compare l'empreinte d'un visage inconnu a TOUTES les
empreintes memorisees via la similarite cosinus (plus haut = plus proche),
puis renvoie la personne la plus proche si le score depasse le seuil.

Les empreintes etant normalisees (L2 = 1), la similarite cosinus se reduit a
un simple produit scalaire, ce qui rend la recherche tres rapide.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from . import config


@dataclass(frozen=True)
class Match:
    """Resultat d'une reconnaissance."""

    name: str         # nom reconnu ou "Inconnu"
    score: float      # similarite cosinus avec la meilleure correspondance
    is_known: bool    # True si le score depasse le seuil


class FaceGallery:
    """Galerie persistante des visages connus."""

    def __init__(self, embedding_dim: int, cosine_threshold: float) -> None:
        self._dim = embedding_dim
        self._threshold = cosine_threshold
        # metadonnees : {label(int): {"name": str, "date": str, "count": int}}
        self._meta: dict[int, dict] = {}
        # cache memoire pour la recherche vectorielle
        self._matrix = np.empty((0, embedding_dim), dtype=np.float32)
        self._labels = np.empty((0,), dtype=np.int64)
        config.ensure_directories()
        self.load()

    # ----------------------------------------------------------- Persistance -
    def _embedding_path(self, label: int) -> str:
        return os.path.join(config.EMBEDDINGS_DIR, f"{label}.npy")

    def load(self) -> None:
        """Recharge les metadonnees et reconstruit la matrice d'empreintes."""
        self._meta = {}
        if os.path.exists(config.GALLERY_FILE):
            try:
                with open(config.GALLERY_FILE, "r", encoding="utf-8") as file:
                    raw = json.load(file)
                self._meta = {int(k): v for k, v in raw.items()}
            except (json.JSONDecodeError, ValueError, OSError):
                self._meta = {}

        matrices: list[np.ndarray] = []
        labels: list[int] = []
        for label in list(self._meta.keys()):
            path = self._embedding_path(label)
            if not os.path.exists(path):
                continue
            data = np.load(path)
            # On ignore les empreintes d'une autre dimension (autre moteur).
            if data.ndim == 2 and data.shape[1] == self._dim:
                matrices.append(data.astype(np.float32))
                labels.extend([label] * data.shape[0])

        if matrices:
            self._matrix = np.vstack(matrices)
            self._labels = np.asarray(labels, dtype=np.int64)
        else:
            self._matrix = np.empty((0, self._dim), dtype=np.float32)
            self._labels = np.empty((0,), dtype=np.int64)

    def _save_meta(self) -> None:
        serializable = {str(k): v for k, v in self._meta.items()}
        with open(config.GALLERY_FILE, "w", encoding="utf-8") as file:
            json.dump(serializable, file, ensure_ascii=False, indent=2)

    # --------------------------------------------------------------- Lecture -
    def count(self) -> int:
        return len(self._meta)

    def is_empty(self) -> bool:
        return self._matrix.shape[0] == 0

    def find_label_by_name(self, name: str) -> int | None:
        target = name.strip().lower()
        for label, info in self._meta.items():
            if info["name"].strip().lower() == target:
                return label
        return None

    def list_people(self) -> list[dict]:
        people = [{"label": label, **info} for label, info in self._meta.items()]
        return sorted(people, key=lambda p: p["label"])

    # ------------------------------------------------------- Enregistrement --
    def enroll(self, name: str, embeddings: list[np.ndarray], date: str) -> int:
        """
        Ajoute (ou complete) une personne avec de nouvelles empreintes.

        Renvoie le label de la personne.
        """
        if not embeddings:
            raise ValueError("Aucune empreinte a enregistrer.")

        new_matrix = np.vstack([e.reshape(1, -1) for e in embeddings]).astype(np.float32)

        existing = self.find_label_by_name(name)
        if existing is not None:
            label = existing
            path = self._embedding_path(label)
            if os.path.exists(path):
                old = np.load(path)
                if old.shape[1] == new_matrix.shape[1]:
                    new_matrix = np.vstack([old, new_matrix])
        else:
            label = (max(self._meta.keys()) + 1) if self._meta else 0

        np.save(self._embedding_path(label), new_matrix)
        self._meta[label] = {
            "name": name.strip(),
            "date": date,
            "count": int(new_matrix.shape[0]),
        }
        self._save_meta()
        self.load()  # reconstruit le cache de recherche
        return label

    def delete(self, label: int) -> None:
        """Supprime une personne (metadonnees + empreintes)."""
        if label in self._meta:
            del self._meta[label]
            self._save_meta()
        path = self._embedding_path(label)
        if os.path.exists(path):
            os.remove(path)
        self.load()

    # ------------------------------------------------------- Reconnaissance --
    def recognize(self, embedding: np.ndarray) -> Match:
        """Trouve la personne la plus proche d'une empreinte donnee."""
        if self.is_empty():
            return Match(name="Inconnu", score=0.0, is_known=False)

        # Similarite cosinus = produit scalaire (vecteurs normalises).
        sims = self._matrix @ embedding.astype(np.float32)
        best_index = int(np.argmax(sims))
        best_score = float(sims[best_index])
        best_label = int(self._labels[best_index])

        if best_score >= self._threshold:
            name = self._meta.get(best_label, {}).get("name", "Inconnu")
            return Match(name=name, score=best_score, is_known=True)
        return Match(name="Inconnu", score=best_score, is_known=False)
