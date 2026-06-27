"""
Tests de la chaine de reconnaissance, sans camera.

On verifie :
  - la galerie (memoire) : enregistrement, persistance, suppression ;
  - la reconnaissance par similarite cosinus (bon nom / "Inconnu") ;
  - le filtrage des empreintes de dimension incompatible ;
  - l'initialisation du moteur de reconnaissance et la detection sur une image.

Lancement :
    ./venv/bin/python tests/test_pipeline.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402


def _unit(vector: np.ndarray) -> np.ndarray:
    """Renvoie le vecteur normalise (norme L2 = 1) en float32."""
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _redirect_config_to(tmp: str) -> None:
    config.DATA_DIR = tmp
    config.MODEL_DIR = os.path.join(tmp, "model")
    config.EMBEDDINGS_DIR = os.path.join(tmp, "embeddings")
    config.GALLERY_FILE = os.path.join(config.MODEL_DIR, "gallery.json")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)


def test_gallery_recognition() -> None:
    """Enregistrement puis reconnaissance par similarite cosinus."""
    from src.gallery import FaceGallery

    with tempfile.TemporaryDirectory() as tmp:
        _redirect_config_to(tmp)
        rng = np.random.RandomState(42)

        alice = _unit(rng.randn(128))
        bob = _unit(rng.randn(128))

        gallery = FaceGallery(embedding_dim=128, cosine_threshold=0.5)
        # Plusieurs empreintes legerement bruitees par personne
        gallery.enroll("Alice", [_unit(alice + 0.01 * rng.randn(128)) for _ in range(5)],
                       date="2026-06-27")
        gallery.enroll("Bob", [_unit(bob + 0.01 * rng.randn(128)) for _ in range(5)],
                       date="2026-06-27")
        assert gallery.count() == 2

        # Une variante d'Alice doit etre reconnue comme Alice
        result_alice = gallery.recognize(_unit(alice + 0.02 * rng.randn(128)))
        assert result_alice.is_known and result_alice.name == "Alice", result_alice

        result_bob = gallery.recognize(_unit(bob + 0.02 * rng.randn(128)))
        assert result_bob.is_known and result_bob.name == "Bob", result_bob

        # Un visage totalement different doit etre "Inconnu"
        stranger = _unit(rng.randn(128))
        result_unknown = gallery.recognize(stranger)
        assert not result_unknown.is_known, result_unknown

    print("OK - test_gallery_recognition : memoire + matching cosinus valides")


def test_gallery_persistence_and_delete() -> None:
    """La galerie est relue depuis le disque ; la suppression fonctionne."""
    from src.gallery import FaceGallery

    with tempfile.TemporaryDirectory() as tmp:
        _redirect_config_to(tmp)
        rng = np.random.RandomState(7)

        g1 = FaceGallery(embedding_dim=64, cosine_threshold=0.5)
        label = g1.enroll("Yassine", [_unit(rng.randn(64)) for _ in range(4)],
                          date="2026-06-27")

        # Nouvelle instance : doit relire la personne depuis le disque
        g2 = FaceGallery(embedding_dim=64, cosine_threshold=0.5)
        assert g2.count() == 1
        assert g2.find_label_by_name("yassine") == label

        # Suppression
        g2.delete(label)
        assert g2.count() == 0 and g2.is_empty()

    print("OK - test_gallery_persistence_and_delete : persistance + suppression")


def test_dimension_filtering() -> None:
    """Les empreintes d'une autre dimension (autre moteur) sont ignorees."""
    from src.gallery import FaceGallery

    with tempfile.TemporaryDirectory() as tmp:
        _redirect_config_to(tmp)
        rng = np.random.RandomState(1)

        # Enregistre en dimension 128
        g128 = FaceGallery(embedding_dim=128, cosine_threshold=0.5)
        g128.enroll("X", [_unit(rng.randn(128)) for _ in range(3)], date="2026-06-27")

        # Une galerie en dimension 512 ne doit pas charger ces empreintes
        g512 = FaceGallery(embedding_dim=512, cosine_threshold=0.5)
        assert g512.is_empty(), "les empreintes 128-d ne doivent pas etre chargees en 512-d"

    print("OK - test_dimension_filtering : compatibilite des dimensions geree")


def test_engine_and_detection() -> None:
    """Le moteur s'initialise et la detection renvoie une liste sur une image."""
    from src.face_engine import create_engine

    # Restaure les chemins reels des modeles (non rediriges).
    import importlib
    importlib.reload(config)

    engine = create_engine()
    assert engine.embedding_dim in (128, 512)
    assert engine.cosine_threshold > 0

    blank = (np.random.RandomState(0).rand(480, 640, 3) * 255).astype(np.uint8)
    faces = engine.detect(blank)
    assert isinstance(faces, list)  # 0 visage attendu sur du bruit
    print(f"OK - test_engine_and_detection : moteur '{engine.name}' "
          f"(dim={engine.embedding_dim}) operationnel")


if __name__ == "__main__":
    test_gallery_recognition()
    test_gallery_persistence_and_delete()
    test_dimension_filtering()
    test_engine_and_detection()
    print("\nTous les tests sont passes avec succes.")
