"""
Telechargement UNIQUE du modele de reconnaissance (InsightFace / ArcFace).

A lancer UNE SEULE FOIS, apres l'installation :

    python scripts/setup_model.py

Le modele est enregistre dans le projet (data/insightface/). Ensuite,
`python main.py` le reutilise sans jamais le retelecharger.

En cas d'echec de telechargement (pas d'internet, pare-feu...), utilisez le
moteur OpenCV qui ne necessite AUCUN telechargement : ouvrez src/config.py et
mettez  ENGINE_PREFERENCE = "opencv".
"""
from __future__ import annotations

import os
import sys

# Permet d'importer le package src/ quand le script est lance directement.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.face_engine import insightface_model_dir, insightface_model_present  # noqa: E402


def main() -> int:
    config.ensure_directories()

    if insightface_model_present():
        print("OK : le modele est deja present.")
        print("Emplacement :", insightface_model_dir())
        print("Vous pouvez lancer :  python main.py")
        return 0

    print("Telechargement unique du modele InsightFace (buffalo_l, ~326 Mo)...")
    print("Destination :", config.INSIGHTFACE_ROOT)
    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name=config.INSIGHTFACE_MODEL,
            root=config.INSIGHTFACE_ROOT,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=config.INSIGHTFACE_DET_SIZE)
    except Exception as error:  # noqa: BLE001
        print("\nEchec du telechargement :", error)
        print("\nPlan B (aucun telechargement) : ouvrez src/config.py et mettez")
        print('   ENGINE_PREFERENCE = "opencv"')
        print("Le moteur YuNet + SFace (deja inclus) sera utilise a la place.")
        return 1

    print("\nTermine ! Modele enregistre dans :", insightface_model_dir())
    print("Lancez maintenant :  python main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
