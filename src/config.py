"""
Configuration centrale du projet de reconnaissance faciale.

Le projet utilise des modeles d'apprentissage profond (deep learning) :
  - Detection  : YuNet (OpenCV) ou SCRFD (InsightFace) -> trouve les visages,
                 meme de profil ou inclines.
  - Reconnaissance : ArcFace (InsightFace) ou SFace (OpenCV) -> transforme
                 chaque visage en un "vecteur d'empreinte" (embedding). Deux
                 visages se comparent par similarite cosinus.

Cette approche par embeddings remplace l'ancien modele LBPH, beaucoup moins
precis (confusion des noms, echec sur les visages non frontaux).
"""
from __future__ import annotations

import os

# --- Chemins du projet -----------------------------------------------------
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
MODEL_DIR: str = os.path.join(DATA_DIR, "model")          # modeles ONNX
EMBEDDINGS_DIR: str = os.path.join(DATA_DIR, "embeddings")  # empreintes par personne
GALLERY_FILE: str = os.path.join(MODEL_DIR, "gallery.json")  # noms + metadonnees

# Modeles OpenCV (utilises par le moteur de secours)
YUNET_PATH: str = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH: str = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")

# Dossier LOCAL du modele InsightFace, fixe a l'interieur du projet.
# Par defaut, InsightFace met son modele dans ~/.insightface ; sous Windows
# ce dossier ne persiste pas toujours (droits, OneDrive, antivirus), ce qui
# provoque un RE-TELECHARGEMENT a chaque lancement. En fixant le cache ici,
# le modele est telecharge UNE SEULE FOIS puis reutilise indefiniment.
# (Le modele se trouvera dans : data/insightface/models/<INSIGHTFACE_MODEL>/)
INSIGHTFACE_ROOT: str = os.path.join(DATA_DIR, "insightface")

# --- Camera ----------------------------------------------------------------
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480

# --- Choix du moteur -------------------------------------------------------
# "auto"        : InsightFace si disponible, sinon OpenCV (recommande)
# "insightface" : force ArcFace (precision maximale)
# "opencv"      : force YuNet + SFace (toujours disponible)
ENGINE_PREFERENCE: str = "auto"
INSIGHTFACE_MODEL: str = "buffalo_l"          # buffalo_l = precis, buffalo_s = rapide
INSIGHTFACE_DET_SIZE: tuple[int, int] = (480, 480)

# --- Seuils de reconnaissance (similarite cosinus, plus HAUT = plus proche)-
# Au-dessus du seuil -> meme personne ; en dessous -> "Inconnu".
COSINE_THRESHOLD_INSIGHTFACE: float = 0.42    # ArcFace (embedding 512-d)
COSINE_THRESHOLD_SFACE: float = 0.363         # SFace (valeur recommandee OpenCV)
YUNET_SCORE_THRESHOLD: float = 0.80           # confiance minimale de detection

# --- Enregistrement / performances -----------------------------------------
# Nombre d'empreintes capturees par personne lors de l'enregistrement.
ENROLL_SAMPLES: int = 15
# La reconnaissance (lourde) est calculee 1 frame sur N ; les autres frames
# reaffichent le dernier resultat -> video fluide.
PROCESS_EVERY: int = 2
# Une empreinte est capturee toutes les N frames pendant l'enregistrement.
CAPTURE_EVERY: int = 2

# --- Couleurs des superpositions (format RGB pour Pillow) ------------------
COLOR_KNOWN_RGB = (60, 210, 90)      # vert : visage reconnu
COLOR_UNKNOWN_RGB = (255, 70, 70)    # rouge : visage inconnu
COLOR_CAPTURE_RGB = (255, 170, 40)   # orange : enregistrement en cours


def ensure_directories() -> None:
    """Cree les dossiers data/ necessaires s'ils n'existent pas encore."""
    for directory in (DATA_DIR, MODEL_DIR, EMBEDDINGS_DIR, INSIGHTFACE_ROOT):
        os.makedirs(directory, exist_ok=True)
