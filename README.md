# Reconnaissance Faciale par Apprentissage Profond — PFE

**Technicien Spécialisé en Réseaux Informatique**

Application de bureau qui utilise la **webcam du PC** pour :

1. **Détecter** les visages en temps réel (même de profil ou inclinés).
2. **Enregistrer** un visage en lui associant un **nom**.
3. **Mémoriser** ce visage de façon permanente.
4. **Reconnaître** automatiquement la personne et **afficher son nom** dès
   qu'elle réapparaît devant la caméra.

> **Note technique** : ce projet utilise des modèles d'**apprentissage profond
> (deep learning)** — *ArcFace / SFace* — et non l'ancien algorithme LBPH.
> Cela corrige deux défauts majeurs : la **confusion entre des visages
> similaires** et l'**échec sur les visages non frontaux**.

---

## 1. Comment ça marche (le pipeline)

La reconnaissance faciale moderne se fait en **deux étapes** :

```
   IMAGE CAMÉRA
        │
        ▼
 ┌──────────────┐   « OÙ sont les visages ? »
 │  DÉTECTION   │   SCRFD (InsightFace) ou YuNet (OpenCV)
 └──────────────┘   → détecte aussi les visages de profil / inclinés
        │
        ▼
 ┌──────────────┐   « QUI est-ce ? »
 │RECONNAISSANCE│   ArcFace (InsightFace) ou SFace (OpenCV)
 └──────────────┘   → transforme chaque visage en une EMPREINTE (vecteur)
        │
        ▼
 ┌──────────────┐   Compare l'empreinte à la mémoire (similarité cosinus).
 │  COMPARAISON │   Score élevé → on affiche le NOM ; sinon → « Inconnu ».
 └──────────────┘
```

L'**empreinte** (embedding) est un vecteur de 512 nombres (ArcFace) ou 128
(SFace) qui décrit un visage. Deux photos de la **même personne** donnent des
vecteurs très proches ; deux personnes **différentes** donnent des vecteurs
éloignés. C'est pourquoi il n'y a **pas de confusion** et **aucun
ré-entraînement** : enregistrer une personne = mémoriser ses empreintes.

### Pourquoi pas LBPH (l'ancien modèle) ?
LBPH compare des *textures* d'images frontales : il confond les visages
proches et ne reconnaît pas un visage tourné. ArcFace/SFace, entraînés sur des
millions de visages, sont bien plus précis (**~99.8 %** vs ~%medium pour LBPH).

### Pourquoi pas « juste YOLOv8 » ?
YOLOv8 est un **détecteur** : il dit *où* est un visage, pas *qui* c'est. Il
faut **toujours** un modèle de reconnaissance (ArcFace/SFace) derrière. Ici, le
détecteur SCRFD/YuNet joue le même rôle qu'un YOLOv8-face.

---

## 2. Technologies utilisées

| Outil | Rôle |
|-------|------|
| **Python 3** | Langage de programmation |
| **InsightFace** (ArcFace + SCRFD) | Moteur principal — précision maximale |
| **OpenCV** (YuNet + SFace) | Moteur de secours + accès caméra |
| **ONNX Runtime** | Exécution des modèles deep learning sur CPU |
| **NumPy** | Calcul des empreintes et de la similarité cosinus |
| **Pillow (PIL)** | Affichage vidéo et texte (accents) dans l'interface |
| **Tkinter** | Interface graphique (fourni avec Python) |

Le programme choisit **automatiquement** le meilleur moteur disponible
(InsightFace si installé, sinon OpenCV). Réglable dans `src/config.py`.

---

## 3. Architecture du projet

```
pfe_camera/
├── main.py                 # Point d'entrée (lance l'application)
├── requirements.txt        # Dépendances Python
├── run.sh                  # Script de lancement automatique
├── README.md               # Ce document
├── src/                    # Code source
│   ├── config.py           # Constantes, chemins et seuils
│   ├── camera.py           # Accès à la webcam
│   ├── face_engine.py      # Détection + reconnaissance (2 moteurs au choix)
│   ├── gallery.py          # Mémoire des personnes (empreintes + recherche)
│   └── gui.py              # Interface graphique
├── tests/
│   └── test_pipeline.py    # Tests automatiques (sans caméra)
└── data/                   # Mémoire persistante
    ├── model/              # Modèles ONNX (YuNet, SFace) + gallery.json
    └── embeddings/         # Empreintes de chaque personne (.npy)
```

---

## 4. Installation

### macOS / Linux

```bash
cd pfe_technicien_specialise_en_reseaux_informatique_camera
chmod +x run.sh
./run.sh
```

### Windows

Double-cliquez sur **`run.bat`**, ou en ligne de commande :

```bat
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python scripts\setup_model.py   :: telechargement UNIQUE du modele
venv\Scripts\python main.py
```

### ⬇️ Téléchargement du modèle — UNE SEULE FOIS

Au premier lancement, le modèle ArcFace (`buffalo_l`, ~326 Mo) est téléchargé
**une seule fois** dans le dossier du projet : **`data/insightface/`**. Les
lancements suivants le réutilisent — **plus aucun téléchargement**.

> 💡 **Le modèle se retéléchargeait à chaque `python main.py` ?** (Problème
> fréquent sous Windows quand `~/.insightface` n'est pas conservé.) C'est
> **corrigé** : le cache est désormais fixé à l'intérieur du projet. Pour le
> télécharger à l'avance, lancez une fois :
> ```bash
> python scripts/setup_model.py
> ```
>
> 🚫 **Pas d'internet / téléchargement bloqué ?** Ouvrez `src/config.py` et
> mettez `ENGINE_PREFERENCE = "opencv"`. Le moteur **YuNet + SFace** (déjà
> inclus dans `data/model/`) fonctionne **sans aucun téléchargement**.

### ⚠️ Autorisation de la caméra (macOS)

Au premier lancement, macOS demande l'autorisation d'utiliser la caméra.
Si l'application affiche « Caméra indisponible » :

1. **Réglages Système → Confidentialité et sécurité → Caméra**
2. Activer l'autorisation pour votre **Terminal**.
3. Relancer l'application.

---

## 5. Utilisation

1. **Lancer l'application** : la webcam s'affiche, les visages sont entourés
   d'un cadre.
2. **Enregistrer une personne** :
   - Saisir le **nom** dans le champ de droite.
   - Cliquer sur **« Enregistrer ce visage »**.
   - Regarder la caméra et **tourner lentement la tête** : l'application
     capture plusieurs empreintes sous différents angles.
3. **Reconnaissance automatique** : dès que la personne réapparaît, son **nom**
   et un **score de confiance** s'affichent (en vert). Un visage non enregistré
   est marqué **« Inconnu »** (en rouge).
4. La mémoire est **permanente** : fermez puis rouvrez l'application, les
   personnes restent reconnues.

---

## 6. Réglages (`src/config.py`)

| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| `ENGINE_PREFERENCE` | `auto` / `insightface` / `opencv` | `auto` |
| `INSIGHTFACE_MODEL` | `buffalo_l` (précis) ou `buffalo_s` (rapide) | `buffalo_l` |
| `ENROLL_SAMPLES` | Empreintes capturées par personne | 15 |
| `COSINE_THRESHOLD_INSIGHTFACE` | Seuil de reconnaissance ArcFace | 0.42 |
| `COSINE_THRESHOLD_SFACE` | Seuil de reconnaissance SFace | 0.363 |
| `PROCESS_EVERY` | Reconnaissance 1 frame sur N (fluidité) | 2 |

> **Score (similarité cosinus)** : plus il est **élevé**, plus la
> correspondance est forte. En dessous du seuil → « Inconnu ».
> *Si l'ordinateur est lent, mettez `INSIGHTFACE_MODEL = "buffalo_s"`.*

---

## 7. Tests

```bash
./venv/bin/python tests/test_pipeline.py
```

Vérifie la mémoire, la persistance, la recherche par cosinus et l'initialisation
du moteur. Résultat attendu : `Tous les tests sont passés avec succès.`

---

## 8. Améliorations possibles (perspectives)

- Système de **pointage / présence** (enregistrer les heures d'arrivée).
- Détection de **vivacité** (anti-photo) pour éviter la triche.
- Journalisation des reconnaissances dans une base de données.
- Caméra **réseau / IP** au lieu de la webcam locale.
```
