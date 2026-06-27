"""
Interface graphique (Tkinter) de l'application de reconnaissance faciale.

Pipeline (apprentissage profond) :
  1. La webcam affiche la video en continu.
  2. Le moteur (InsightFace/ArcFace ou OpenCV/SFace) detecte chaque visage et
     calcule son empreinte (embedding), meme de profil ou incline.
  3. L'empreinte est comparee a la galerie des personnes connues : si elle
     correspond, le nom s'affiche ; sinon "Inconnu".
  4. "Enregistrer ce visage" capture plusieurs empreintes d'une personne et
     les memorise definitivement (aucun reentrainement necessaire).

La detection (couteuse) est calculee une frame sur N : les frames
intermediaires reaffichent le dernier resultat, ce qui garde une video fluide.
"""
from __future__ import annotations

import datetime
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageTk

from . import config
from .camera import Camera
from .face_engine import create_engine
from .gallery import FaceGallery

# Theme de l'interface (sombre).
BG_DARK = "#1e1f2b"
BG_PANEL = "#272a3d"
FG_LIGHT = "#e8e9f0"
ACCENT = "#4f8cff"

FRAME_DELAY_MS = 15


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Charge une police TrueType lisible, avec repli sur la police par defaut."""
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


class FaceRecognitionApp:
    """Application principale."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Reconnaissance Faciale - PFE")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        config.ensure_directories()

        # Composants metier
        self.camera = Camera()
        self.engine = create_engine()
        self.gallery = FaceGallery(self.engine.embedding_dim, self.engine.cosine_threshold)

        self.overlay_font = _load_font(22)

        # Etat de capture
        self.mode = "recognize"          # "recognize" ou "capture"
        self.capture_name = ""
        self.capture_embeddings: list = []
        self.frame_index = 0
        self.last_overlays: list[tuple] = []  # overlays reaffiches entre 2 calculs

        self._photo: ImageTk.PhotoImage | None = None  # reference anti-GC

        self._build_ui()
        self._start_camera()
        self._update_people_list()
        self._update_frame()

    # ----------------------------------------------------------------- UI ---
    def _build_ui(self) -> None:
        container = tk.Frame(self.root, bg=BG_DARK, padx=12, pady=12)
        container.pack()

        video_wrap = tk.Frame(container, bg="#000000")
        video_wrap.grid(row=0, column=0, padx=(0, 12))
        self.video_label = tk.Label(video_wrap, bg="#000000")
        self.video_label.pack()

        panel = tk.Frame(container, bg=BG_PANEL, padx=16, pady=16)
        panel.grid(row=0, column=1, sticky="n")

        tk.Label(panel, text="Reconnaissance Faciale", bg=BG_PANEL, fg=FG_LIGHT,
                 font=("Helvetica", 16, "bold")).pack(anchor="w")
        tk.Label(panel, text=f"Moteur : {self.engine.name}", bg=BG_PANEL,
                 fg=ACCENT, font=("Helvetica", 9, "bold")).pack(anchor="w")
        tk.Label(panel, text="PFE - Technicien Specialise en Reseaux",
                 bg=BG_PANEL, fg="#9aa0c0", font=("Helvetica", 9)
                 ).pack(anchor="w", pady=(0, 14))

        tk.Label(panel, text="Nom de la personne :", bg=BG_PANEL, fg=FG_LIGHT,
                 font=("Helvetica", 10)).pack(anchor="w")
        self.name_entry = tk.Entry(panel, font=("Helvetica", 12), width=24,
                                   bg="#1b1d29", fg=FG_LIGHT,
                                   insertbackground=FG_LIGHT, relief="flat")
        self.name_entry.pack(anchor="w", pady=(4, 10), ipady=5)
        self.name_entry.bind("<Return>", lambda _e: self._start_capture())

        self.btn_register = tk.Button(
            panel, text="Enregistrer ce visage", command=self._start_capture,
            bg=ACCENT, fg="white", font=("Helvetica", 11, "bold"), relief="flat",
            activebackground="#3a6fd0", activeforeground="white", cursor="hand2",
            padx=10, pady=8)
        self.btn_register.pack(fill="x")

        tk.Label(panel, text="Personnes en memoire :", bg=BG_PANEL, fg=FG_LIGHT,
                 font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(16, 4))

        list_frame = tk.Frame(panel, bg=BG_PANEL)
        list_frame.pack(fill="x")
        self.people_list = tk.Listbox(
            list_frame, height=8, width=28, bg="#1b1d29", fg=FG_LIGHT,
            selectbackground=ACCENT, relief="flat", highlightthickness=0,
            font=("Helvetica", 10), activestyle="none")
        self.people_list.pack(side="left", fill="x", expand=True)
        scroll = ttk.Scrollbar(list_frame, command=self.people_list.yview)
        scroll.pack(side="right", fill="y")
        self.people_list.config(yscrollcommand=scroll.set)

        self.btn_delete = tk.Button(
            panel, text="Supprimer la personne selectionnee",
            command=self._delete_selected, bg="#5a2a3a", fg=FG_LIGHT,
            font=("Helvetica", 9), relief="flat", activebackground="#7a3a4f",
            activeforeground="white", cursor="hand2", padx=8, pady=5)
        self.btn_delete.pack(fill="x", pady=(8, 0))

        self.status_var = tk.StringVar(value="Pret.")
        tk.Label(panel, textvariable=self.status_var, bg=BG_PANEL, fg="#9aa0c0",
                 font=("Helvetica", 9), wraplength=240, justify="left"
                 ).pack(anchor="w", pady=(16, 0))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------ Camera ---
    def _start_camera(self) -> None:
        try:
            self.camera.open()
            self.status_var.set("Camera active. Placez-vous face a l'objectif.")
        except RuntimeError as error:
            messagebox.showerror("Erreur camera", str(error))
            self.status_var.set("Camera indisponible.")

    # -------------------------------------------------------- Boucle video --
    def _update_frame(self) -> None:
        frame = self.camera.read() if self.camera.is_open else None
        if frame is not None:
            self.frame_index += 1
            # Calcul lourd seulement 1 frame sur PROCESS_EVERY.
            if self.frame_index % config.PROCESS_EVERY == 0:
                faces = self.engine.detect(frame)
                if self.mode == "capture":
                    self.last_overlays = self._handle_capture(faces)
                else:
                    self.last_overlays = self._handle_recognition(faces)
            self._render(frame, self.last_overlays)

        self.root.after(FRAME_DELAY_MS, self._update_frame)

    def _handle_recognition(self, faces) -> list[tuple]:
        """Reconnait chaque visage detecte."""
        overlays: list[tuple] = []
        for face in faces:
            if self.gallery.is_empty():
                overlays.append((face.bbox, "Visage detecte", config.COLOR_UNKNOWN_RGB))
                continue
            match = self.gallery.recognize(face.embedding)
            if match.is_known:
                text = f"{match.name} ({match.score:.2f})"
                color = config.COLOR_KNOWN_RGB
            else:
                text = f"Inconnu ({match.score:.2f})"
                color = config.COLOR_UNKNOWN_RGB
            overlays.append((face.bbox, text, color))
        return overlays

    def _handle_capture(self, faces) -> list[tuple]:
        """Collecte les empreintes du plus grand visage pendant l'enregistrement."""
        if not faces:
            return [((10, 10, 0, 0), "Aucun visage detecte...", config.COLOR_CAPTURE_RGB)]

        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        self.capture_embeddings.append(face.embedding)
        collected = len(self.capture_embeddings)
        self.status_var.set(
            f"Capture de '{self.capture_name}' : {collected}/{config.ENROLL_SAMPLES}"
        )
        if collected >= config.ENROLL_SAMPLES:
            self._finish_capture()
        return [(face.bbox, f"Capture {collected}/{config.ENROLL_SAMPLES}",
                 config.COLOR_CAPTURE_RGB)]

    def _render(self, frame, overlays: list[tuple]) -> None:
        """Dessine les superpositions puis affiche l'image dans Tkinter."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)

        for bbox, text, color in overlays:
            x, y, w, h = bbox
            if w > 0 and h > 0:
                draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            ty = max(0, y - 30)
            box = draw.textbbox((x, ty), text, font=self.overlay_font)
            draw.rectangle([box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2], fill=color)
            draw.text((x, ty), text, fill="white", font=self.overlay_font)

        self._photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self._photo)

    # ------------------------------------------------------- Enregistrement -
    def _start_capture(self) -> None:
        if self.mode == "capture":
            return
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Nom manquant",
                                   "Veuillez saisir un nom avant l'enregistrement.")
            return
        if not self.camera.is_open:
            messagebox.showerror("Camera", "La camera n'est pas disponible.")
            return

        self.capture_name = name
        self.capture_embeddings = []
        self.mode = "capture"
        self.btn_register.configure(state="disabled", text="Capture en cours...")
        self.status_var.set(f"Regardez la camera : capture de '{name}'... "
                            "Tournez lentement la tete.")

    def _finish_capture(self) -> None:
        """Termine la capture : enregistre les empreintes en galerie."""
        self.mode = "recognize"
        date = datetime.date.today().isoformat()
        try:
            self.gallery.enroll(self.capture_name, self.capture_embeddings, date)
            self.status_var.set(
                f"'{self.capture_name}' enregistre "
                f"({self.gallery.count()} personnes en memoire)."
            )
        except ValueError as error:
            self.status_var.set(f"Erreur : {error}")

        self.capture_embeddings = []
        self.btn_register.configure(state="normal", text="Enregistrer ce visage")
        self.name_entry.delete(0, tk.END)
        self._update_people_list()

    # --------------------------------------------------- Liste / suppression -
    def _update_people_list(self) -> None:
        self.people_list.delete(0, tk.END)
        people = self.gallery.list_people()
        if not people:
            self.people_list.insert(tk.END, "  (aucune personne enregistree)")
            return
        for person in people:
            self.people_list.insert(
                tk.END, f"  {person['name']}  -  {person['count']} empreintes")

    def _delete_selected(self) -> None:
        selection = self.people_list.curselection()
        people = self.gallery.list_people()
        if not selection or not people or selection[0] >= len(people):
            return
        person = people[selection[0]]
        if not messagebox.askyesno("Confirmation",
                                   f"Supprimer '{person['name']}' de la memoire ?"):
            return
        self.gallery.delete(person["label"])
        self._update_people_list()
        self.status_var.set(f"'{person['name']}' supprime de la memoire.")

    # ----------------------------------------------------------- Fermeture --
    def _on_close(self) -> None:
        self.camera.release()
        self.root.destroy()


def run() -> None:
    """Point d'entree : cree la fenetre et lance la boucle Tkinter."""
    root = tk.Tk()
    FaceRecognitionApp(root)
    root.mainloop()
