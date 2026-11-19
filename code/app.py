import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"      # Gradio să nu trimită statistici de utilizare
import cv2
import gradio as gr
import matplotlib
matplotlib.use("Agg")                                  # desenez graficele în memorie, fără fereastră
import matplotlib.pyplot as plt

from preprocesare import deseneaza_puncte
from fuziune import z
from detector import Detector
from config import FOLDER_PROIECT, FOLDER_DATE
from date import este_imagine
from ramura_b import suprapune_harta

detector = None 


def ia_detectorul():
    global detector
    if detector is None:
        detector = Detector()
    return detector


def text_rezultat(rez, calibrare):
    """Verdictul și tabelul cu scorurile, în Markdown."""
    text = "## Verdict: " + rez["eticheta"] + "\n\n"
    text = text + "| Ramură | Scor brut | z-score |\n|---|---|---|\n"
    if rez["scor_a"] is not None:
        z_a = z(rez["scor_a"], calibrare["z"]["A"])
        text = text + "| A · SVM pe 104 trăsături | " + str(round(rez["scor_a"], 3)) + " | " + str(round(z_a, 2)) + " |\n"
    if rez["scor_b"] is not None:
        z_b = z(rez["scor_b"], calibrare["z"]["B"])
        text = text + "| B · EfficientNet-B0 | " + str(round(rez["scor_b"], 3)) + " | " + str(round(z_b, 2)) + " |\n"
    text = text + "| **Fuzionat** (media z-score-urilor) | | **" + str(round(rez["scor"], 2)) + "** |\n\n"
    text = text + "Pragul de decizie, ales pe validare: **" + str(round(rez["prag"], 2)) + "**. "
    text = text + "Scor mai mare sau egal cu pragul = FALS."
    ponderi = calibrare["ponderi"]
    if len(ponderi) == 2 and ponderi["A"] != ponderi["B"]:
        text = text + " Ponderile ramurilor, alese pe validare: A = " + str(ponderi["A"]) + ", B = " + str(ponderi["B"]) + "."
    return text


def analizeaza(imagine):
    if imagine is None:
        return "Încarcă o imagine.", None, None
    try:
        det = ia_detectorul()
    except FileNotFoundError as eroare:
        return "Modelele nu sunt gata: " + str(eroare), None, None
    img = cv2.cvtColor(imagine, cv2.COLOR_RGB2BGR)
    rez = det.analizeaza_imagine(img)
    if not rez["ok"]:
        return "Nu am găsit nicio față în imagine.", None, None
    desen = deseneaza_puncte(rez["fata"], rez["puncte"])
    desen = cv2.cvtColor(desen, cv2.COLOR_BGR2RGB)
    harta = None
    if rez["harta"] is not None:
        harta = suprapune_harta(rez["fata"], rez["harta"])
        harta = cv2.cvtColor(harta, cv2.COLOR_BGR2RGB)
    return text_rezultat(rez, det.calibrare), desen, harta


def analizeaza_video(cale_video):
    if cale_video is None:
        return "Încarcă un video.", None, []
    try:
        det = ia_detectorul()
    except FileNotFoundError as eroare:
        return "Modelele nu sunt gata: " + str(eroare), None, []
    rez = det.analizeaza_video(cale_video)
    if not rez["ok"]:
        return "Nu am găsit nicio față în cadrele video-ului.", None, []

    # graficul: scorul fiecărui cadru, pragul și media
    numere_cadre = []
    scoruri = []
    for cadru in rez["cadre"]:
        numere_cadre.append(cadru["cadru"])
        scoruri.append(cadru["scor"])
    figura, ax = plt.subplots(figsize=(6, 2.6))
    ax.plot(numere_cadre, scoruri, "o-", label="scorul cadrului")
    ax.axhline(rez["prag"], color="black", linestyle="--", label="prag")
    ax.axhline(rez["scor"], color="red", label="media cadrelor")
    ax.set_xlabel("cadrul din video")
    ax.set_ylabel("scor fuzionat")
    ax.legend(fontsize=8)
    figura.tight_layout()
    plt.close(figura)                                  # Gradio o desenează singur; eu nu mai am nevoie de ea

    # galeria: fața din fiecare cadru, cu scorul ei
    galerie = []
    for cadru in rez["cadre"]:
        imagine = cv2.cvtColor(cadru["fata"], cv2.COLOR_BGR2RGB)
        eticheta = "cadrul " + str(cadru["cadru"]) + ": " + str(round(cadru["scor"], 2))
        galerie.append((imagine, eticheta))

    text = "## Verdict: " + rez["eticheta"] + "\n\n"
    text = text + "Cadre cu față analizate: " + str(len(rez["cadre"])) + ". "
    text = text + "Scorul video-ului e media cadrelor: **" + str(round(rez["scor"], 2)) + "**. "
    text = text + "Pragul pentru video: **" + str(round(rez["prag"], 2)) + "**."
    return text, figura, galerie


def lista_exemple(cate=4):
    """Primele imagini din date/test/real și date/test/fake, ca exemple pe care dai click."""
    exemple = []
    for clasa in ["real", "fake"]:
        folder = os.path.join(FOLDER_DATE, "test", clasa)
        if not os.path.isdir(folder):
            continue
        gasite = 0
        for nume_fisier in sorted(os.listdir(folder)):
            if este_imagine(nume_fisier) and gasite < cate:
                exemple.append([os.path.join(folder, nume_fisier)])
                gasite = gasite + 1
    return exemple


with gr.Blocks(title="Demo Tema") as demo:
    gr.Markdown("# Detector deepfake\n"
                "Ramura A măsoară 104 trăsături (granița feței, frecvență, dinți, ochi, piele) și le dă unui SVM. "
                "Ramura B e o rețea EfficientNet-B0. Scorurile se combină, iar un prag ales pe validare dă verdictul.")
    with gr.Tab("Imagine"):
        with gr.Row():
            with gr.Column():
                intrare = gr.Image(type="numpy", label="Imagine cu o față")
                buton = gr.Button("Analizează", variant="primary")
                exemple = lista_exemple()
                if len(exemple) > 0:
                    gr.Examples(exemple, inputs=intrare, label="Exemple din date/test")
            with gr.Column():
                rezultat = gr.Markdown()
                with gr.Row():
                    fata = gr.Image(label="Fața decupată și punctele găsite")
                    harta = gr.Image(label="Grad-CAM: unde s-a uitat rețeaua")
        buton.click(analizeaza, inputs=intrare, outputs=[rezultat, fata, harta])

    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column():
                intrare_video = gr.Video(label="Un video scurt cu o față")
                buton_video = gr.Button("Analizează video-ul", variant="primary")
            with gr.Column():
                rezultat_video = gr.Markdown()
                grafic = gr.Plot(label="Scorul fiecărui cadru")
        galerie = gr.Gallery(label="Fețele găsite în cadre", columns=8)
        buton_video.click(analizeaza_video, inputs=intrare_video, outputs=[rezultat_video, grafic, galerie])

    with gr.Tab("Cum funcționează"):
        gr.Markdown(
            "1. **Preprocesare:** MediaPipe găsește fața și 478 de puncte; fața e îndreptată și decupată la 256 x 256.\n"
            "2. **Ramura A:** 104 trăsături măsurate pe față -> SVM cu kernel RBF -> scorul A.\n"
            "3. **Ramura B:** EfficientNet-B0 -> scorul B. Grad-CAM arată zonele care au contat.\n"
            "4. **Fuziune:** fiecare scor devine z-score (cu media și abaterea de pe validare), apoi fac media.\n"
            "5. **Decizie:** scor mai mare sau egal cu pragul ales pe validare (Youden) -> FALS. La video: media cadrelor.\n\n")



if __name__ == "__main__":
    demo.launch(inbrowser=True)
