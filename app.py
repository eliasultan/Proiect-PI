import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"      # Gradio să nu trimită statistici de utilizare
import cv2
import gradio as gr
import joblib

from preprocesare import creeaza_detector, pregateste_fata, deseneaza_puncte
from config import FISIER_SVM
from trasaturi import vector_trasaturi
from ramura_a import scor_svm

detector_fata = None
model_svm = None

"""Un rând din tabelul cu scoruri"""
def rand_tabel(nume, scor):
    if scor > 0:
        spre = "fals"
    else:
        spre = "real"
    return "| " + nume + " | " + str(round(scor, 3)) + " | " + spre + " |\n"


def analizeaza(imagine):
    global detector_fata, model_svm
    if imagine is None:
        return "Încarcă o imagine.", None
    if detector_fata is None:
        detector_fata = creeaza_detector()
    img = cv2.cvtColor(imagine, cv2.COLOR_RGB2BGR)     # Gradio dă imaginea în RGB, OpenCV lucrează în BGR
    fata, puncte = pregateste_fata(detector_fata, img)
    if fata is None:
        return "Nu am găsit nicio față în imagine.", None
    desen = deseneaza_puncte(fata, puncte)
    desen = cv2.cvtColor(desen, cv2.COLOR_BGR2RGB)     # înapoi în RGB, pentru afișare
    text = "Am găsit o față: " + str(len(puncte)) + " de puncte, decupată la "
    text = text + str(fata.shape[1]) + " x " + str(fata.shape[0]) + " pixeli."

    if not os.path.isfile(FISIER_SVM):
        return text + "\n\nSVM-ul nu e antrenat încă. Rulează: `python antreneaza_svm.py`", desen
    if model_svm is None:
        salvat = joblib.load(FISIER_SVM)
        model_svm = salvat["model"]
    vector = vector_trasaturi(fata, puncte)
    scoruri = scor_svm(model_svm, [vector])            # SVM-ul primește o listă de vectori; eu am unul singur
    scor_a = scoruri[0]
    text = text + "\n\n| Ramură | Scor | Spre |\n|---|---|---|\n"
    text = text + rand_tabel("A · SVM pe trăsături", scor_a)
    text = text + "\nDeocamdată fiecare ramură decide singură, după semnul scorului. Fuziunea vine mai târziu."
    return text, desen

with gr.Blocks(title="Demo Tema") as demo:
    gr.Markdown("# Detector deepfake\nÎncarcă o imagine cu o față și apasă „Analizează”.")
    with gr.Row():
        with gr.Column():
            intrare = gr.Image(type="numpy", label="Imagine cu o față")
            buton = gr.Button("Analizează", variant="primary")
        with gr.Column():
            rezultat = gr.Markdown()
            fata = gr.Image(label="Fața decupată și punctele găsite")
    buton.click(analizeaza, inputs=intrare, outputs=[rezultat, fata])


if __name__ == "__main__":
    demo.launch(inbrowser=True)
