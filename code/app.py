
import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"      # Gradio să nu trimită statistici de utilizare
import cv2
import gradio as gr
import joblib

from preprocesare import creeaza_detector, pregateste_fata, deseneaza_puncte
from config import FISIER_SVM
from trasaturi import vector_trasaturi
from ramura_a import scor_svm
from config import FISIER_CNN
from ramura_b import incarca_model, scor_o_fata


detector_fata = None
model_svm = None
model_cnn = None


def rand_tabel(nume, scor):
    if scor > 0:
        spre = "fals"
    else:
        spre = "real"
    return "| " + nume + " | " + str(round(scor, 3)) + " | " + spre + " |\n"


def analizeaza(imagine):
    global detector_fata, model_svm, model_cnn
    if imagine is None:
        return "Incarca o imagine.", None
    if detector_fata is None:
        detector_fata = creeaza_detector()
    img = cv2.cvtColor(imagine, cv2.COLOR_RGB2BGR)     # Gradio dă imaginea în RGB, OpenCV lucrează în BGR
    fata, puncte = pregateste_fata(detector_fata, img)
    if fata is None:
        return "Nu am gasit nicio fata in imagine.", None
    desen = deseneaza_puncte(fata, puncte)
    desen = cv2.cvtColor(desen, cv2.COLOR_BGR2RGB)
    text = "Am gasit o fata: " + str(len(puncte)) + " de puncte, decupata la "
    text = text + str(fata.shape[1]) + " x " + str(fata.shape[0]) + " pixeli."

    if not os.path.isfile(FISIER_SVM):
        return text + "\n\nSVM-ul nu e antrenat inca. Ruleaza: `python antreneaza_svm.py`", desen
    if model_svm is None:
        salvat = joblib.load(FISIER_SVM)
        model_svm = salvat["model"]
    vector = vector_trasaturi(fata, puncte)
    scoruri = scor_svm(model_svm, [vector])            # SVM-ul primeste o lista de vectori
    scor_a = scoruri[0]
    text = text + "\n\n| Ramura | Scor | Spre |\n|---|---|---|\n"
    text = text + rand_tabel("A · SVM pe trasaturi", scor_a)
    if os.path.isfile(FISIER_CNN):
        if model_cnn is None:
            model_cnn = incarca_model(FISIER_CNN)
        scor_b = scor_o_fata(model_cnn, fata)
        text = text + rand_tabel("B · EfficientNet-B0", scor_b)
    else:
        text = text + "\nRețeaua nu e antrenata inca. Ruleaza: `python antreneaza_cnn.py`\n"
    text = text + "\nDeocamdata fiecare ramură decide singura, dupa semnul scorului. Fuziunea vine mai tarziu."
    return text, desen


with gr.Blocks(title="Deepfake detector") as demo:
    gr.Markdown("# Detector deepfake \nIncarca o imagine cu o fata si apasa „Analizeaza”.")
    with gr.Row():
        with gr.Column():
            intrare = gr.Image(type="numpy", label="Imagine cu o fata")
            buton = gr.Button("Analizeaza", variant="primary")
        with gr.Column():
            rezultat = gr.Markdown()
            fata = gr.Image(label="Fata decupata si punctele gasite")
    buton.click(analizeaza, inputs=intrare, outputs=[rezultat, fata])


if __name__ == "__main__":
    demo.launch(inbrowser=True)
