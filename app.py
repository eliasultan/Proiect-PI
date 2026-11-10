import os

import cv2

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import gradio as gr

from preprocesare import creeaza_detector, pregateste_fata, deseneaza_puncte

detector_fata = None


def analizeaza(imagine):
    global detector_fata
    if imagine is None:
        return "Incarca o imagine.", None
    if detector_fata is None:
        detector_fata = creeaza_detector()
    img = cv2.cvtColor(imagine, cv2.COLOR_RGB2BGR)
    fata, puncte = pregateste_fata(detector_fata, img)
    if fata is None:
        return "Nu am gasit nicio fata in imagine.", None
    desen = deseneaza_puncte(fata, puncte)
    desen = cv2.cvtColor(desen, cv2.COLOR_BGR2RGB)     # inapoi în RGB, pentru afișare
    text = "Am gasit o fata: " + str(len(puncte)) + " de puncte, decupata la "
    text = text + str(fata.shape[1]) + " x " + str(fata.shape[0]) + " pixeli."
    return text, desen


with gr.Blocks(title="Demo Tema 37") as demo:
    gr.Markdown("# Detector deepfake ·\nIncarcă o imagine cu o fata si apasa „Analizeaza”.")
    with gr.Row():
        with gr.Column():
            intrare = gr.Image(type="numpy", label="Imagine cu o fată")
            buton = gr.Button("Analizeaza", variant="primary")
        with gr.Column():
            rezultat = gr.Markdown()
            fata = gr.Image(label="Fata decupata si punctele gasite")
    buton.click(analizeaza, inputs=intrare, outputs=[rezultat, fata])


if __name__ == "__main__":
    demo.launch(inbrowser=True)
