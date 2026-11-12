"""Descarca modelul MediaPipe pt gasirea fetelor."""
import os
import urllib.request

from config import FOLDER_MODELE, MODEL_MEDIAPIPE, URL_MEDIAPIPE

if os.path.isfile(MODEL_MEDIAPIPE):
    print("Modelul există deja:", MODEL_MEDIAPIPE)
else:
    os.makedirs(FOLDER_MODELE, exist_ok=True)
    print("Descarc", URL_MEDIAPIPE)
    urllib.request.urlretrieve(URL_MEDIAPIPE, MODEL_MEDIAPIPE)
    marime = os.path.getsize(MODEL_MEDIAPIPE) / 1000000
    print("Gata:", MODEL_MEDIAPIPE, "(", round(marime, 1), "MB )")
