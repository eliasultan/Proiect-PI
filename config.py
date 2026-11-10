import os

FOLDER_PROIECT = os.path.dirname(os.path.abspath(__file__))
FOLDER_DATE = os.path.join(FOLDER_PROIECT, "date")
FOLDER_CACHE = os.path.join(FOLDER_PROIECT, "cache")

EXTENSII_IMAGINI = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
SEED = 42

# preprocesare (etapa 2)
FOLDER_MODELE = os.path.join(FOLDER_PROIECT, "modele")
MODEL_MEDIAPIPE = os.path.join(FOLDER_MODELE, "face_landmarker.task")
URL_MEDIAPIPE = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MARIME_FATA = 256              # fata decupata - 256 x 256 pixeli
MARGINE = 0.30                 # 30% margine in jurul conturului fetei
