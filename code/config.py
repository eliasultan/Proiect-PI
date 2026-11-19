"""Setările proiectului"""
import os

FOLDER_PROIECT = os.path.dirname(os.path.abspath(__file__))
FOLDER_DATE = os.path.join(FOLDER_PROIECT, "date")
FOLDER_CACHE = os.path.join(FOLDER_PROIECT, "cache")

EXTENSII_IMAGINI = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
SEED = 42                      

FOLDER_MODELE = os.path.join(FOLDER_PROIECT, "modele")
MODEL_MEDIAPIPE = os.path.join(FOLDER_MODELE, "face_landmarker.task")
URL_MEDIAPIPE = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MARIME_FATA = 256              
MARGINE = 0.30                 
FISIER_SVM = os.path.join(FOLDER_MODELE, "svm.joblib")
FISIER_CNN = os.path.join(FOLDER_MODELE, "cnn.pt")
MARIME_CNN = 224               # mărimea intrării pentru EfficientNet-B0
FISIER_CALIBRARE = os.path.join(FOLDER_MODELE, "calibrare.json")
FOLDER_RAPOARTE = os.path.join(FOLDER_PROIECT, "rapoarte")
FOLDER_VIDEO = os.path.join(FOLDER_PROIECT, "date_video")
CADRE_VIDEO = 16               # câte cadre analizez dintr-un video
EXTENSII_VIDEO = [".mp4", ".avi", ".mov", ".mkv", ".webm"]