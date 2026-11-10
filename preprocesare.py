import os
import math
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarksConnections

from config import MODEL_MEDIAPIPE, MARIME_FATA, MARGINE


def citeste_imagine(cale):
    octeti = np.fromfile(cale, dtype=np.uint8)
    imagine = cv2.imdecode(octeti, cv2.IMREAD_COLOR)
    return imagine


def salveaza_imagine(cale, imagine):
    folder = os.path.dirname(cale)
    if folder != "":                                  # daca in cale apare si un folder, il creez
        os.makedirs(folder, exist_ok=True)
    rezultat = cv2.imencode(".png", imagine)
    octeti = rezultat[1]
    octeti.tofile(cale)


def puncte_din_conexiuni(conexiuni):
    puncte = []
    for conexiune in conexiuni:
        if conexiune.start not in puncte:
            puncte.append(conexiune.start)
        if conexiune.end not in puncte:
            puncte.append(conexiune.end)
    puncte.sort()
    return puncte


# indicii punctelor de care am nevoie
CONTUR = []                                           # de jur imprejurul fetei, in ordine
for conexiune in FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL:
    CONTUR.append(conexiune.start)
IRIS_STANG = puncte_din_conexiuni(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS)
IRIS_DREPT = puncte_din_conexiuni(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS)
BUZE_INTERIOR = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                 324, 318, 402, 317, 14, 87, 178, 88, 95]   # interiorul gurii, in ordine



def centrul(puncte, indici):
    suma_x = 0.0
    suma_y = 0.0
    for i in indici:
        suma_x = suma_x + float(puncte[i][0])
        suma_y = suma_y + float(puncte[i][1])
    return suma_x / len(indici), suma_y / len(indici)


def distanta(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def stanga_dreapta(p, q):
    if p[0] <= q[0]:
        return p, q
    return q, p


def puncte_intregi(puncte, indici):
    lista = []
    for i in indici:
        lista.append([int(puncte[i][0]), int(puncte[i][1])])
    return np.array(lista, dtype=np.int32)


def roteste_puncte(puncte, M):
    rotite = []
    for punct in puncte:
        x = punct[0]
        y = punct[1]
        x_nou = M[0][0] * x + M[0][1] * y + M[0][2]
        y_nou = M[1][0] * x + M[1][1] * y + M[1][2]
        rotite.append([x_nou, y_nou])
    return np.array(rotite, dtype=np.float32)


def dreptunghi(puncte, indici):
    xs = []
    ys = []
    for i in indici:
        xs.append(puncte[i][0])
        ys.append(puncte[i][1])
    return min(xs), min(ys), max(xs), max(ys)

def creeaza_detector(cale_model=MODEL_MEDIAPIPE):
    if not os.path.isfile(cale_model):
        raise FileNotFoundError("Lipseste modelul MediaPipe. Rulează intai: python descarca_model.py")
    optiuni_model = BaseOptions(model_asset_path=cale_model)
    optiuni = vision.FaceLandmarkerOptions(base_options=optiuni_model,
                                           running_mode=vision.RunningMode.IMAGE,
                                           num_faces=1)
    return vision.FaceLandmarker.create_from_options(optiuni)


def gaseste_puncte(detector, imagine):
    rgb = cv2.cvtColor(imagine, cv2.COLOR_BGR2RGB)
    imagine_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    rezultat = detector.detect(imagine_mp)
    if len(rezultat.face_landmarks) == 0:
        return None
    inaltime = imagine.shape[0]
    latime = imagine.shape[1]
    puncte = []
    for p in rezultat.face_landmarks[0]:
        puncte.append([p.x * latime, p.y * inaltime])
    return np.array(puncte, dtype=np.float32)

def aliniaza_si_decupeaza(imagine, puncte):
    #rotesc imaginea ca sa fie orizontala
    ochi_1 = centrul(puncte, IRIS_STANG)
    ochi_2 = centrul(puncte, IRIS_DREPT)
    ochi_stanga, ochi_dreapta = stanga_dreapta(ochi_1, ochi_2)
    dx = ochi_dreapta[0] - ochi_stanga[0]
    dy = ochi_dreapta[1] - ochi_stanga[1]
    unghi = math.degrees(math.atan2(dy, dx))          # unghiul liniei dintre ochi
    centru = centrul(puncte, CONTUR)                  # rotesc in jurul centrului feței
    M = cv2.getRotationMatrix2D(centru, unghi, 1.0)
    inaltime = imagine.shape[0]
    latime = imagine.shape[1]
    rotita = cv2.warpAffine(imagine, M, (latime, inaltime), borderMode=cv2.BORDER_REFLECT)
    puncte_rotite = roteste_puncte(puncte, M)

    # un patrat in jurul conturului fetei, cu margine
    x_min, y_min, x_max, y_max = dreptunghi(puncte_rotite, CONTUR)
    latura = max(x_max - x_min, y_max - y_min)
    latura = latura + 2 * MARGINE * latura
    latura = int(round(latura))
    stanga = int(round((x_min + x_max) / 2 - latura / 2))
    sus = int(round((y_min + y_max) / 2 - latura / 2))

    #bordez imaginea
    bordata = cv2.copyMakeBorder(rotita, latura, latura, latura, latura, cv2.BORDER_REFLECT)
    x0 = stanga + latura
    y0 = sus + latura
    decupat = bordata[y0:y0 + latura, x0:x0 + latura]
    fata = cv2.resize(decupat, (MARIME_FATA, MARIME_FATA), interpolation=cv2.INTER_AREA)

    #  mut punctele in coordonatele fetei decupate
    scara = MARIME_FATA / latura
    puncte_fata = []
    for punct in puncte_rotite:
        x = (punct[0] - stanga) * scara
        y = (punct[1] - sus) * scara
        puncte_fata.append([x, y])
    return fata, np.array(puncte_fata, dtype=np.float32)


def pregateste_fata(detector, imagine):
    puncte = gaseste_puncte(detector, imagine)
    if puncte is None:
        return None, None
    return aliniaza_si_decupeaza(imagine, puncte)


def raza_medie(puncte, indici, centru):
    suma = 0.0
    for i in indici:
        suma = suma + distanta(puncte[i], centru)
    return suma / len(indici)


def deseneaza_puncte(fata, puncte):
    desen = fata.copy()
    cv2.polylines(desen, [puncte_intregi(puncte, CONTUR)], True, (0, 200, 255), 1)
    for iris in [IRIS_STANG, IRIS_DREPT]:
        centru = centrul(puncte, iris)
        raza = raza_medie(puncte, iris, centru)
        if raza < 2:
            raza = 2
        cv2.circle(desen, (int(centru[0]), int(centru[1])), int(raza), (80, 255, 120), 1)
    cv2.polylines(desen, [puncte_intregi(puncte, BUZE_INTERIOR)], True, (255, 140, 80), 1)
    return desen
