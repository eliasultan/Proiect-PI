"""Gaseste fata cu MediaPipe o îndreapta și o decupeaza la 256 x 256."""
import os
import math
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarksConnections

from config import MODEL_MEDIAPIPE, MARIME_FATA, MARGINE

"""Citesc imaginea ca octeți"""
def citeste_imagine(cale):
    octeti = np.fromfile(cale, dtype=np.uint8)
    imagine = cv2.imdecode(octeti, cv2.IMREAD_COLOR)
    return imagine

"""Salvez imaginea în format PNG."""
def salveaza_imagine(cale, imagine):
    folder = os.path.dirname(cale)
    if folder != "":                                 
        os.makedirs(folder, exist_ok=True)
    rezultat = cv2.imencode(".png", imagine)          
    octeti = rezultat[1]
    octeti.tofile(cale)

"""Transformare in lista a punctelor din MediaPipe"""
def puncte_din_conexiuni(conexiuni):
    puncte = []
    for conexiune in conexiuni:
        if conexiune.start not in puncte:
            puncte.append(conexiune.start)
        if conexiune.end not in puncte:
            puncte.append(conexiune.end)
    puncte.sort()
    return puncte

CONTUR = []                                           
for conexiune in FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL:
    CONTUR.append(conexiune.start)
IRIS_STANG = puncte_din_conexiuni(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS)
IRIS_DREPT = puncte_din_conexiuni(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS)
BUZE_INTERIOR = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                 324, 318, 402, 317, 14, 87, 178, 88, 95] 

"""Centrul unei zone a feței"""
def centrul(puncte, indici):
    suma_x = 0.0
    suma_y = 0.0
    for i in indici:
        suma_x = suma_x + float(puncte[i][0])      
        suma_y = suma_y + float(puncte[i][1])
    return suma_x / len(indici), suma_y / len(indici)

"""Distanța dintre două puncte (x, y)."""
def distanta(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])

"""Primește două puncte (x, y) și le întoarce în ordine"""
def stanga_dreapta(p, q):
    if p[0] <= q[0]:
        return p, q
    return q, p

"""Punctele cu indicii dați în forma cerută de funcțiile de desenat din OpenCV."""
def puncte_intregi(puncte, indici):
    lista = []
    for i in indici:
        lista.append([int(puncte[i][0]), int(puncte[i][1])])
    return np.array(lista, dtype=np.int32)

"""Aplic punctelor aceeași rotație ca imaginii: x' = a·x + b·y + c,  y' = d·x + e·y + f."""
def roteste_puncte(puncte, M):
    rotite = []
    for punct in puncte:
        x = punct[0]
        y = punct[1]
        x_nou = M[0][0] * x + M[0][1] * y + M[0][2]
        y_nou = M[1][0] * x + M[1][1] * y + M[1][2]
        rotite.append([x_nou, y_nou])
    return np.array(rotite, dtype=np.float32)

"""Cel mai mic dreptunghi care cuprinde punctele date: x_min, y_min, x_max, y_max."""
def dreptunghi(puncte, indici):
    xs = []
    ys = []
    for i in indici:
        xs.append(puncte[i][0])
        ys.append(puncte[i][1])
    return min(xs), min(ys), max(xs), max(ys)


"""MediaPipe"""
def creeaza_detector(cale_model=MODEL_MEDIAPIPE):
    if not os.path.isfile(cale_model):
        raise FileNotFoundError("Lipsește modelul MediaPipe. Rulează întâi: python descarca_model.py")
    optiuni_model = BaseOptions(model_asset_path=cale_model)
    optiuni = vision.FaceLandmarkerOptions(base_options=optiuni_model,
                                           running_mode=vision.RunningMode.IMAGE,
                                           num_faces=1)
    return vision.FaceLandmarker.create_from_options(optiuni)

"""Cele 478 de puncte ale fetei, în pixeli (un array de 478 x 2), sau None dacă nu e nicio față."""
def gaseste_puncte(detector, imagine):
    rgb = cv2.cvtColor(imagine, cv2.COLOR_BGR2RGB)   # OpenCV citește BGR, MediaPipe vrea RGB
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


"""aliniere și decupare"""
def aliniaza_si_decupeaza(imagine, puncte):
    #rotesc imaginea astfel încât ochii să fie pe orizontală
    ochi_1 = centrul(puncte, IRIS_STANG)
    ochi_2 = centrul(puncte, IRIS_DREPT)
    ochi_stanga, ochi_dreapta = stanga_dreapta(ochi_1, ochi_2)
    dx = ochi_dreapta[0] - ochi_stanga[0]
    dy = ochi_dreapta[1] - ochi_stanga[1]
    unghi = math.degrees(math.atan2(dy, dx))          
    centru = centrul(puncte, CONTUR)                  
    M = cv2.getRotationMatrix2D(centru, unghi, 1.0)
    inaltime = imagine.shape[0]
    latime = imagine.shape[1]
    rotita = cv2.warpAffine(imagine, M, (latime, inaltime), borderMode=cv2.BORDER_REFLECT)
    puncte_rotite = roteste_puncte(puncte, M)

    #un pătrat în jurul conturului feței, cu margine
    x_min, y_min, x_max, y_max = dreptunghi(puncte_rotite, CONTUR)
    latura = max(x_max - x_min, y_max - y_min)        
    latura = latura + 2 * MARGINE * latura            
    latura = int(round(latura))
    stanga = int(round((x_min + x_max) / 2 - latura / 2))
    sus = int(round((y_min + y_max) / 2 - latura / 2))

    #bordez imaginea, ca pătratul să nu iasă din ea când fața e lângă margine
    bordata = cv2.copyMakeBorder(rotita, latura, latura, latura, latura, cv2.BORDER_REFLECT)
    x0 = stanga + latura                              
    y0 = sus + latura
    decupat = bordata[y0:y0 + latura, x0:x0 + latura]
    fata = cv2.resize(decupat, (MARIME_FATA, MARIME_FATA), interpolation=cv2.INTER_AREA)

    #mut punctele în coordonatele feței decupate
    scara = MARIME_FATA / latura
    puncte_fata = []
    for punct in puncte_rotite:
        x = (punct[0] - stanga) * scara
        y = (punct[1] - sus) * scara
        puncte_fata.append([x, y])
    return fata, np.array(puncte_fata, dtype=np.float32)

"""fața decupată și punctele ei"""
def pregateste_fata(detector, imagine):
    puncte = gaseste_puncte(detector, imagine)
    if puncte is None:
        return None, None
    return aliniaza_si_decupeaza(imagine, puncte)
