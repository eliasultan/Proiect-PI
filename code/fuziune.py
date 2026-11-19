"""combin cele două scoruri (z-score) și decid cu un prag (Youden)."""
import os
import json
import numpy as np
from sklearn.metrics import roc_curve

from config import FOLDER_CACHE, FISIER_CALIBRARE
from date import citeste_tabel


def parametri_z(scoruri_validare):
    """Media și abaterea standard ale unui scor, calculate o singură dată, pe validare."""
    medie = float(np.mean(scoruri_validare))
    abatere = float(np.std(scoruri_validare)) + 1e-8      # + 1e-8: ca să nu împart niciodată la zero
    return {"medie": medie, "abatere": abatere}


def z(scor, parametri):
    """z-score: cu câte abateri standard e scorul peste medie. Așa scorurile A și B ajung pe aceeași scară."""
    return (scor - parametri["medie"]) / parametri["abatere"]


def fuzioneaza(scor_a, scor_b, calibrare):
    """Scorul final al unei fețe: media ponderată a z-score-urilor ramurilor care există."""
    suma = 0.0
    suma_ponderilor = 0.0
    if scor_a is not None and "A" in calibrare["ramuri"]:
        pondere = calibrare["ponderi"]["A"]
        suma = suma + pondere * z(scor_a, calibrare["z"]["A"])
        suma_ponderilor = suma_ponderilor + pondere
    if scor_b is not None and "B" in calibrare["ramuri"]:
        pondere = calibrare["ponderi"]["B"]
        suma = suma + pondere * z(scor_b, calibrare["z"]["B"])
        suma_ponderilor = suma_ponderilor + pondere
    if suma_ponderilor == 0:
        raise ValueError("Nu am niciun scor de combinat.")
    return suma / suma_ponderilor


def prag_youden(etichete, scoruri):
    """Pragul la care (falsuri prinse) − (reale acuzate pe nedrept) e cea mai mare."""
    fpr, tpr, praguri = roc_curve(etichete, scoruri)
    cel_mai_bun = 0
    for i in range(len(praguri)):
        if tpr[i] - fpr[i] > tpr[cel_mai_bun] - fpr[cel_mai_bun]:
            cel_mai_bun = i
    return float(praguri[cel_mai_bun]), float(tpr[cel_mai_bun]), float(fpr[cel_mai_bun])


def citeste_scorurile(ramuri):
    """Scorurile ramurilor (cache/scoruri_A.csv, cache/scoruri_B.csv), câte un dicționar pe față:
    fata, split, eticheta, grup, scor_A, scor_B. Păstrez doar fețele care au scor de la toate ramurile."""
    pe_fata = {}                                   # cheia: calea feței
    for r in ramuri:
        cale = os.path.join(FOLDER_CACHE, "scoruri_" + r + ".csv")
        for rand in citeste_tabel(cale):
            cheie = rand["fata"]
            if cheie not in pe_fata:
                pe_fata[cheie] = {"fata": rand["fata"], "split": rand["split"],
                                  "eticheta": int(rand["eticheta"]), "grup": rand["grup"]}
            pe_fata[cheie]["scor_" + r] = float(rand["scor_" + r])

    fete = []
    incomplete = 0
    for cheie in pe_fata:
        fata = pe_fata[cheie]
        are_tot = True
        for r in ramuri:
            if "scor_" + r not in fata:
                are_tot = False
        if are_tot:
            fete.append(fata)
        else:
            incomplete = incomplete + 1
    if incomplete > 0:
        print("ATENȚIE: scorurile A și B nu sunt pentru aceleași fețe. "
              "Rulează din nou antreneaza_svm.py și antreneaza_cnn.py.")
    return fete


def scoruri_fuzionate(fete, calibrare):
    """Scorul fuzionat pentru fiecare față din listă."""
    rezultat = []
    for fata in fete:
        scor_a = fata.get("scor_A")                # None dacă fața nu are scor A
        scor_b = fata.get("scor_B")
        rezultat.append(fuzioneaza(scor_a, scor_b, calibrare))
    return rezultat


def medii_pe_video(fete, scoruri):
    """Pentru video-uri (grupuri cu mai multe cadre): eticheta și media scorurilor cadrelor."""
    grupuri = {}                                   # grup -> {"eticheta": ..., "scoruri": [...]}
    for i in range(len(fete)):
        grup = fete[i]["grup"]
        if grup not in grupuri:
            grupuri[grup] = {"eticheta": fete[i]["eticheta"], "scoruri": []}
        grupuri[grup]["scoruri"].append(scoruri[i])

    etichete_video = []
    scoruri_video = []
    for grup in grupuri:
        lista = grupuri[grup]["scoruri"]
        if len(lista) > 1:                         # un grup cu un singur cadru e o poză obișnuită
            etichete_video.append(grupuri[grup]["eticheta"])
            scoruri_video.append(sum(lista) / len(lista))
    return etichete_video, scoruri_video


def salveaza_calibrare(calibrare, cale=FISIER_CALIBRARE):
    with open(cale, "w", encoding="utf-8") as fisier:
        json.dump(calibrare, fisier, indent=2)


def incarca_calibrare(cale=FISIER_CALIBRARE):
    with open(cale, encoding="utf-8") as fisier:
        return json.load(fisier)
