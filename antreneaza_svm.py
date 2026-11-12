import os
import math
import joblib
import numpy as np

from config import FOLDER_PROIECT, FOLDER_CACHE, FOLDER_MODELE, FISIER_SVM
from date import citeste_fetele, citeste_tabel, scrie_tabel
from preprocesare import citeste_imagine
from trasaturi import vector_trasaturi, NUME_TRASATURI
from ramura_a import antreneaza_svm, scor_svm, date_pentru

"""Trăsăturile din cache/trasaturi.csv"""
def citeste_trasaturile_salvate(fete):
    cale = os.path.join(FOLDER_CACHE, "trasaturi.csv")
    if not os.path.isfile(cale):
        return None
    randuri = citeste_tabel(cale)
    if len(randuri) == 0:
        return None
    coloane = list(randuri[0].keys())
    if coloane != ["fata"] + NUME_TRASATURI:
        return None                                # lista de trăsături s-a schimbat
    trasaturi = {}
    for rand in randuri:
        vector = []
        for nume in NUME_TRASATURI:
            vector.append(float(rand[nume]))       # din CSV totul vine ca text; îl fac număr
        trasaturi[rand["fata"]] = vector
    for fata in fete:
        if fata["fata"] not in trasaturi:
            return None                            # a aparut o fata noua
    return trasaturi

"""Un dicționar: calea feței -> lista celor N trăsături"""
def calculeaza_trasaturile(fete):
    trasaturi = citeste_trasaturile_salvate(fete)
    if trasaturi is not None:
        print("Folosesc trăsăturile deja calculate din cache/trasaturi.csv")
        return trasaturi
    print("Calculez trăsăturile (prima dată, sau lista de trăsături ori fețele s-au schimbat).")

    trasaturi = {}
    randuri = []
    for i in range(len(fete)):
        cale_fata = fete[i]["fata"]
        fata = citeste_imagine(os.path.join(FOLDER_PROIECT, cale_fata))
        puncte = np.load(os.path.join(FOLDER_PROIECT, fete[i]["puncte"]))
        vector = vector_trasaturi(fata, puncte)
        trasaturi[cale_fata] = vector

        rand = {"fata": cale_fata}                 # un rând din CSV: calea și cele N valori
        for j in range(len(NUME_TRASATURI)):
            rand[NUME_TRASATURI[j]] = vector[j]
        randuri.append(rand)
        if (i + 1) % 100 == 0 or i + 1 == len(fete):
            print("   trăsături:", i + 1, "/", len(fete))
    scrie_tabel(os.path.join(FOLDER_CACHE, "trasaturi.csv"), randuri, ["fata"] + NUME_TRASATURI)
    return trasaturi

"""Afișez trasaturile care lipsesc cel mai des."""
def afiseaza_lipsurile(trasaturi):
    perechi = []
    for j in range(len(NUME_TRASATURI)):
        lipsesc = 0
        for vector in trasaturi.values():
            if math.isnan(vector[j]):
                lipsesc = lipsesc + 1
        procent = 100 * lipsesc / len(trasaturi)
        perechi.append((procent, NUME_TRASATURI[j]))
    perechi.sort(reverse=True)                     # sortez după procent, de la cel mai mare
    print("Trăsăturile care lipsesc cel mai des:")
    for procent, nume in perechi[:4]:
        print("  ", nume, ":", round(procent, 1), "% din fețe")


def main():
    fete = citeste_fetele()                        # doar imaginile în care s-a găsit o față
    trasaturi = calculeaza_trasaturile(fete)
    afiseaza_lipsurile(trasaturi)

    X_train, y_train = date_pentru(fete, trasaturi, "train")
    X_val, y_val = date_pentru(fete, trasaturi, "val")
    print("Antrenez SVM-ul (fețe de antrenare:", len(X_train), ", fețe de validare pentru alegerea lui C:", len(X_val), ")")
    model, C, auc = antreneaza_svm(X_train, y_train, X_val, y_val)
    print("C ales:", C, "(AUC pe validare", round(auc, 4), ")")

    os.makedirs(FOLDER_MODELE, exist_ok=True)
    salvat = {"model": model, "coloane": NUME_TRASATURI, "C": C}
    joblib.dump(salvat, FISIER_SVM)
    print("Am salvat modelul în", FISIER_SVM)

    # scorul A pentru fețele din validare și test
    rezultat = []
    for split in ["val", "test"]:
        X, y = date_pentru(fete, trasaturi, split)
        if len(X) == 0:
            continue
        scoruri = scor_svm(model, X)
        k = 0
        for fata in fete:
            if fata["split"] == split:
                rand = {"fata": fata["fata"], "split": split, "eticheta": fata["eticheta"], "grup": fata["grup"]}
                rand["scor_A"] = float(scoruri[k])
                rezultat.append(rand)
                k = k + 1
    scrie_tabel(os.path.join(FOLDER_CACHE, "scoruri_A.csv"), rezultat, ["fata", "split", "eticheta", "grup", "scor_A"])
    print("Am salvat scorurile în cache/scoruri_A.csv")


if __name__ == "__main__":
    main()
