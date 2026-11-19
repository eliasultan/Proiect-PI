import os
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")                              # desenez graficul direct în fișier, fără fereastră
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, confusion_matrix

from config import FOLDER_PROIECT, FOLDER_RAPOARTE, SEED
from date import citeste_fetele
from fuziune import incarca_calibrare, citeste_scorurile, scoruri_fuzionate, medii_pe_video
from preprocesare import citeste_imagine

NUME_RAMURI = {"A": "Ramura A (SVM)", "B": "Ramura B (EfficientNet-B0)", "fuzionat": "Fuzionat"}


#degradări pentru testul de robustețe
def jpeg_50(imagine):
    """Recomprim imaginea JPEG la calitate 50."""
    rezultat = cv2.imencode(".jpg", imagine, [cv2.IMWRITE_JPEG_QUALITY, 50])
    octeti = rezultat[1]
    return cv2.imdecode(octeti, cv2.IMREAD_COLOR)


def zgomot(imagine):
    """Adaug zgomot gaussian, ca la o cameră proastă."""
    zg = np.random.normal(0, 8, imagine.shape)
    stricata = imagine + zg
    stricata = np.clip(stricata, 0, 255)           # valorile trebuie să rămână între 0 și 255
    return stricata.astype(np.uint8)


def blur(imagine):
    """Încețoșez imaginea (blur gaussian, sigma 1,5)."""
    return cv2.GaussianBlur(imagine, (0, 0), 1.5)


DEGRADARI = {"fără degradare": None, "JPEG calitate 50": jpeg_50,
             "zgomot gaussian (sigma 8)": zgomot, "blur (sigma 1,5)": blur}


def metrici(etichete, scoruri, prag):
    """AUC, plus acuratețea, precizia, recall-ul și matricea de confuzie pentru pragul dat."""
    predictii = []
    for scor in scoruri:
        if scor >= prag:                           # la fel ca în detector.py
            predictii.append(1)
        else:
            predictii.append(0)
    rezultat = {}
    rezultat["AUC"] = roc_auc_score(etichete, scoruri)
    rezultat["acuratete"] = accuracy_score(etichete, predictii)
    rezultat["precizie"] = precision_score(etichete, predictii, zero_division=0)
    rezultat["recall"] = recall_score(etichete, predictii, zero_division=0)
    rezultat["matrice"] = confusion_matrix(etichete, predictii, labels=[0, 1])
    return rezultat


def robustete(test, cate):
    """Trec imaginile originale din test, stricate în mai multe feluri, prin tot sistemul."""
    from detector import Detector                  # îl import aici: încarcă modelele doar dacă e nevoie
    detector = Detector()

    # calea imaginii originale pentru fiecare față (din cache/index.csv)
    originala = {}
    for fata in citeste_fetele():
        originala[fata["fata"]] = fata["cale"]

    # cel mult `cate` imagini din fiecare clasă
    alese = []
    numar = {0: 0, 1: 0}
    for fata in test:
        if numar[fata["eticheta"]] < cate:
            alese.append(fata)
            numar[fata["eticheta"]] = numar[fata["eticheta"]] + 1

    rezultat = {}
    for nume in DEGRADARI:
        functie = DEGRADARI[nume]
        np.random.seed(SEED)                       # același zgomot la fiecare rulare
        etichete = []
        scoruri = []
        for fata in alese:
            cale = os.path.join(FOLDER_PROIECT, originala[fata["fata"]])   # imaginea originală, nu fața decupată
            imagine = citeste_imagine(cale)
            if functie is not None:
                imagine = functie(imagine)
            rez = detector.analizeaza_imagine(imagine, cu_harta=False)
            if rez["ok"]:                          # dacă degradarea a stricat fața, MediaPipe poate să n-o mai găsească
                etichete.append(fata["eticheta"])
                scoruri.append(rez["scor"])
        auc = roc_auc_score(etichete, scoruri)
        rezultat[nume] = [auc, len(etichete)]
        print("  ", nume, ": AUC =", round(auc, 4), "(imagini cu față găsită:", len(etichete), ")")
    detector.inchide()
    return rezultat


def deseneaza_roc(y, coloane, cale):
    """Curbele ROC ale ramurilor și ale fuziunii, într-un singur grafic salvat ca PNG."""
    plt.figure(figsize=(5, 5))
    for nume in coloane:
        fpr, tpr, praguri = roc_curve(y, coloane[nume])
        auc = roc_auc_score(y, coloane[nume])
        plt.plot(fpr, tpr, label=NUME_RAMURI[nume] + " (AUC " + str(round(auc, 3)) + ")")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)  # diagonala: ghicit la întâmplare
    plt.xlabel("rata de fals pozitive (reale acuzate)")
    plt.ylabel("rata de adevărat pozitive (falsuri prinse)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(cale, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robustete", action="store_true")
    parser.add_argument("--cate", type=int, default=200, help="imagini pe clasă pentru testul de robustețe")
    args = parser.parse_args()

    calibrare = incarca_calibrare()
    test = []
    for fata in citeste_scorurile(calibrare["ramuri"]):
        if fata["split"] == "test":
            test.append(fata)
    if len(test) == 0:
        print("Nu am imagini de test. Pune imagini în date/test/real și date/test/fake.")
        return

    # etichetele și scorurile fiecărei ramuri, plus scorul fuzionat
    y = []
    for fata in test:
        y.append(fata["eticheta"])
    coloane = {}
    for r in calibrare["ramuri"]:
        coloane[r] = []
        for fata in test:
            coloane[r].append(fata["scor_" + r])
    coloane["fuzionat"] = scoruri_fuzionate(test, calibrare)
    os.makedirs(FOLDER_RAPOARTE, exist_ok=True)

    #AUC pentru fiecare ramură și pentru fuziune
    reale = y.count(0)                             # câte etichete 0 sunt în listă
    false = y.count(1)
    raport = []
    raport.append("# Rezultate pe setul de test\n")
    raport.append("Imagini de test: " + str(len(test)) + " (" + str(reale) + " reale, " + str(false) + " false)\n")
    raport.append("| Metodă | AUC |")
    raport.append("|---|---|")
    for nume in coloane:
        auc = roc_auc_score(y, coloane[nume])
        raport.append("| " + NUME_RAMURI[nume] + " | " + str(round(auc, 4)) + " |")
        print(NUME_RAMURI[nume], ": AUC =", round(auc, 4))

    #decizia finală, cu pragul ales pe validare
    m = metrici(y, coloane["fuzionat"], calibrare["prag"])
    matrice = m["matrice"]                         # [[reale corecte, reale acuzate], [falsuri ratate, falsuri prinse]]
    raport.append("\n## Decizia finală (prag " + str(round(calibrare["prag"], 3)) + ", ales pe validare)\n")
    raport.append("| Acuratețe | Precizie | Recall |")
    raport.append("|---|---|---|")
    raport.append("| " + str(round(m["acuratete"], 4)) + " | " + str(round(m["precizie"], 4)) + " | "
                  + str(round(m["recall"], 4)) + " |\n")
    raport.append("| | prezis real | prezis fals |")
    raport.append("|---|---|---|")
    raport.append("| **real** | " + str(matrice[0][0]) + " | " + str(matrice[0][1]) + " |")
    raport.append("| **fals** | " + str(matrice[1][0]) + " | " + str(matrice[1][1]) + " |")
    print("Acuratețe", round(m["acuratete"], 4), "· precizie", round(m["precizie"], 4), "· recall", round(m["recall"], 4))

    #pe video (dacă imaginile sunt cadre din video-uri): scorul unui video = media cadrelor
    etichete_video, scoruri_video = medii_pe_video(test, coloane["fuzionat"])
    if 0 in etichete_video and 1 in etichete_video:
        mv = metrici(etichete_video, scoruri_video, calibrare["prag_video"])
        raport.append("\n## Pe video (video-uri: " + str(len(etichete_video)) + "; scorul unui video = media cadrelor)\n")
        raport.append("| AUC | Acuratețe |")
        raport.append("|---|---|")
        raport.append("| " + str(round(mv["AUC"], 4)) + " | " + str(round(mv["acuratete"], 4)) + " |")
        print("Pe video: AUC", round(mv["AUC"], 4), "· acuratețe", round(mv["acuratete"], 4))

    #curbele ROC
    deseneaza_roc(y, coloane, os.path.join(FOLDER_RAPOARTE, "roc_test.png"))

    #robustețe (opțional)
    if args.robustete:
        print("Robustețe:")
        rob = robustete(test, args.cate)
        raport.append("\n## Robustețe (tot sistemul, pe imaginile originale stricate)\n")
        raport.append("| Degradare | AUC | Imagini |")
        raport.append("|---|---|---|")
        for nume in rob:
            raport.append("| " + nume + " | " + str(round(rob[nume][0], 4)) + " | " + str(rob[nume][1]) + " |")

    with open(os.path.join(FOLDER_RAPOARTE, "rezultate_test.md"), "w", encoding="utf-8") as fisier:
        for rand in raport:
            fisier.write(rand + "\n")
    print("Am salvat rapoarte/rezultate_test.md și rapoarte/roc_test.png")


if __name__ == "__main__":
    main()
