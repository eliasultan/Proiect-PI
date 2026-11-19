"""Calculez media și abaterea fiecărui scor pe validare, combin scorurile și aleg pragul (Youden)."""
import os
import argparse
from sklearn.metrics import roc_auc_score

from config import FOLDER_CACHE, FOLDER_MODELE
from fuziune import parametri_z, prag_youden, salveaza_calibrare, citeste_scorurile
from fuziune import scoruri_fuzionate, medii_pe_video

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ponderi", default="egale", choices=["egale", "auto"])
    args = parser.parse_args()

    #ce ramuri am antrenat? (cele care au fișierul cu scoruri)
    ramuri = []
    for r in ["A", "B"]:
        if os.path.isfile(os.path.join(FOLDER_CACHE, "scoruri_" + r + ".csv")):
            ramuri.append(r)
    if len(ramuri) == 0:
        print("Nu există încă niciun scor. Rulează întâi antreneaza_svm.py și/sau antreneaza_cnn.py.")
        return
    print("Ramuri găsite:", ramuri)

    #fețele de validare
    val = []
    for fata in citeste_scorurile(ramuri):
        if fata["split"] == "val":
            val.append(fata)
    etichete = []
    for fata in val:
        etichete.append(fata["eticheta"])

    #pentru fiecare ramură: media și abaterea pe validare, plus AUC-ul ei
    calibrare = {"ramuri": ramuri, "z": {}, "ponderi": {}, "auc_val": {}}
    for r in ramuri:
        scoruri_r = []
        for fata in val:
            scoruri_r.append(fata["scor_" + r])
        calibrare["z"][r] = parametri_z(scoruri_r)
        calibrare["ponderi"][r] = 1.0
        calibrare["auc_val"][r] = float(roc_auc_score(etichete, scoruri_r))
        print("Ramura", r, ": AUC pe validare", round(calibrare["auc_val"][r], 4))

    #ponderile ramurilor: încerc 0; 0,1; ...; 1 pentru A (B primește restul)
    if args.ponderi == "auto" and len(ramuri) == 2:
        cea_mai_buna = 0.5
        cel_mai_bun_auc = -1
        for pas in range(11):
            pondere_a = pas / 10
            calibrare["ponderi"] = {"A": pondere_a, "B": 1 - pondere_a}
            auc = roc_auc_score(etichete, scoruri_fuzionate(val, calibrare))
            if auc > cel_mai_bun_auc + 1e-9:
                cea_mai_buna = pondere_a
                cel_mai_bun_auc = auc
        calibrare["ponderi"] = {"A": round(cea_mai_buna, 1), "B": round(1 - cea_mai_buna, 1)}
        print("Ponderi alese pe validare: A =", calibrare["ponderi"]["A"], ", B =", calibrare["ponderi"]["B"])

    #scorul fuzionat pe validare și AUC-ul lui
    scor = scoruri_fuzionate(val, calibrare)
    calibrare["auc_val"]["fuzionat"] = float(roc_auc_score(etichete, scor))
    print("Fuzionat: AUC pe validare", round(calibrare["auc_val"]["fuzionat"], 4))
    if len(ramuri) == 2:
        cea_mai_buna_ramura = max(calibrare["auc_val"]["A"], calibrare["auc_val"]["B"])
        if calibrare["auc_val"]["fuzionat"] < cea_mai_buna_ramura - 0.01:
            print("ATENȚIE: fuziunea e mai slabă decât cea mai bună ramură singură. "
                  "Încearcă: python calibreaza.py --ponderi auto")

    #pragul (Youden)
    prag, tpr, fpr = prag_youden(etichete, scor)
    calibrare["prag"] = prag
    print("Prag (cadre):", round(prag, 3), "-> prinde", round(tpr * 100), "% din falsuri, acuză",
          round(fpr * 100), "% din reale")

    #pentru video: scorul unui video e media cadrelor lui, deci îi aleg un prag separat
    calibrare["prag_video"] = prag
    etichete_video, scoruri_video = medii_pe_video(val, scor)
    if 0 in etichete_video and 1 in etichete_video:           # am video-uri și reale, și false
        prag_v, tpr, fpr = prag_youden(etichete_video, scoruri_video)
        calibrare["prag_video"] = prag_v
        print("Prag (video):", round(prag_v, 3), "-> prinde", round(tpr * 100), "% din falsuri, acuză",
              round(fpr * 100), "% din reale")

    os.makedirs(FOLDER_MODELE, exist_ok=True)
    salveaza_calibrare(calibrare)
    print("Am salvat modele/calibrare.json")


if __name__ == "__main__":
    main()
