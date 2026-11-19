"""Studiu de ablație pentru ramura A: cât contează fiecare grup de trăsături?
Pentru fiecare grup antrenez două SVM-uri: unul doar cu grupul respectiv și unul cu toate trăsăturile
în afară de el. Dacă AUC-ul scade mult când scot un grup, grupul acela contează.
"""
import os
from sklearn.metrics import roc_auc_score

from config import FOLDER_RAPOARTE
from date import citeste_fetele
from trasaturi import NUME_TRASATURI
from ramura_a import antreneaza_svm, scor_svm, date_pentru
from antreneaza_svm import citeste_trasaturile_salvate

# grupurile de trăsături și cum încep numele lor
GRUPURI = {"graniță": "gr", "frecvență": "fq_", "dinți": "di_", "ochi": "oc_", "piele": "pi_"}


def pastreaza_coloanele(X, pozitii):
    """Din fiecare vector păstrez doar valorile de pe pozițiile date."""
    X_nou = []
    for vector in X:
        vector_nou = []
        for p in pozitii:
            vector_nou.append(vector[p])
        X_nou.append(vector_nou)
    return X_nou


def auc_pe_test(fete, trasaturi, pozitii):
    """Antrenez un SVM doar pe trăsăturile de pe pozițiile date și întorc AUC-ul lui pe test."""
    X_train, y_train = date_pentru(fete, trasaturi, "train")
    X_val, y_val = date_pentru(fete, trasaturi, "val")
    X_test, y_test = date_pentru(fete, trasaturi, "test")
    X_train = pastreaza_coloanele(X_train, pozitii)
    X_val = pastreaza_coloanele(X_val, pozitii)
    X_test = pastreaza_coloanele(X_test, pozitii)
    model, C, auc_val = antreneaza_svm(X_train, y_train, X_val, y_val, afiseaza=False)
    return roc_auc_score(y_test, scor_svm(model, X_test))


def main():
    fete = citeste_fetele()
    trasaturi = citeste_trasaturile_salvate(fete)
    if trasaturi is None:
        print("Nu găsesc trăsăturile actuale. Rulează întâi: python antreneaza_svm.py")
        return

    toate = []
    for p in range(len(NUME_TRASATURI)):
        toate.append(p)
    auc_toate = auc_pe_test(fete, trasaturi, toate)
    print("Toate cele", len(NUME_TRASATURI), "de trăsături: AUC pe test =", round(auc_toate, 4))
    raport = []
    raport.append("# Ablație pentru ramura A (AUC pe test)\n")
    raport.append("Cu toate cele " + str(len(NUME_TRASATURI)) + " de trăsături: **" + str(round(auc_toate, 4)) + "**\n")
    raport.append("| Grup | Câte trăsături | Doar grupul | Toate fără grup | Diferență |")
    raport.append("|---|---|---|---|---|")

    for nume_grup in GRUPURI:
        inceput = GRUPURI[nume_grup]
        din_grup = []                              # pozițiile trăsăturilor din grup
        fara_grup = []                             # pozițiile celorlalte
        for p in range(len(NUME_TRASATURI)):
            if NUME_TRASATURI[p].startswith(inceput):
                din_grup.append(p)
            else:
                fara_grup.append(p)
        doar = auc_pe_test(fete, trasaturi, din_grup)
        fara = auc_pe_test(fete, trasaturi, fara_grup)
        diferenta = fara - auc_toate
        print("  ", nume_grup, "(", len(din_grup), "): doar grupul", round(doar, 4), "· fără grup", round(fara, 4))
        raport.append("| " + nume_grup + " | " + str(len(din_grup)) + " | " + str(round(doar, 4)) + " | "
                      + str(round(fara, 4)) + " | " + str(round(diferenta, 4)) + " |")

    raport.append("\nDiferența = AUC fără grup − AUC cu toate. Cu cât e mai negativă, cu atât grupul contează mai mult.")
    os.makedirs(FOLDER_RAPOARTE, exist_ok=True)
    with open(os.path.join(FOLDER_RAPOARTE, "ablatie.md"), "w", encoding="utf-8") as fisier:
        for rand in raport:
            fisier.write(rand + "\n")
    print("Am salvat rapoarte/ablatie.md")


if __name__ == "__main__":
    main()
