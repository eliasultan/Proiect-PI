import os
import argparse
import numpy as np

from config import FOLDER_PROIECT, FOLDER_CACHE
from date import listeaza_imaginile, separa_validare, numara, scrie_tabel
from preprocesare import creeaza_detector, citeste_imagine, salveaza_imagine, pregateste_fata

COLOANE_INDEX = ["cale", "split", "clasa", "eticheta", "grup", "fata", "puncte", "gasita"]


def pastreaza_primele(imagini, cate):
    pastrate = []
    numar = {}
    for imagine in imagini:
        cheie = imagine["split"] + " " + imagine["clasa"]
        if cheie not in numar:
            numar[cheie] = 0
        if numar[cheie] < cate:
            pastrate.append(imagine)
            numar[cheie] = numar[cheie] + 1
    return pastrate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=0, help="cate imagini pe split si clasa (0 = toate)")
    args = parser.parse_args()

    imagini = listeaza_imaginile()
    if len(imagini) == 0:
        print("Nu am gasit imagini în folderul date/. Vezi structura ceruta în README.")
        return
    imagini = separa_validare(imagini)
    if args.max > 0:
        imagini = pastreaza_primele(imagini, args.max)
    print("Imagini de pregatit:", len(imagini))

    detector = creeaza_detector()
    for i in range(len(imagini)):
        imagine = imagini[i]
        nume_fisier = os.path.basename(imagine["cale"])                       # de exemplu "r001.jpg"
        nume = os.path.splitext(nume_fisier)[0]                               # fara extensie: "r001"
        folder_fata = os.path.join(FOLDER_CACHE, "fete", imagine["split"], imagine["clasa"])
        cale_fata = os.path.join(folder_fata, nume + ".png")
        cale_puncte = os.path.join(folder_fata, nume + ".npy")

        if not os.path.isfile(cale_fata):          # daca am rulat deja o data, nu o mai fac
            img = citeste_imagine(imagine["cale"])
            if img is not None:
                fata, puncte = pregateste_fata(detector, img)
                if fata is not None:
                    salveaza_imagine(cale_fata, fata)
                    np.save(cale_puncte, puncte)

        if os.path.isfile(cale_fata):
            imagine["fata"] = os.path.relpath(cale_fata, FOLDER_PROIECT)       # cale relativa
            imagine["puncte"] = os.path.relpath(cale_puncte, FOLDER_PROIECT)
            imagine["gasita"] = 1
        else:
            imagine["fata"] = ""                   # nicio fata gasita în imagine
            imagine["puncte"] = ""
            imagine["gasita"] = 0
        imagine["cale"] = os.path.relpath(imagine["cale"], FOLDER_PROIECT)

        if (i + 1) % 50 == 0 or i + 1 == len(imagini):
            print("  ", i + 1, "/", len(imagini), "imagini")
    detector.close()

    os.makedirs(FOLDER_CACHE, exist_ok=True)
    scrie_tabel(os.path.join(FOLDER_CACHE, "index.csv"), imagini, COLOANE_INDEX)

    # cat am gasit din fiecare split si clasa
    cu_fata = []
    for imagine in imagini:
        if imagine["gasita"] == 1:
            cu_fata.append(imagine)
    total = numara(imagini)
    gasite = numara(cu_fata)
    print()
    print("Fete gasite (gasite / total):")
    for cheie in sorted(total):
        print("  ", cheie, ":", gasite.get(cheie, 0), "/", total[cheie])
    print("Am salvat cache/index.csv")


if __name__ == "__main__":
    main()
