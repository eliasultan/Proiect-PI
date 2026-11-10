import os
import random
import csv

from config import FOLDER_DATE, EXTENSII_IMAGINI, SEED
from config import FOLDER_CACHE

# numele de foldere pe care le accept ("valid" e numele folosit de setul 140k de pe Kaggle)
SPLITURI = {"train": "train", "val": "val", "valid": "val", "test": "test"}
CLASE = {"real": 0, "fake": 1}


def este_imagine(nume_fisier):
    nume_fisier = nume_fisier.lower()
    for extensie in EXTENSII_IMAGINI:
        if nume_fisier.endswith(extensie):
            return True
    return False


def grup_imagine(nume_fisier):
    #Cadrele din același video (clip7_f0012.png, clip7_f0040.png)
    #primesc acelasi grup, „clip7”, ca sa ajunga toate în acelasi split
    nume = os.path.splitext(nume_fisier)[0]          # fara extensie: clip7_f0012
    pozitie = nume.rfind("_f")                        # unde apare ultima data _f
    if pozitie != -1:
        dupa = nume[pozitie + 2:]                     # ce vine dupa _f: 0012
        if dupa.isdigit():
            return nume[:pozitie]                     # clip7
    return nume                                       # o poza obisnuita e propriul ei grup


def listeaza_imaginile(folder=FOLDER_DATE):
    imagini = []
    if not os.path.isdir(folder):
        return imagini
    for nume_split in sorted(os.listdir(folder)):
        cale_split = os.path.join(folder, nume_split)
        if not os.path.isdir(cale_split):
            continue
        if nume_split.lower() not in SPLITURI:
            continue                                  # un folder care nu e train / val / test
        split = SPLITURI[nume_split.lower()]
        for nume_clasa in sorted(os.listdir(cale_split)):
            cale_clasa = os.path.join(cale_split, nume_clasa)
            clasa = nume_clasa.lower()
            if not os.path.isdir(cale_clasa):
                continue
            if clasa not in CLASE:
                continue                              # un folder care nu e real / fake
            for nume_fisier in sorted(os.listdir(cale_clasa)):
                if este_imagine(nume_fisier):
                    imagine = {}
                    imagine["cale"] = os.path.join(cale_clasa, nume_fisier)
                    imagine["split"] = split
                    imagine["clasa"] = clasa
                    imagine["eticheta"] = CLASE[clasa]
                    imagine["grup"] = grup_imagine(nume_fisier)
                    imagini.append(imagine)
    return imagini


def separa_validare(imagini, procent=0.15):
    for imagine in imagini:
        if imagine["split"] == "val":
            return imagini

    random.seed(SEED)
    for clasa in ["real", "fake"]:
        #grupurile din train ale clasei, fiecare o singura data
        grupuri = []
        deja_vazute = set()
        for imagine in imagini:
            if imagine["split"] == "train" and imagine["clasa"] == clasa:
                if imagine["grup"] not in deja_vazute:
                    deja_vazute.add(imagine["grup"])
                    grupuri.append(imagine["grup"])

        #le amestec si aleg 15% dintre ele, dar cel putin unul
        grupuri.sort()                                # pornesc mereu de la aceeași ordine
        random.shuffle(grupuri)
        cate = round(len(grupuri) * procent)
        if cate < 1:
            cate = 1
        alese = set(grupuri[:cate])

        #toate imaginile grupurilor alese trec în val
        for imagine in imagini:
            if imagine["split"] == "train" and imagine["clasa"] == clasa:
                if imagine["grup"] in alese:
                    imagine["split"] = "val"
    return imagini


def numara(imagini):
    numar = {}
    for imagine in imagini:
        cheie = imagine["split"] + " " + imagine["clasa"]
        if cheie not in numar:
            numar[cheie] = 0
        numar[cheie] = numar[cheie] + 1
    return numar


def scrie_tabel(cale, randuri, coloane):
    with open(cale, "w", newline="", encoding="utf-8") as fisier:
        scriitor = csv.DictWriter(fisier, fieldnames=coloane, extrasaction="ignore")
        scriitor.writeheader()
        for rand in randuri:
            scriitor.writerow(rand)


def citeste_tabel(cale):
    randuri = []
    with open(cale, newline="", encoding="utf-8") as fisier:
        cititor = csv.DictReader(fisier)
        for rand in cititor:
            randuri.append(rand)
    return randuri


def citeste_fetele():
    fete = []
    for rand in citeste_tabel(os.path.join(FOLDER_CACHE, "index.csv")):
        if rand["gasita"] == "1":
            rand["eticheta"] = int(rand["eticheta"])
            fete.append(rand)
    return fete


if __name__ == "__main__":
    imagini = listeaza_imaginile()
    if len(imagini) == 0:
        print("Nu am gasit nicio imagine in", FOLDER_DATE)
        print("Structura ceruta: date/train/real, date/train/fake, date/test/real, date/test/fake")
    else:
        imagini = separa_validare(imagini)
        numar = numara(imagini)
        print("Imagini gasite:")
        for cheie in sorted(numar):
            print("  ", cheie, ":", numar[cheie])
