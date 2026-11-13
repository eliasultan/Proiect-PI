
import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import FOLDER_PROIECT, FOLDER_CACHE, FOLDER_MODELE, FISIER_CNN, SEED
from date import citeste_fetele, scrie_tabel
from ramura_b import SetFete, construieste_model, dispozitiv, antreneaza, scoruri


def fete_din(fete, split):
    rezultat = []
    for fata in fete:
        if fata["split"] == split:
            rezultat.append(fata)
    return rezultat


def fa_incarcator(fete, pentru_antrenare, lot, workers):
    cai = []
    etichete = []
    for fata in fete:
        cai.append(os.path.join(FOLDER_PROIECT, fata["fata"]))
        etichete.append(fata["eticheta"])
    set_date = SetFete(cai, etichete, pentru_antrenare)
    return DataLoader(set_date, batch_size=lot, shuffle=pentru_antrenare, num_workers=workers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoci", type=int, default=5)
    parser.add_argument("--lot", type=int, default=16, help="cate imagini intra deodata in retea")
    parser.add_argument("--lr", type=float, default=0.0003, help="rata de invatare")
    parser.add_argument("--workers", type=int, default=2, help="procese care incarca imaginile")
    args = parser.parse_args()
    random.seed(SEED)                              # aceleasi numere aleatoare la fiecare rulare
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    fete = citeste_fetele()
    train = fete_din(fete, "train")
    val = fete_din(fete, "val")
    test = fete_din(fete, "test")

    reale = 0
    false = 0
    for fata in train:
        if fata["eticheta"] == 0:
            reale = reale + 1
        else:
            false = false + 1
    device = dispozitiv()
    print("Dispozitiv:", device, "· fete de antrenare:", len(train), "(", reale, "reale,", false, "false ) · de validare:", len(val))

    # daca am mai putine falsuri decat reale, greselile pe falsuri conteaza mai mult (si invers)
    if false > 0:
        pondere_fals = reale / false
    else:
        pondere_fals = 1.0
    etichete_val = []
    for fata in val:
        etichete_val.append(fata["eticheta"])

    model = construieste_model(preantrenat=True)
    incarcator_train = fa_incarcator(train, True, args.lot, args.workers)
    incarcator_val = fa_incarcator(val, False, args.lot, args.workers)
    model, auc = antreneaza(model, incarcator_train, incarcator_val, etichete_val,
                            args.epoci, args.lr, device, pondere_fals)

    os.makedirs(FOLDER_MODELE, exist_ok=True)
    torch.save({"stare": model.state_dict(), "auc_val": auc}, FISIER_CNN)
    print("Cel mai bun AUC pe validare:", round(auc, 4), "· am salvat modelul in", FISIER_CNN)

    # scorul B pentru fetele din validare și test
    rezultat = []
    for parte in [val, test]:
        if len(parte) == 0:
            continue
        scoruri_parte = scoruri(model, fa_incarcator(parte, False, args.lot, args.workers), device)
        for i in range(len(parte)):
            fata = parte[i]
            rand = {"fata": fata["fata"], "split": fata["split"], "eticheta": fata["eticheta"], "grup": fata["grup"]}
            rand["scor_B"] = scoruri_parte[i]
            rezultat.append(rand)
    scrie_tabel(os.path.join(FOLDER_CACHE, "scoruri_B.csv"), rezultat, ["fata", "split", "eticheta", "grup", "scor_B"])
    print("Am salvat scorurile in cache/scoruri_B.csv")


if __name__ == "__main__":
    main()
