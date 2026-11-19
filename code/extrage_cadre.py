"""Dacă setul de date are video-uri: scot câteva cadre din fiecare și le pun în date/, ca imagini."""
import os
import argparse

from config import FOLDER_DATE, FOLDER_VIDEO, CADRE_VIDEO, EXTENSII_VIDEO
from preprocesare import extrage_cadre, salveaza_imagine


def este_video(nume_fisier):
    """True dacă numele fișierului se termină cu o extensie de video."""
    nume_fisier = nume_fisier.lower()
    for extensie in EXTENSII_VIDEO:
        if nume_fisier.endswith(extensie):
            return True
    return False


def listeaza_video_urile():
    """Toate video-urile din date_video/<split>/<clasă>/, ca listă de dicționare."""
    video_uri = []
    if not os.path.isdir(FOLDER_VIDEO):
        return video_uri
    for split in sorted(os.listdir(FOLDER_VIDEO)):
        cale_split = os.path.join(FOLDER_VIDEO, split)
        if not os.path.isdir(cale_split):
            continue
        for clasa in sorted(os.listdir(cale_split)):
            cale_clasa = os.path.join(cale_split, clasa)
            if not os.path.isdir(cale_clasa):
                continue
            for nume_fisier in sorted(os.listdir(cale_clasa)):
                if este_video(nume_fisier):
                    video = {"cale": os.path.join(cale_clasa, nume_fisier), "split": split, "clasa": clasa}
                    video["nume"] = os.path.splitext(nume_fisier)[0]
                    video_uri.append(video)
    return video_uri


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cadre", type=int, default=CADRE_VIDEO, help="câte cadre din fiecare video")
    args = parser.parse_args()

    video_uri = listeaza_video_urile()
    if len(video_uri) == 0:
        print("Nu am găsit video-uri în", FOLDER_VIDEO)
        print("Structura cerută: date_video/train/real, date_video/train/fake, date_video/test/real, ...")
        return

    for i in range(len(video_uri)):
        video = video_uri[i]
        cadre = extrage_cadre(video["cale"], args.cadre)
        for pereche in cadre:
            numar_cadru = pereche[0]
            imagine = pereche[1]
            nume = video["nume"] + "_f" + str(numar_cadru).zfill(4) + ".png"   # de exemplu clip7_f0031.png
            salveaza_imagine(os.path.join(FOLDER_DATE, video["split"], video["clasa"], nume), imagine)
        print("  ", i + 1, "/", len(video_uri), " ", video["split"], "/", video["clasa"], "/", video["nume"], ":",
              len(cadre), "cadre")
    print("Gata. Cadrele sunt în folderul date/.")


if __name__ == "__main__":
    main()
