import os
import joblib

from config import FISIER_SVM, FISIER_CNN, FISIER_CALIBRARE
from preprocesare import creeaza_detector, pregateste_fata
from trasaturi import vector_trasaturi
from ramura_a import scor_svm
from ramura_b import incarca_model, scor_o_fata
from fuziune import fuzioneaza, incarca_calibrare
from config import CADRE_VIDEO
from preprocesare import extrage_cadre
from ramura_b import harta_gradcam, tensor_din_imagine

class Detector:
    def __init__(self):
        if not os.path.isfile(FISIER_CALIBRARE):
            raise FileNotFoundError("Lipsește modele/calibrare.json. Rulează întâi: python calibreaza.py")
        self.calibrare = incarca_calibrare()
        self.detector_fata = creeaza_detector()
        # încarc doar ramurile care au fost calibrate
        self.svm = None
        self.cnn = None
        if "A" in self.calibrare["ramuri"]:
            salvat = joblib.load(FISIER_SVM)
            self.svm = salvat["model"]
        if "B" in self.calibrare["ramuri"]:
            self.cnn = incarca_model(FISIER_CNN)

    def scoruri(self, fata, puncte, cu_harta=False):
        """Scorul A, scorul B, scorul fuzionat și harta Grad-CAM, într-un dicționar."""
        rez = {"scor_a": None, "scor_b": None, "harta": None}
        if self.svm is not None:
            vector = vector_trasaturi(fata, puncte)
            scoruri_svm = scor_svm(self.svm, [vector])    # SVM-ul primește o listă de vectori; eu am unul
            rez["scor_a"] = float(scoruri_svm[0])
        if self.cnn is not None:
            if cu_harta:
                x = tensor_din_imagine(fata)
                harta, scor_b = harta_gradcam(self.cnn, x)
                rez["harta"] = harta
                rez["scor_b"] = scor_b
            else:
                rez["scor_b"] = scor_o_fata(self.cnn, fata)
        rez["scor"] = fuzioneaza(rez["scor_a"], rez["scor_b"], self.calibrare)
        return rez

    def analizeaza_imagine(self, imagine, cu_harta=True):
        """imagine e o imagine OpenCV (BGR). Întoarce un dicționar cu tot ce afișează interfața."""
        fata, puncte = pregateste_fata(self.detector_fata, imagine)
        if fata is None:
            return {"ok": False}
        rez = self.scoruri(fata, puncte, cu_harta)
        rez["ok"] = True
        rez["fata"] = fata
        rez["puncte"] = puncte
        rez["prag"] = self.calibrare["prag"]
        if rez["scor"] >= rez["prag"]:
            rez["eticheta"] = "FALS"
        else:
            rez["eticheta"] = "REAL"
        return rez

    def analizeaza_video(self, cale_video, numar_cadre=CADRE_VIDEO):
        """Iau câteva cadre din video, le analizez pe fiecare și fac media scorurilor."""
        cadre = []
        for pereche in extrage_cadre(cale_video, numar_cadre):
            numar_cadru = pereche[0]
            imagine = pereche[1]
            fata, puncte = pregateste_fata(self.detector_fata, imagine)
            if fata is None:
                continue                           # cadru fără față: îl sar
            rez = self.scoruri(fata, puncte)
            cadre.append({"cadru": numar_cadru, "fata": fata, "scor": rez["scor"]})
        if len(cadre) == 0:
            return {"ok": False}

        suma = 0.0
        for cadru in cadre:
            suma = suma + cadru["scor"]
        scor = suma / len(cadre)                   # scorul video-ului = media cadrelor
        prag = self.calibrare["prag_video"]
        if scor >= prag:
            eticheta = "FALS"
        else:
            eticheta = "REAL"
        return {"ok": True, "cadre": cadre, "scor": scor, "prag": prag, "eticheta": eticheta}

    def inchide(self):
        self.detector_fata.close()
