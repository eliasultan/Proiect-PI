"""104 trăsături:  graniță 8 · frecvență 64 · dinți 13 · ochi 3 · piele 16"""
import numpy as np
import cv2
import math
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

from preprocesare import CONTUR, IRIS_STANG, IRIS_DREPT, BUZE_INTERIOR
from preprocesare import centrul, distanta, stanga_dreapta, puncte_intregi, raza_medie

#frecvența (64 de valori)
"""Spectrul Fourier al feței, mediat pe 64 de inele în jurul centrului: o valoare pentru fiecare frecvență."""
def trasaturi_frecventa(fata, numar_valori=64):
    gri = cv2.cvtColor(fata, cv2.COLOR_BGR2GRAY)
    gri = gri.astype(np.float32)                   
    inaltime = gri.shape[0]
    latime = gri.shape[1]

    #scad media și estompez marginile (fereastra Hann), altfel apar linii false în spectru
    gri = gri - gri.mean()
    fereastra = np.outer(np.hanning(inaltime), np.hanning(latime))
    gri = gri * fereastra

    #transformata Fourier 2D, cu frecvențele joase mutate în centru
    spectru = np.fft.fft2(gri)
    spectru = np.fft.fftshift(spectru)
    amplitudine = np.log(1 + np.abs(spectru))

    #pentru fiecare pixel: distanța până la centru (0 = centrul, 1 = marginea cercului)
    randuri, coloane = np.indices((inaltime, latime))
    dist = np.sqrt((coloane - latime / 2) ** 2 + (randuri - inaltime / 2) ** 2)
    dist = dist / (min(inaltime, latime) / 2)
    inel = (dist * numar_valori).astype(int)       # numărul inelului în care cade fiecare pixel

    #media amplitudinii pe fiecare inel
    rezultat = {}
    for i in range(numar_valori):
        valori = amplitudine[inel == i]            # pixelii de pe inelul i
        rezultat["fq_" + str(i).zfill(2)] = float(valori.mean())    
    return rezultat


def media(valori):
    if len(valori) == 0:
        return float("nan")
    return float(valori.mean())

"""Varianța (cât de împrăștiate sunt valorile)."""
def varianta(valori):
    if len(valori) == 0:
        return float("nan")
    return float(valori.var())

"""Abaterea standard (rădăcina varianței)"""
def abatere(valori):
    if len(valori) == 0:
        return float("nan")
    return float(valori.std())

"""Un dicționar cu toate numele date și valoarea NaN"""
def lipsa(nume_trasaturi):
    rezultat = {}
    for nume in nume_trasaturi:
        rezultat[nume] = float("nan")
    return rezultat


def masca_fata(fata, puncte):
    masca = np.zeros((fata.shape[0], fata.shape[1]), np.uint8)
    contur = puncte_intregi(puncte, CONTUR)
    cv2.fillConvexPoly(masca, cv2.convexHull(contur), 255)
    return masca

"""Zgomotul imaginii: diferența dintre imagine și varianta ei netezită."""
def zgomot(gri):
    netezita = cv2.medianBlur(gri, 3)
    return gri.astype(np.float32) - netezita.astype(np.float32)


#granița măștii (8 valori)
"""Compar o bandă chiar în interiorul conturului cu una chiar în exterior.
    La o față lipită, cele două benzi diferă la culoare, claritate și zgomot."""
def trasaturi_granita(fata, puncte):
    masca = masca_fata(fata, puncte)
    lab = cv2.cvtColor(fata, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]                               # luminozitatea
    A = lab[:, :, 1]                               # culoarea, axa verde – roșu
    B = lab[:, :, 2]                               # culoarea, axa albastru – galben
    gri = cv2.cvtColor(fata, cv2.COLOR_BGR2GRAY)
    laplace = cv2.Laplacian(gri.astype(np.float32), cv2.CV_32F)   # cât de clare sunt detaliile
    zg = zgomot(gri)

    rezultat = {}
    for latime in [7, 15]:
        nucleu = np.ones((latime, latime), np.uint8)
        erodata = cv2.erode(masca, nucleu)         # masca „roasă” pe margini
        dilatata = cv2.dilate(masca, nucleu)       # masca „umflată” pe margini
        interior = np.logical_and(masca > 0, erodata == 0)       # în față, dar lângă contur
        exterior = np.logical_and(dilatata > 0, masca == 0)      # în afara feței, dar lângă contur

        nume = "gr" + str(latime) + "_"
        rezultat[nume + "dL"] = abs(media(L[interior]) - media(L[exterior]))
        da = media(A[interior]) - media(A[exterior])
        db = media(B[interior]) - media(B[exterior])
        rezultat[nume + "dab"] = math.sqrt(da * da + db * db)
        clar_in = varianta(laplace[interior]) + 1e-6
        clar_out = varianta(laplace[exterior]) + 1e-6
        rezultat[nume + "clar"] = math.log(clar_in / clar_out)
        zg_in = abatere(zg[interior]) + 1e-6
        zg_out = abatere(zg[exterior]) + 1e-6
        rezultat[nume + "zg"] = math.log(zg_in / zg_out)
    return rezultat

#dinții (13 valori)
NUME_DINTI = ["di_frac", "di_muchii", "di_entropie"]
for k in range(10):
    NUME_DINTI.append("di_lbp" + str(k))

"""Trasaturi dinti"""
def trasaturi_dinti(fata, puncte):
    #e gura deschisă?
    deschidere = distanta(puncte[13], puncte[14]) / (distanta(puncte[78], puncte[308]) + 1e-6)
    if deschidere < 0.12:
        return lipsa(NUME_DINTI)                   # gura închisă: nu am ce măsura

    #masca interiorului gurii
    gura = np.zeros((fata.shape[0], fata.shape[1]), np.uint8)
    cv2.fillPoly(gura, [puncte_intregi(puncte, BUZE_INTERIOR)], 255)
    interior = gura > 0
    if interior.sum() < 50:
        return lipsa(NUME_DINTI)

    #dinții = pixelii din gură luminoși (peste pragul Otsu) și aproape fără culoare
    hsv = cv2.cvtColor(fata, cv2.COLOR_BGR2HSV)
    saturatie = hsv[:, :, 1]
    luminozitate = hsv[:, :, 2]
    valori_gura = luminozitate[interior].reshape(-1, 1)          # Otsu vrea o coloană de valori
    rezultat_otsu = cv2.threshold(valori_gura, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    prag = rezultat_otsu[0]                        # threshold întoarce (prag, imagine)
    dinti = np.logical_and(interior, luminozitate > prag)
    dinti = np.logical_and(dinti, saturatie < 90)
    numar_dinti = dinti.sum()
    if numar_dinti < 30:
        return lipsa(NUME_DINTI)

    #măsurătorile
    gri = cv2.cvtColor(fata, cv2.COLOR_BGR2GRAY)
    muchii = np.abs(cv2.Sobel(gri.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))   # muchii verticale
    rezultat = {}
    rezultat["di_frac"] = numar_dinti / interior.sum()            # ce parte din gură e dinte
    rezultat["di_muchii"] = media(muchii[dinti]) / 255

    rezultat_histograma = np.histogram(gri[dinti], bins=32, range=(0, 256))
    histograma = rezultat_histograma[0]            # câți pixeli cad în fiecare din cele 32 de intervale
    entropie = 0.0
    for numar in histograma:
        if numar > 0:
            p = numar / numar_dinti
            entropie = entropie - p * math.log2(p)
    rezultat["di_entropie"] = entropie

    lbp = local_binary_pattern(gri, P=8, R=1, method="uniform")     # un cod de textură (0–9) pentru fiecare pixel
    coduri = lbp[dinti]
    for k in range(10):
        rezultat["di_lbp" + str(k)] = (coduri == k).sum() / numar_dinti
    return rezultat


#ochii (3 valori)
NUME_OCHI = ["oc_iou", "oc_dist", "oc_arie"]

"""Masca reflexiei de lumină din iris"""
def reflexie(gri, puncte, iris, marime=32):
    centru = centrul(puncte, iris)
    raza = raza_medie(puncte, iris, centru)
    x0 = int(round(centru[0] - raza))
    y0 = int(round(centru[1] - raza))
    latura = int(round(2 * raza))
    if latura < 4:
        latura = 4
    if x0 < 0 or y0 < 0:
        return None                                # pătratul iese din imagine
    bucata = gri[y0:y0 + latura, x0:x0 + latura]
    if bucata.shape[0] != latura or bucata.shape[1] != latura:
        return None                                # pătratul a fost tăiat de marginea imaginii
    bucata = cv2.resize(bucata, (marime, marime)).astype(np.float32)

    # păstrez doar discul din mijloc (irisul), fără colțurile pătratului
    randuri, coloane = np.indices((marime, marime))
    dist = np.sqrt((coloane - marime / 2 + 0.5) ** 2 + (randuri - marime / 2 + 0.5) ** 2)
    disc = dist < 0.45 * marime
    valori = bucata[disc]
    cel_mai_luminos = valori.max()
    if cel_mai_luminos < 1.3 * np.median(valori) or cel_mai_luminos < 120:
        return None                                # niciun punct nu iese în evidență: nu se vede reflexia
    prag = max(np.percentile(valori, 95), 0.85 * cel_mai_luminos)
    return np.logical_and(disc, bucata >= prag)

"""Centrul pixelilor albi dintr-o mască: (x, y)."""
def centrul_mastii(masca):
    randuri, coloane = np.nonzero(masca)           # pozițiile pixelilor True
    return coloane.mean(), randuri.mean()

"""analiza refelexie ochi"""
def trasaturi_ochi(fata, puncte):
    gri = cv2.cvtColor(fata, cv2.COLOR_BGR2GRAY)
    r1 = reflexie(gri, puncte, IRIS_DREPT)
    r2 = reflexie(gri, puncte, IRIS_STANG)
    if r1 is None or r2 is None:
        return lipsa(NUME_OCHI)
    comun = np.logical_and(r1, r2).sum()           # pixeli de reflexie în ambii ochi
    total = np.logical_or(r1, r2).sum()            # pixeli de reflexie în cel puțin un ochi
    arie1 = r1.sum()
    arie2 = r2.sum()
    rezultat = {}
    rezultat["oc_iou"] = comun / total
    rezultat["oc_dist"] = distanta(centrul_mastii(r1), centrul_mastii(r2)) / r1.shape[0]
    rezultat["oc_arie"] = min(arie1, arie2) / max(arie1, arie2)
    return rezultat


#pielea și zgomotul (16 valori)
PROPRIETATI_GLCM = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
NUME_PIELE = []
for k in range(10):
    NUME_PIELE.append("pi_lbp" + str(k))
for proprietate in PROPRIETATI_GLCM:
    NUME_PIELE.append("pi_" + proprietate)
NUME_PIELE.append("pi_zgomot")

"""Trei pătrate mici de piele: obrazul stâng, obrazul drept și fruntea."""
def bucati_piele(gri, puncte):
    ochi_1 = centrul(puncte, IRIS_DREPT)
    ochi_2 = centrul(puncte, IRIS_STANG)
    ochi_st, ochi_dr = stanga_dreapta(ochi_1, ochi_2)
    gura_st, gura_dr = stanga_dreapta(puncte[78], puncte[308])     # colțurile gurii
    d = distanta(ochi_st, ochi_dr)                 # distanța dintre ochi: unitatea mea de măsură

    centre = []
    centre.append((ochi_st[0] - 0.05 * d, (ochi_st[1] + gura_st[1]) / 2))      # obrazul stâng
    centre.append((ochi_dr[0] + 0.05 * d, (ochi_dr[1] + gura_dr[1]) / 2))      # obrazul drept
    centre.append(((ochi_st[0] + ochi_dr[0]) / 2, (ochi_st[1] + ochi_dr[1]) / 2 - 0.55 * d))   # fruntea

    jumatate = int(0.15 * d)                       # jumătate din latura pătratului
    if jumatate < 4:
        jumatate = 4
    bucati = []
    for centru in centre:
        x0 = int(centru[0]) - jumatate
        y0 = int(centru[1]) - jumatate
        if x0 < 0 or y0 < 0:
            continue
        bucata = gri[y0:y0 + 2 * jumatate, x0:x0 + 2 * jumatate]
        if bucata.shape[0] == 2 * jumatate and bucata.shape[1] == 2 * jumatate:
            bucati.append(bucata)
    return bucati

"""Textura pielii (LBP și GLCM) și zgomotul feței comparat cu al fundalului."""
def trasaturi_piele(fata, puncte):
    gri = cv2.cvtColor(fata, cv2.COLOR_BGR2GRAY)
    bucati = bucati_piele(gri, puncte)
    rezultat = lipsa(NUME_PIELE)                   # pornesc cu toate lipsă și completez ce pot măsura

    if len(bucati) > 0:
        # LBP: pentru fiecare cod 0–9, ce parte din pixelii unei bucăți au codul acela (media pe bucăți)
        coduri_lbp = []
        for bucata in bucati:
            coduri_lbp.append(local_binary_pattern(bucata, 8, 1, "uniform"))
        for k in range(10):
            suma = 0.0
            for lbp in coduri_lbp:
                suma = suma + (lbp == k).sum() / lbp.size
            rezultat["pi_lbp" + str(k)] = suma / len(coduri_lbp)

        # GLCM: cât de des apar două niveluri de gri unul lângă altul
        matrici = []
        for bucata in bucati:
            matrici.append(graycomatrix(bucata, distances=[1], angles=[0, np.pi / 2],
                                        levels=256, symmetric=True, normed=True))
        for proprietate in PROPRIETATI_GLCM:
            suma = 0.0
            numar = 0
            for glcm in matrici:
                valoare = graycoprops(glcm, proprietate).mean()
                if not math.isnan(valoare):        # pe o bucată complet uniformă, corelația nu există
                    suma = suma + valoare
                    numar = numar + 1
            if numar > 0:
                rezultat["pi_" + proprietate] = suma / numar

    # zgomotul din interiorul feței, comparat cu cel din fundal
    masca = masca_fata(fata, puncte)
    nucleu = np.ones((15, 15), np.uint8)
    pe_fata = cv2.erode(masca, nucleu) > 0         # interiorul feței, departe de contur
    pe_fundal = cv2.dilate(masca, nucleu) == 0     # fundalul, departe de față
    zg = zgomot(gri)
    zg_fata = abatere(zg[pe_fata]) + 1e-6
    zg_fundal = abatere(zg[pe_fundal]) + 1e-6
    rezultat["pi_zgomot"] = math.log(zg_fata / zg_fundal)
    return rezultat


#vectorul final
NUME_TRASATURI = []
for latime in [7, 15]:
    for tip in ["dL", "dab", "clar", "zg"]:
        NUME_TRASATURI.append("gr" + str(latime) + "_" + tip)
for i in range(64):
    NUME_TRASATURI.append("fq_" + str(i).zfill(2))
for nume in NUME_DINTI:
    NUME_TRASATURI.append(nume)
for nume in NUME_OCHI:
    NUME_TRASATURI.append(nume)
for nume in NUME_PIELE:
    NUME_TRASATURI.append(nume)

"""Toate trăsăturile unei fețe, ca listă de numere"""
def vector_trasaturi(fata, puncte):
    valori = {}
    valori.update(trasaturi_frecventa(fata))
    valori.update(trasaturi_granita(fata, puncte))
    valori.update(trasaturi_dinti(fata, puncte))
    valori.update(trasaturi_ochi(fata, puncte))
    valori.update(trasaturi_piele(fata, puncte))
    vector = []
    for nume in NUME_TRASATURI:
        vector.append(valori[nume])
    return vector
