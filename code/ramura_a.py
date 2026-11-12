from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score


def construieste_svm(C=1.0):
    """Trei pași legați unul după altul:
    1. SimpleImputer: o valoare lipsă (NaN) devine mediana coloanei, plus o coloană „aici lipsea”;
    2. StandardScaler: fiecare trăsătură e adusă la medie 0 și abatere 1;
    3. SVC: SVM-ul propriu-zis, cu kernel RBF (graniță curbă între real și fals)."""
    completare = SimpleImputer(strategy="median", add_indicator=True)
    scalare = StandardScaler()
    svm = SVC(C=C, kernel="rbf", gamma="scale", class_weight="balanced")
    return make_pipeline(completare, scalare, svm)

"""Scorul A pentru fiecare rând din X"""
def scor_svm(model, X):
    return model.decision_function(X)

"""Antrenare SVM pentru fiecare valoare a lui C și îl păstrez pe cel cu cel mai bun AUC pe validare."""
def antreneaza_svm(X_train, y_train, X_val, y_val, valori_C=[0.1, 1.0, 10.0], afiseaza=True):
    cel_mai_bun_auc = -1
    for C in valori_C:
        model = construieste_svm(C)
        model.fit(X_train, y_train)                # antrenarea
        auc = roc_auc_score(y_val, scor_svm(model, X_val))
        if afiseaza:
            print("   C =", C, "   AUC pe validare =", round(auc, 4))
        if auc > cel_mai_bun_auc:
            cel_mai_bun_C = C
            cel_mai_bun_auc = auc
    model = construieste_svm(cel_mai_bun_C)        # antrenez din nou, cu cel mai bun C
    model.fit(X_train, y_train)
    return model, cel_mai_bun_C, cel_mai_bun_auc

"""Pentru un split: X = lista vectorilor de trăsături, y = lista etichetelor (0 sau 1)."""
def date_pentru(fete, trasaturi, split):   
    X = []
    y = []
    for fata in fete:
        if fata["split"] == split:
            X.append(trasaturi[fata["fata"]])
            y.append(fata["eticheta"])
    return X, y
