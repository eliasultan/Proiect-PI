"""Blocurile 6–8 din schemă, ramura B: augmentări, rețeaua EfficientNet-B0 și Grad-CAM."""
import io
import copy
import random
import cv2
import torch
import timm
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score

from config import MARIME_CNN

# media și abaterea culorilor din ImageNet: rețeaua preantrenată se așteaptă la imagini normalizate așa
MEDIE = [0.485, 0.456, 0.406]
ABATERE = [0.229, 0.224, 0.225]


# ------------------------------------------------------------ blocul 6: augmentări
def jpeg_aleator(imagine):
    """În jumătate din cazuri recomprim imaginea JPEG cu o calitate la întâmplare (între 40 și 95),
    ca rețeaua să nu învețe compresia în loc de artefactele măștii."""
    if random.random() < 0.5:
        return imagine                             # în cealaltă jumătate o las neschimbată
    calitate = random.randint(40, 95)
    memorie = io.BytesIO()                         # un „fișier” ținut în memorie, nu pe disc
    imagine.save(memorie, format="JPEG", quality=calitate)
    memorie.seek(0)                                # mă întorc la începutul „fișierului”, ca să-l citesc
    return Image.open(memorie).convert("RGB")


def transformari(pentru_antrenare):
    """Pașii prin care trece o imagine înainte să intre în rețea."""
    pasi = []
    if pentru_antrenare:
        pasi.append(T.RandomResizedCrop(MARIME_CNN, scale=(0.85, 1.0), ratio=(0.95, 1.05)))   # decupaj puțin diferit
        pasi.append(T.RandomHorizontalFlip())                                                # oglindire
        pasi.append(T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.02)], p=0.5))              # lumină și culoare
        pasi.append(T.Lambda(jpeg_aleator))                                                  # compresie
        pasi.append(T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.2))             # blur
    else:
        pasi.append(T.Resize((MARIME_CNN, MARIME_CNN)))                                      # doar redimensionare
    pasi.append(T.ToTensor())
    pasi.append(T.Normalize(MEDIE, ABATERE))
    return T.Compose(pasi)


class SetFete(torch.utils.data.Dataset):
    """Fețele decupate (fișiere PNG) și etichetele lor: 0 = real, 1 = fals."""

    def __init__(self, cai, etichete, pentru_antrenare):
        self.cai = cai
        self.etichete = etichete
        self.transformare = transformari(pentru_antrenare)

    def __len__(self):
        return len(self.cai)

    def __getitem__(self, i):
        imagine = Image.open(self.cai[i]).convert("RGB")
        x = self.transformare(imagine)
        y = torch.tensor(float(self.etichete[i]))
        return x, y


def tensor_din_imagine(fata):
    """O față decupată din OpenCV (BGR) -> tensorul pe care îl primește rețeaua (1 x 3 x 224 x 224)."""
    rgb = cv2.cvtColor(fata, cv2.COLOR_BGR2RGB)
    imagine = Image.fromarray(rgb)
    transformare = transformari(False)
    x = transformare(imagine)                      # 3 x 224 x 224
    return x.unsqueeze(0)                          # 1 x 3 x 224 x 224: un lot cu o singură imagine


# ------------------------------------------------------------ blocul 7: EfficientNet-B0
def construieste_model(preantrenat=True):
    """EfficientNet-B0 cu o singură ieșire: un număr care e mare când imaginea pare falsă."""
    if preantrenat:
        try:
            return timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)
        except Exception:
            print("ATENȚIE: nu pot descărca greutățile preantrenate (fără internet?). Antrenez de la zero.")
    return timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)


def dispozitiv():
    """Placa video, dacă există; altfel procesorul."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def scoruri(model, incarcator, device):
    """Scorul B pentru fiecare imagine: ieșirea rețelei înainte de sigmoidă. Pozitiv = spre fals."""
    model.eval()
    rezultat = []
    with torch.no_grad():                          # fără gradienți: doar calculez, nu învăț
        for x, y in incarcator:
            iesire = model(x.to(device))           # forma: (câte imagini, 1)
            for valoare in iesire:
                rezultat.append(float(valoare[0]))
    return rezultat


def scor_o_fata(model, fata):
    """Scorul B pentru o singură față (îl folosește demo-ul)."""
    x = tensor_din_imagine(fata)
    with torch.no_grad():
        iesire = model(x.to(dispozitiv()))
    return float(iesire[0][0])


def antreneaza(model, incarcator_train, incarcator_val, y_val, epoci, lr, device, pondere_fals=1.0):
    """Antrenez rețeaua și păstrez epoca cu cel mai bun AUC pe validare."""
    model = model.to(device)
    optimizator = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    planificator = torch.optim.lr_scheduler.CosineAnnealingLR(optimizator, T_max=epoci)   # lr scade treptat
    pondere = torch.tensor(pondere_fals, device=device)
    functie_pierdere = torch.nn.BCEWithLogitsLoss(pos_weight=pondere)
    cel_mai_bun_auc = -1
    cea_mai_buna_stare = None

    for epoca in range(1, epoci + 1):
        model.train()
        pierdere_totala = 0.0
        numar = 0
        for x, y in incarcator_train:
            x = x.to(device)
            y = y.to(device)
            optimizator.zero_grad()                # șterg gradienții de la pasul anterior
            iesire = model(x).view(-1)             # forma (lot, 1) devine un șir de `lot` valori
            pierdere = functie_pierdere(iesire, y)
            pierdere.backward()                    # calculez gradienții
            optimizator.step()                     # modific puțin greutățile rețelei
            pierdere_totala = pierdere_totala + pierdere.item() * len(y)
            numar = numar + len(y)
        planificator.step()

        auc = roc_auc_score(y_val, scoruri(model, incarcator_val, device))
        print("   epoca", epoca, "/", epoci, "  pierdere =", round(pierdere_totala / numar, 4),
              "  AUC pe validare =", round(auc, 4))
        if auc > cel_mai_bun_auc or cea_mai_buna_stare is None:
            cel_mai_bun_auc = auc
            cea_mai_buna_stare = copy.deepcopy(model.state_dict())   # o copie a tuturor greutăților
    model.load_state_dict(cea_mai_buna_stare)      # la final pun înapoi greutățile celei mai bune epoci
    return model, cel_mai_bun_auc


def incarca_model(cale, device=None):
    if device is None:
        device = dispozitiv()
    salvat = torch.load(cale, map_location=device, weights_only=False)
    model = construieste_model(preantrenat=False)
    model.load_state_dict(salvat["stare"])
    model = model.to(device)
    model.eval()
    return model
