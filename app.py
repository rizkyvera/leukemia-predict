import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import albumentations as A
import os
import gdown
import zipfile
from albumentations.pytorch import ToTensorV2

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Leukemia Detection System",
    layout="wide"
)

# ── Custom CSS & Icons ─────────────────────────────────────────
st.markdown("""
<style>
    /* Styling to make the app more attractive */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #e0f2fe 0%, #d1fae5 100%);
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1rem;
        margin-top: 1rem;
        font-size: 1.5rem;
    }
    .section-title svg {
        width: 28px;
        height: 28px;
        stroke: #3b82f6;
    }
    .main-title {
        display: flex;
        align-items: center;
        gap: 12px;
        color: #1f2937;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .main-title svg {
        width: 48px;
        height: 48px;
        stroke: #3b82f6;
    }
    .alert-success {
        padding: 1rem; background-color: #d1fae5; border-radius: 8px; color: #065f46; display: flex; align-items: center; gap: 8px; margin-bottom: 1rem; border: 1px solid #a7f3d0;
    }
    .alert-error {
        padding: 1rem; background-color: #fee2e2; border-radius: 8px; color: #991b1b; display: flex; align-items: center; gap: 8px; margin-bottom: 1rem; border: 1px solid #fecaca;
    }
    .alert-info {
        padding: 1rem; background-color: #dbeafe; border-radius: 8px; color: #1e40af; display: flex; align-items: center; gap: 8px; margin-bottom: 1rem; border: 1px solid #bfdbfe;
    }
    .pred-malignant {
        display: flex; align-items: center; gap: 8px; color: #ef4444; font-weight: 700; font-size: 2rem; margin: 0;
    }
    .pred-benign {
        display: flex; align-items: center; gap: 8px; color: #10b981; font-weight: 700; font-size: 2rem; margin: 0;
    }
</style>
""", unsafe_allow_html=True)


def get_icon(name, size=24, color="currentColor"):
    icons = {
        "microscope": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 18h8"/><path d="M3 22h18"/><path d="M14 22a7 7 0 1 0 0-14h-1"/><path d="M9 14h2"/><path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"/><path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"/></svg>',
        "settings": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>',
        "chart": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>',
        "upload": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>',
        "image": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>',
        "mask": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12A10 10 0 1 0 22 12 10 10 0 1 0 2 12Z"/><path d="M12 2v20"/></svg>',
        "crop": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/></svg>',
        "target": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
        "check": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
        "alert": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/></svg>',
        "info": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="#1e40af" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="16" y2="12"/><line x1="12" x2="12.01" y1="8" y2="8"/></svg>',
        "pointer": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="#1e40af" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 14a8 8 0 0 1-8 8"/><path d="M18 11v-1a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0"/><path d="M14 10V9a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v1"/><path d="M10 9.5V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v10"/><path d="M18 11a2 2 0 1 1 4 0v3a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/></svg>',
        "warn": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/></svg>',
        "dot-red": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="#ef4444" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/></svg>',
        "dot-green": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="#10b981" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/></svg>'
    }
    return icons.get(name, "")


def render_section_title(text, icon_name):
    st.markdown(f'<div class="section-title">{get_icon(icon_name)} {text}</div>', unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────
DEVICE = torch.device('cpu')  # CPU untuk deployment lokal
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

MODEL_PATHS = {
    "unet":     "models/best_unet.pth",
    "cnn_unet": "models/best_vgg16.pth",
    "baseline": "models/best_vgg16_baseline.pth",
}

MODEL_URLS = {
    "unet":     "https://docs.google.com/uc?export=download&id=16sP1Hqp3Gi8YllqGEBM1VHynWONH3B5F",
    "cnn_unet": "https://docs.google.com/uc?export=download&id=1C2lPU2yfdrQtiIcEG2ovh591BfBBh0kU",
    "baseline": "https://docs.google.com/uc?export=download&id=1BO4YzQYzXDVB8cCErxp88OemC5_znBB9",
}

def download_model_if_not_exists(model_key):
    path = MODEL_PATHS[model_key]
    url = MODEL_URLS[model_key]
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading {model_key} model from cloud...")
        gdown.download(url, path, quiet=False)

# ── Transform ─────────────────────────────────────────────────
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
    ToTensorV2(),
])

# ── Model Definitions ──────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c,  out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pool     = nn.MaxPool2d(2)
        self.decoders = nn.ModuleList()
        self.upconvs  = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(ConvBlock(ch, f)); ch = f
        self.bottleneck = ConvBlock(features[-1], features[-1]*2)
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f*2, f, 2, stride=2))
            self.decoders.append(ConvBlock(f*2, f))
        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x); skips = skips[::-1]
        for i, (up, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = up(x)
            if x.shape != skips[i].shape:
                x = torch.nn.functional.interpolate(x, size=skips[i].shape[2:])
            x = torch.cat([skips[i], x], dim=1); x = dec(x)
        return torch.sigmoid(self.final(x))

def build_vgg16():
    vgg = models.vgg16(weights=None)
    for param in vgg.features[:24].parameters():
        param.requires_grad = False
    vgg.classifier = nn.Sequential(
        nn.Linear(25088, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
        nn.Linear(4096,  1024), nn.ReLU(inplace=True), nn.Dropout(0.5),
        nn.Linear(1024,  1),    nn.Sigmoid()
    )
    return vgg

# ── Prepare Datasets ───────────────────────────────────────────
@st.cache_resource
def prepare_datasets():
    datasets = [
        ("Test.zip", "Test", "https://docs.google.com/uc?export=download&id=1x-PplP-0eyNqS8X73XHnScs5vHjACiG8"),
        ("Ground Truth.zip", "Ground Truth", "https://docs.google.com/uc?export=download&id=1IYpNv-bf7jPPFdFP8h-tHk0Nk-91sG1d")
    ]
    for zip_name, folder_name, url in datasets:
        if not os.path.exists(folder_name):
            if not os.path.exists(zip_name):
                print(f"Downloading {zip_name} from cloud...")
                gdown.download(url, zip_name, quiet=False)
            
            if os.path.exists(zip_name):
                print(f"Extracting {zip_name}...")
                with zipfile.ZipFile(zip_name, 'r') as zip_ref:
                    zip_ref.extractall(".")

# ── Load Models (cached) ───────────────────────────────────────
@st.cache_resource
def load_models():
    for key in MODEL_PATHS.keys():
        download_model_if_not_exists(key)

    # U-Net
    unet = UNet().to(DEVICE)
    unet.load_state_dict(torch.load(MODEL_PATHS["unet"], map_location=DEVICE))
    unet.eval()

    # VGG16 + U-Net
    cnn_unet = build_vgg16().to(DEVICE)
    cnn_unet.load_state_dict(torch.load(MODEL_PATHS["cnn_unet"], map_location=DEVICE))
    cnn_unet.eval()

    # VGG16 Baseline
    baseline = build_vgg16().to(DEVICE)
    baseline.load_state_dict(torch.load(MODEL_PATHS["baseline"], map_location=DEVICE))
    baseline.eval()

    return unet, cnn_unet, baseline

# ── Helper functions ───────────────────────────────────────────
def denorm(tensor):
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * STD + MEAN, 0, 1)

def preprocess(img_array):
    aug = val_transform(image=img_array)
    return aug['image'].unsqueeze(0).to(DEVICE)

def predict_baseline(model, img_t):
    with torch.no_grad():
        prob = model(img_t).squeeze().item()
    label = "Malignant" if prob > 0.5 else "Benign"
    return prob, label


def predict_unet_only(unet, img_t):
    """Segmentasi U-Net saja tanpa klasifikasi"""
    with torch.no_grad():
        mask_pred = unet(img_t)[0][0].cpu().numpy()
        binary_mask = (mask_pred > 0.5).astype(np.float32)
        # Apply mask ke citra
        img_np = img_t[0].cpu()
        img_denorm = denorm(img_np)
        masked_img = img_denorm * binary_mask[:, :, np.newaxis]
        # Hitung coverage
        coverage = binary_mask.mean() * 100
    return binary_mask, masked_img, mask_pred, coverage

def predict_unet_vgg(unet, vgg, img_t, img_original):
    with torch.no_grad():
        # Segmentasi
        mask_pred = unet(img_t)[0][0].cpu().numpy()
        binary_mask = (mask_pred > 0.5).astype(np.float32)
        # Apply mask
        img_np = img_t[0].cpu()
        img_denorm = denorm(img_np)
        masked_img = img_denorm * binary_mask[:, :, np.newaxis]
        # Klasifikasi
        masked_t = torch.tensor(
            masked_img.transpose(2, 0, 1), dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)
        prob = vgg(masked_t).squeeze().item()
    label = "Malignant" if prob > 0.5 else "Benign"
    return prob, label, binary_mask, masked_img


def find_ground_truth_for_path(path):
    """Cari ground truth berdasarkan variasi nama/folder untuk path yang diberikan.
    Mengembalikan path file ground truth jika ditemukan, atau None.
    """
    if not path:
        return None
    gt_path = None
    base, ext = os.path.splitext(path)
    suffixes = ['_mask', '_gt', '-mask', ' mask', '_Mask', '_GT']
    exts = ['.png', '.jpg', '.jpeg']
    candidates = []
    # same folder patterns
    for s in suffixes:
        for e in exts:
            candidates.append(base + s + e)
    # sibling folders like masks/, gt/, ground_truth/
    for folder in ['masks', 'gt', 'ground_truth', 'mask']:
        for e in exts:
            try:
                candidates.append(os.path.join(os.path.dirname(path), folder, os.path.basename(base) + e))
            except Exception:
                pass
    # top-level sibling folder attempts (e.g., replace 'Test' with 'Ground Truth')
    try:
        parts = path.replace('\\', '/').split('/')
        if 'Test' in parts:
            idx = parts.index('Test')
            root_parts = parts[:idx]
            rel_parts = parts[idx+1:]
            basename_only = os.path.splitext(rel_parts[-1])[0]
            gt_dirs = ['Ground Truth', 'Ground_Truth', 'GroundTruth', 'GroundTruths', 'Groundtruth', 'Ground-Truth']
            for gd in gt_dirs:
                for e in exts:
                    cand = os.path.join('/'.join(root_parts), gd, *rel_parts[:-1], basename_only + e).replace('\\', '/')
                    candidates.append(cand)
    except Exception:
        pass

    for c in candidates:
        try:
            if os.path.normpath(c) == os.path.normpath(path):
                continue
        except Exception:
            pass
        if os.path.exists(c):
            gt_path = c
            break
    return gt_path

# ── UI ─────────────────────────────────────────────────────────
st.markdown(f'<div class="main-title">{get_icon("microscope")} Leukemia Detection System</div>', unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #4b5563; margin-bottom: 2rem;'><strong>Model Prediksi Leukemia Berbasis Segmentasi Otomatis Sel Darah Putih — U-Net + VGG16</strong></p>", unsafe_allow_html=True)

# Load models & datasets
with st.spinner("Menyiapkan dataset dan memuat model... Harap tunggu."):
    try:
        prepare_datasets()
        unet_model, cnn_unet_model, baseline_model = load_models()
        st.markdown(f'<div class="alert-success">{get_icon("check")} Dataset dan Model siap digunakan!</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="alert-error">{get_icon("alert")} Gagal memuat model: {e}</div>', unsafe_allow_html=True)
        st.stop()

st.divider()

# Sidebar
with st.sidebar:
    render_section_title("Pengaturan", "settings")
    model_choice = st.radio(
        "Pilih Model:",
        ["U-Net Segmentation Only", "VGG16 + U-Net (Proposed)", "VGG16 Baseline"],
        index=0
    )
    st.divider()
    render_section_title("Info Model", "chart")
    if model_choice == "U-Net Segmentation Only":
        st.markdown(f'<div class="alert-info">{get_icon("info")} Pipeline: Citra → U-Net → Mask Prediksi</div>', unsafe_allow_html=True)
        st.metric("DSC Test", "0.7885")
        st.metric("IoU Test", "0.7134")
    elif model_choice == "VGG16 + U-Net (Proposed)":
        st.markdown(f'<div class="alert-info">{get_icon("info")} Pipeline: Citra → U-Net → Mask → VGG16 → Prediksi</div>', unsafe_allow_html=True)
        st.metric("Akurasi Test", "98.62%")
        st.metric("AUC-ROC", "0.9958")
        st.metric("DSC Segmentasi", "0.7885")
    else:
        st.markdown(f'<div class="alert-info">{get_icon("info")} Pipeline: Citra → VGG16 → Prediksi</div>', unsafe_allow_html=True)
        st.metric("Akurasi Test", "99.69%")
        st.metric("AUC-ROC", "0.9999")

# Single Inference page
render_section_title("Upload Citra Apusan Darah", "upload")
uploaded = st.file_uploader(
    "Pilih file citra (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

# Pilihan file langsung dari folder Test/
TEST_DIR = "Test"
test_files = []
if os.path.exists(TEST_DIR):
    for root, dirs, files in os.walk(TEST_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                rel = os.path.join(root, f).replace('\\', '/')
                test_files.append(rel)

test_choice = None
if test_files:
    st.markdown("**Atau pilih file contoh dari folder `Test/`:**")
    selected = st.selectbox("Pilih file dari Test/", ["-- pilih --"] + test_files)
    if selected and selected != "-- pilih --":
        test_choice = selected

def get_true_label(filename):
    name = filename.lower()
    if "benign" in name:
        return "Benign"
    elif "malignant" in name:
        return "Malignant"
    return None

def render_verdict(pred_label, true_label):
    if true_label is None:
        st.caption("ℹ️ Label asli tidak diketahui dari nama file.")
        return
    if pred_label == true_label:
        st.markdown(f"""
        <div style='background:#d1fae5;border-radius:8px;padding:1rem;border:1px solid #6ee7b7;margin-top:0.5rem'>
            <span style='font-size:1.3rem;font-weight:700;color:#065f46'>✅ Prediksi BENAR</span><br>
            <span style='color:#065f46'>Label asli: <strong>{true_label}</strong></span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:#fee2e2;border-radius:8px;padding:1rem;border:1px solid #fca5a5;margin-top:0.5rem'>
            <span style='font-size:1.3rem;font-weight:700;color:#991b1b'>❌ Prediksi SALAH!</span><br>
            <span style='color:#991b1b'>Harusnya: <strong>{true_label}</strong> → Terprediksi: <strong>{pred_label}</strong></span>
        </div>""", unsafe_allow_html=True)

if uploaded or test_choice:
    if test_choice:
        filename = os.path.basename(test_choice)
        true_label = get_true_label(filename)
        pil_img = Image.open(test_choice).convert('RGB')
        img_rgb = np.array(pil_img)
    else:
        true_label = get_true_label(uploaded.name)
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_display = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_t = preprocess(img_display)

    st.divider()

    # Cari path ground truth untuk file yang dipilih dari Test/
    gt_path_single = None
    if test_choice:
        gt_path_single = find_ground_truth_for_path(test_choice)

    if model_choice == "U-Net Segmentation Only":
        with st.spinner("Melakukan segmentasi U-Net..."):
            mask, masked, mask_raw, coverage = predict_unet_only(unet_model, img_t)

        # Tampilkan: [Original] | [Ground Truth Mask] | [Prediction (Masked)]
        col1, col2, col3 = st.columns(3)
        with col1:
            render_section_title("Original", "image")
            st.image(img_display, width=360)
        with col2:
            render_section_title("Ground Truth Mask", "mask")
            if gt_path_single:
                gt = cv2.imread(gt_path_single, cv2.IMREAD_GRAYSCALE)
                gt_resized = cv2.resize(gt, (IMG_SIZE, IMG_SIZE))
                st.image(gt_resized, clamp=True, width=360)
            else:
                st.caption("Ground truth tidak ditemukan untuk file ini.")
        with col3:
            render_section_title("Prediction (Masked)", "crop")
            st.image(np.clip(masked, 0, 1), width=360)

        st.divider()
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("DSC (Test Set)", "0.7885")
        with col_s2:
            st.metric("IoU (Test Set)", "0.7134")
        with col_s3:
            st.metric("WBC Coverage (gambar ini)", f"{coverage:.2f}%")

        st.divider()
        st.markdown(f'<div style="display: flex; align-items: center; gap: 8px; color: #6b7280; font-size: 0.875rem;">{get_icon("warn", 16)} Segmentasi U-Net mengisolasi area White Blood Cell dari background.</div>', unsafe_allow_html=True)

    elif model_choice == "VGG16 + U-Net (Proposed)":
        with st.spinner("Melakukan segmentasi dan klasifikasi..."):
            prob, label, mask, masked = predict_unet_vgg(
                unet_model, cnn_unet_model, img_t, img_display
            )

        # Tampilkan: [Citra Asli] | [ground truth] | [Citra Tersegmentasi (Hasil Model)]
        col1, col2, col3 = st.columns(3)
        with col1:
            render_section_title("Citra Asli", "image")
            st.image(img_display, width=360)
        with col2:
            render_section_title("ground truth", "mask")
            if gt_path_single:
                gt = cv2.imread(gt_path_single, cv2.IMREAD_GRAYSCALE)
                gt_resized = cv2.resize(gt, (IMG_SIZE, IMG_SIZE))
                st.image(gt_resized, clamp=True, width=360)
            else:
                st.caption("Ground truth tidak ditemukan untuk file ini.")
        with col3:
            render_section_title("Citra Tersegmentasi", "crop")
            st.image(np.clip(masked, 0, 1), width=360)

        st.divider()
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            render_section_title("Hasil Prediksi", "target")
            if prob > 0.5:
                st.markdown(f'<div class="pred-malignant">{get_icon("dot-red", 32)} {label}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pred-benign">{get_icon("dot-green", 32)} {label}</div>', unsafe_allow_html=True)
            st.write("")
            st.metric("Probabilitas Malignant", f"{prob:.4f}")
            st.metric("Probabilitas Benign", f"{1-prob:.4f}")
            render_verdict(label, true_label)
        with col_res2:
            render_section_title("Confidence", "chart")
            st.progress(prob if prob > 0.5 else 1 - prob)
            conf_pct = max(prob, 1-prob) * 100
            st.markdown(f"<p style='font-size: 1.2rem; font-weight: 600; color: #374151;'>Confidence: {conf_pct:.2f}%</p>", unsafe_allow_html=True)
            st.caption(f"Model: VGG16 + U-Net (Proposed)")

        st.divider()
        st.divider()
        st.markdown(f'<div style="display: flex; align-items: center; gap: 8px; color: #6b7280; font-size: 0.875rem;">{get_icon("warn", 16)} Sistem ini hanya untuk keperluan penelitian dan tidak menggantikan diagnosis medis profesional.</div>', unsafe_allow_html=True)

    elif model_choice == "VGG16 Baseline":
        # Baseline: hanya klasifikasi VGG pada citra asli
        with st.spinner("Melakukan klasifikasi (VGG16 Baseline)..."):
            prob, label = predict_baseline(baseline_model, img_t)
        # Tampilkan: [Original] | [Hasil Klasifikasi]
        col1, col2 = st.columns(2)
        with col1:
            render_section_title("Original", "image")
            st.image(img_display, width=360)
        with col2:
            render_section_title("Hasil Klasifikasi (Baseline)", "target")
            if prob > 0.5:
                st.markdown(f'<div class="pred-malignant">{get_icon("dot-red", 32)} {label}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pred-benign">{get_icon("dot-green", 32)} {label}</div>', unsafe_allow_html=True)
            st.metric("Probabilitas Malignant", f"{prob:.4f}")
            st.metric("Probabilitas Benign", f"{1-prob:.4f}")
            render_verdict(label, true_label)

        st.divider()
        st.markdown(f'<div style="display: flex; align-items: center; gap: 8px; color: #6b7280; font-size: 0.875rem;">{get_icon("warn", 16)} VGG16 Baseline melakukan klasifikasi langsung pada citra tanpa segmentasi.</div>', unsafe_allow_html=True)

# Tampilkan Galeri Testing hanya jika model U-Net dipilih
if model_choice == "U-Net Segmentation Only":
    render_section_title("Galeri Testing", "chart")
    TEST_DIR = "Test"
    test_files = []
    if os.path.exists(TEST_DIR):
        for root, dirs, files in os.walk(TEST_DIR):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    test_files.append(os.path.join(root, f).replace('\\', '/'))

    if not test_files:
        st.markdown(f'<div class="alert-error">{get_icon("alert")} Folder `Test/` tidak ditemukan atau kosong.</div>', unsafe_allow_html=True)
    else:
        test_files = sorted(test_files)[:20]
        st.markdown(f"Menampilkan {len(test_files)} contoh dari folder `Test/` (ori | ground-truth | hasil U-Net)")
        for path in test_files:
            try:
                filename = os.path.basename(path)
                pil_img = Image.open(path).convert('RGB')
                img_rgb = np.array(pil_img)
                img_display = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
                img_t = preprocess(img_display)

                # Prediksi U-Net
                with st.spinner(f"Memproses {filename}..."):
                    mask, masked, mask_raw, coverage = predict_unet_only(unet_model, img_t)

                # Cari ground truth menggunakan helper
                gt_path = find_ground_truth_for_path(path)

                col1, col2, col3 = st.columns(3)
                with col1:
                    render_section_title(f"Original — {filename}", "image")
                    st.image(img_display, width=360)
                with col2:
                    render_section_title("ground truth", "mask")
                    if gt_path:
                        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                        gt_resized = cv2.resize(gt, (IMG_SIZE, IMG_SIZE))
                        st.image(gt_resized, clamp=True, width=360)
                    else:
                        st.caption("Ground truth tidak ditemukan")
                with col3:
                    render_section_title("Hasil U-Net", "mask")
                    mask_vis = (mask * 255).astype(np.uint8)
                    mask_color = cv2.applyColorMap(mask_vis, cv2.COLORMAP_VIRIDIS)
                    mask_color = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)
                    st.image(mask_color, width=360)
                st.divider()
            except Exception as e:
                st.markdown(f"<div class='alert-error'>Gagal memproses {path}: {e}</div>", unsafe_allow_html=True)
