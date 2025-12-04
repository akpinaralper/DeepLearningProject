import os
import torch
import librosa
import numpy as np
from model import AudioCNN

DATASET_DIR = "dataset"

model = AudioCNN()
model.load_state_dict(torch.load("audio_model.pth", map_location="cpu"))
model.eval()

def predict_file(path):
    x, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
    mfcc = librosa.util.fix_length(mfcc, size=20, axis=1)
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out = model(mfcc)
        probs = torch.softmax(out, dim=1)[0].numpy()
        pred = int(probs.argmax())
    return pred, probs

for cls in ["cat", "dog"]:
    cls_dir = os.path.join(DATASET_DIR, cls)
    print(f"\n== {cls.upper()} dosyaları ==")
    for f in os.listdir(cls_dir):
        if not f.endswith(".wav"):
            continue
        path = os.path.join(cls_dir, f)
        pred, probs = predict_file(path)
        label = "KEDİ" if pred == 0 else "KÖPEK"
        print(f"{f} -> {label}  (p_cat={probs[0]:.2f}, p_dog={probs[1]:.2f})")
