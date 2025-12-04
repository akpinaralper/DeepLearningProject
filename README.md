
#  Ses Tabanlı Kedi–Köpek Sınıflandırma (MFCC + CNN)

Bu proje, kısa ses kayıtlarını (WAV) kullanarak **kedi mi yoksa köpek mi** olduğunu tahmin eden bir derin öğrenme modelidir. Sesler üzerinden **MFCC (Mel-Frequency Cepstral Coefficients)** çıkarılmış, ardından **CNN (Convolutional Neural Network)** modeli ile sınıflandırma yapılmıştır.

Proje; veri toplama, ön işleme, model eğitimi ve Gradio tabanlı demo arayüzü ile **uçtan uca çalışan bir sistem** sunmaktadır.

---

## 📌 Özellikler
- WAV formatında ses dosyası girişi  
- Librosa ile MFCC çıkarımı  
- PyTorch CNN ile ses sınıflandırma  
- Gradio web arayüzü  
- Cat / Dog olmak üzere 2 sınıf  
- Küçük veri setiyle hızlı eğitim  

---

## 📁 Proje Yapısı


                    project/ 
                    │── model.py # CNN model mimarisi
                    │── train.py # Eğitim scripti
                    │── serve.py # Gradio arayüzü (web demo)
                    │── dataset/
                    │ ├── cat/ # Kedi sesleri (wav)
                    │ └── dog/ # Köpek sesleri (wav)
                    │── requirements.txt # Gerekli paketler
                    │── README.md # Bu dosya

## 🎯 Amaç

- Ses tabanlı sınıflandırma sürecini anlamak  
- Ses sinyallerinden MFCC çıkarma  
- CNN tabanlı bir modelin ses verisi üzerinde çalışmasını göstermek  
- Uçtan uca AI uygulaması (eğitim + inference + web UI) oluşturmak  

---

## 🧠 Kullanılan Yöntem

### 📍 1. MFCC Özellik Çıkarımı
Ses dosyaları zaman domeninden frekans domenine dönüştürülerek **40 MFCC katsayısı** çıkarılmıştır.

```
mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
mfcc = librosa.util.fix_length(mfcc, size=20, axis=1)
````
📍 2. CNN Modeli
```
Model mimarisi:
Conv2D (1 → 16)
ReLU
Conv2D (16 → 32)
MaxPool (2×2)
Flatten
Dense (32 × 20 × 10 → 64)
Dense (64 → 2)

````


```
3.🔧 Kurulum
Aşağıdaki paketleri yükle:

pip install torch librosa gradio soundfile numpy

````

```
4.🏋️ Modeli Eğitme
Dataset klasörünü şu şekilde düzenleyin:
dataset/
    cat/
        meow1.wav
        meow2.wav
        ...
    dog/
        bark1.wav
        bark2.wav
        ...
````

```
5.Eğitim tamamlanınca proje klasöründe:
audio_model.pth
oluşacaktır.
````

```
6.🚀 Gradio Arayüzünü Çalıştırma
python serve.py

Terminalde çıkan link üzerinden web arayüzüne erişebilirsiniz:
http://127.0.0.1:7860

Ardından bir kedi veya köpek sesi yükleyerek test edebilirsiniz.
````










