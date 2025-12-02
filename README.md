Tomato Leaf Health Check

Bu proje, domates yapraklarının sağlıklı mı yoksa hastalığa sahip mi olduğunu tespit eden basit bir derin öğrenme uygulamasıdır. Görüntüler üzerinde eğitim alan model, tek bir yaprak fotoğrafından tahmin üretebilir ve GUI arayüzü sayesinde kullanıcılar kendi görsellerini yükleyerek sonucu anında görebilir.

🚀 Özellikler

Kendi veri setiyle eğitim (healthy / early blight)

Görüntü ön işleme (resize, normalization, augmentation)

Basit ve anlaşılır bir CNN mimarisi

Eğitim sonrası .h5 model kaydı

Fotoğraf seçildiğinde sonucu gösteren kullanıcı dostu bir Tkinter arayüzü

Model çıktısına göre renkli durum kartları ve güven yüzdesi

🧠 Model

Model, Keras kullanılarak oluşturulmuş küçük bir CNN yapısından oluşuyor.
Eğitim sürecinde:

128×128 çözünürlük

Adam optimizer

Binary cross-entropy

Accuracy & loss takibi

Elde edilen model leaf_model.h5 adıyla kaydedilir ve GUI tarafından kullanılır.

📦 Proje Yapısı
plant_disease_project/
│
├── preprocessing.py        # Veri okuma, augmenting ve generator'lar
├── model_kfold.py          # K-Fold denemeleri
├── train_final_model.py    # Nihai model eğitimi
├── gui_test.py             # Tkinter arayüzü
├── .gitignore
└── data/                   # (GitHub'a dahil değil)

🖥️ GUI Önizleme

Arayüz, kullanıcıya fotoğraf seçme butonu, görsel önizleme ve model çıktısı sunar.
Yaprak sağlıklıysa yeşil, hastalıklıysa kırmızı bir durum kartı görüntülenir.

🔧 Kurulum
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


GUI çalıştırmak için:

python gui_test.py

