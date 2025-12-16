import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# -------------------------------
# -------------------------------
IMG_HEIGHT = 128
IMG_WIDTH  = 128

MODEL_PATH = "leaf_model.h5"

model = load_model(MODEL_PATH)

root = tk.Tk()
root.title("Yaprak Hastalığı Tespiti")
root.geometry("700x850")
root.configure(bg="#0f0f23")

header_frame = tk.Frame(root, bg="#1a1a3e", height=100)
header_frame.pack(fill=tk.X)
header_frame.pack_propagate(False)

title_label = tk.Label(
    header_frame, 
    text="🌿 AI Yaprak Analiz Sistemi", 
    font=("Helvetica", 24, "bold"),
    bg="#1a1a3e",
    fg="#00ff88"
)
title_label.pack(pady=15)

subtitle_label = tk.Label(
    header_frame,
    text="Yapay Zeka Destekli Hastalık Tespiti",
    font=("Helvetica", 11),
    bg="#1a1a3e",
    fg="#64ffda"
)
subtitle_label.pack()

content_frame = tk.Frame(root, bg="#0f0f23")
content_frame.pack(expand=True, fill=tk.BOTH, padx=40, pady=30)

image_card = tk.Frame(content_frame, bg="#1e1e3f", relief=tk.FLAT, bd=0)
image_card.pack(pady=20)

border_frame = tk.Frame(image_card, bg="#00ff88", bd=0)
border_frame.pack(padx=2, pady=2)

img_frame = tk.Frame(border_frame, bg="#1e1e3f", relief=tk.FLAT, bd=0)
img_frame.pack()

img_label = tk.Label(img_frame, bg="#1e1e3f", width=40, height=15)
img_label.pack(padx=15, pady=15)

placeholder_frame = tk.Frame(img_label, bg="#1e1e3f")
placeholder_frame.place(relx=0.5, rely=0.5, anchor="center")

placeholder_icon = tk.Label(
    placeholder_frame,
    text="📸",
    font=("Helvetica", 48),
    bg="#1e1e3f",
    fg="#00ff88"
)
placeholder_icon.pack()

placeholder_text = tk.Label(
    placeholder_frame,
    text="Yaprak Görseli Yükleyin",
    font=("Helvetica", 13),
    bg="#1e1e3f",
    fg="#64ffda"
)
placeholder_text.pack(pady=10)

result_container = tk.Frame(content_frame, bg="#0f0f23")
result_container.pack(pady=25, fill=tk.X)

result_left = tk.Frame(result_container, bg="#1e1e3f", relief=tk.FLAT, bd=0)
result_left.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)

result_right = tk.Frame(result_container, bg="#1e1e3f", relief=tk.FLAT, bd=0)
result_right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5)

status_title = tk.Label(
    result_left,
    text="DURUM",
    font=("Helvetica", 10, "bold"),
    bg="#1e1e3f",
    fg="#64ffda"
)
status_title.pack(pady=(15, 5))

status_label = tk.Label(
    result_left,
    text="Bekleniyor...",
    font=("Helvetica", 16, "bold"),
    bg="#1e1e3f",
    fg="#ffffff",
    wraplength=250
)
status_label.pack(pady=(0, 15))

stats_title = tk.Label(
    result_right,
    text="ANALİZ DETAYI",
    font=("Helvetica", 10, "bold"),
    bg="#1e1e3f",
    fg="#64ffda"
)
stats_title.pack(pady=(15, 5))

stats_label = tk.Label(
    result_right,
    text="─",
    font=("Helvetica", 16, "bold"),
    bg="#1e1e3f",
    fg="#ffffff"
)
stats_label.pack(pady=(0, 15))

progress_container = tk.Frame(content_frame, bg="#0f0f23")
progress_container.pack(pady=10, fill=tk.X)

progress_bg = tk.Frame(progress_container, bg="#1e1e3f", height=8, relief=tk.FLAT, bd=0)
progress_bg.pack(fill=tk.X, padx=50)

progress_bar = tk.Frame(progress_bg, bg="#00ff88", height=8, width=0, relief=tk.FLAT, bd=0)
progress_bar.place(x=0, y=0)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)


    prob = float(model.predict(img_array, verbose=0)[0][0])  # sigmoid çıktı

    THRESHOLD = 0.5  # eski hal

    if prob >= THRESHOLD:
        label = "Sağlıklı (healthy)"
    else:
        label = "Hasta (early blight)"

    return label, prob


def animate_progress(target_width):
    """Progress bar animasyonu"""
    current_width = progress_bar.winfo_width()
    if current_width < target_width:
        progress_bar.config(width=current_width + 5)
        root.after(10, lambda: animate_progress(target_width))


def open_image():
    """Dosya seç, resmi göster, tahmin yap."""
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if not file_path:
        return

    placeholder_frame.place_forget()

    pil_img = Image.open(file_path)
    pil_img = pil_img.resize((320, 320))
    tk_img = ImageTk.PhotoImage(pil_img)

    img_label.config(image=tk_img)
    img_label.image = tk_img

    label, prob = predict_image(file_path)
    
    if "Sağlıklı" in label:
        status_icon = "✅"
        status_text = "Sağlıklı Yaprak"
        status_color = "#00ff88"
        border_color = "#00ff88"
        confidence_percent = prob * 100
        bar_color = "#00ff88"
    else:
        status_icon = "⚠️"
        status_text = "Hastalık Tespit Edildi"
        status_color = "#ff6b6b"
        border_color = "#ff6b6b"
        confidence_percent = (1 - prob) * 100  # Hastalık için ters çevir
        bar_color = "#ff6b6b"
    
    border_frame.config(bg=border_color)
    progress_bar.config(bg=bar_color)
    
    status_label.config(
        text=f"{status_icon} {status_text}",
        fg=status_color
    )
    
    stats_label.config(
        text=f"Model Güveni\n{confidence_percent:.1f}%",
        fg="#ffffff"
    )
    
    max_width = progress_bg.winfo_width()
    target_width = int(max_width * (confidence_percent / 100))
    progress_bar.config(width=0)
    root.after(100, lambda: animate_progress(target_width))

button_container = tk.Frame(content_frame, bg="#0f0f23")
button_container.pack(pady=20)

glow_frame = tk.Frame(button_container, bg="#00ff88", bd=0)
glow_frame.pack(padx=3, pady=3)

btn = tk.Button(
    glow_frame,
    text="🔍  Görsel Analiz Et",
    command=open_image,
    font=("Helvetica", 14, "bold"),
    bg="#00ff88",
    fg="#0f0f23",
    activebackground="#64ffda",
    activeforeground="#0f0f23",
    relief=tk.FLAT,
    bd=0,
    padx=40,
    pady=15,
    cursor="hand2"
)
btn.pack()

# Hover efektleri
def on_enter(e):
    btn.config(bg="#64ffda")
    glow_frame.config(bg="#64ffda")

def on_leave(e):
    btn.config(bg="#00ff88")
    glow_frame.config(bg="#00ff88")

btn.bind("<Enter>", on_enter)
btn.bind("<Leave>", on_leave)

info_frame = tk.Frame(content_frame, bg="#1e1e3f", relief=tk.FLAT, bd=0)
info_frame.pack(pady=15, fill=tk.X)

info_label = tk.Label(
    info_frame,
    text="BERAT ERDOGAN © 2025 • Yaprak Hastalığı Tespit Uygulaması",
    font=("Helvetica", 9),
    bg="#1e1e3f",
    fg="#64ffda",
    pady=10
)
info_label.pack()

footer_label = tk.Label(
    root,
    text="Powered by TensorFlow & Keras • Deep Learning Technology",
    font=("Helvetica", 9),
    bg="#0f0f23",
    fg="#4a5568"
)
footer_label.pack(side=tk.BOTTOM, pady=15)

root.mainloop()