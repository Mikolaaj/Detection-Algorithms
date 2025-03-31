import cv2
import torch
import os
import numpy as np
from tkinter import filedialog, Tk
from ultralytics import YOLO  # Import YOLOv11

def choose_model():
    """Wybór modelu YOLOv11 przez okno dialogowe"""
    root = Tk()
    root.withdraw()
    model_path = filedialog.askopenfilename(title="Wybierz model YOLOv11", filetypes=[("Pliki PyTorch", "*.pt")])
    return model_path

def choose_image_folder():
    """Wybór folderu ze zdjęciami przez okno dialogowe"""
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Wybierz folder ze zdjęciami")
    return folder_path

def process_images(model, folder_path, device):
    """Przetwarzanie obrazów z folderu za pomocą YOLOv11"""
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    if not image_files:
        print("❌ Brak obrazów w wybranym folderze.")
        return
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Nie można załadować obrazu: {image_file}")
            continue

        print(f"📷 Przetwarzanie: {image_file}...")
        results = model(image, device=device)  # YOLOv11 wykrywa obiekty
        
        detected_image = results[0].plot()  # Rysowanie wykrytych obiektów
        cv2.imshow("Detekcja YOLOv11", detected_image)
        
        if cv2.waitKey(0) & 0xFF == ord('q'):  # Zamknij okno po naciśnięciu 'q'
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("📌 Wybierz plik modelu YOLOv11")
    model_path = choose_model()
    if not model_path:
        print("❌ Nie wybrano modelu. Kończenie programu.")
        exit()

    # Wybór urządzenia (GPU lub CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Używane urządzenie: {device.upper()}")

    try:
        model = YOLO(model_path)  # YOLOv11 model
    except Exception as e:
        print(f"❌ Błąd podczas ładowania modelu: {e}")
        exit()

    print("📂 Wybierz folder ze zdjęciami")
    folder_path = choose_image_folder()
    if not folder_path:
        print("❌ Nie wybrano folderu. Kończenie programu.")
        exit()

    process_images(model, folder_path, device)
