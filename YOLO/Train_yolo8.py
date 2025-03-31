import os
import torch
import tkinter as tk
from tkinter import filedialog
import multiprocessing
from ultralytics import YOLO

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Trening będzie wykonywany na: {device.upper()}")

    def choose_dataset():
        root = tk.Tk()
        root.withdraw()
        dataset_path = filedialog.askdirectory(title="Wybierz folder zawierający zbiór treningowy")
        return dataset_path

    print("Wybierz folder zawierający zbiór treningowy YOLO")
    dataset_path = choose_dataset()

    if not dataset_path:
        print("Nie wybrano folderu. Kończenie programu.")
        exit()

    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml_path):
        print("Brak pliku data.yaml w folderze. Upewnij się, że dane są w formacie YOLO.")
        exit()

    num_workers = min(4, multiprocessing.cpu_count())
    #num_workers = 4
    
    try:
        epochs = int(input("Podaj liczbę epok (np. 50): "))
    except ValueError:
        print("Błędna wartość, ustawiam 50 epok domyślnie.")
        epochs = 50

    model = YOLO("yolov8n.pt")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = "cuda:0"

    print(f"🔄 Rozpoczynam trening na zbiorze: {dataset_path} przez {epochs} epok...")

    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=640,
        workers=num_workers,
        device=device,
        batch=64,
        amp=True,
        half=True,
        cache=False,
        verbose=True,
        save=True,
    )

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    model_save_path_pt = os.path.join(desktop_path, "trained_yolo_model_YOLO8.pt")

    best_model_path = "runs/detect/train/weights/best.pt"
    if os.path.exists(best_model_path):
        os.rename(best_model_path, model_save_path_pt)
        print(f"✅ Model YOLOv8 zapisany: {model_save_path_pt}")
    else:
        print("⚠️ Nie znaleziono zapisanego modelu. Sprawdź katalog runs/detect/train/weights.")

    print("🎉 Trening zakończony sukcesem!")

if __name__ == "__main__":
    main()
