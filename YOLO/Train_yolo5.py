import os
import torch
import tkinter as tk
from tkinter import filedialog
import multiprocessing
from pathlib import Path
from yolov5.train import run

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

    num_workers = min(2, multiprocessing.cpu_count())

    try:
        epochs = int(input("Podaj liczbę epok (np. 50): "))
    except ValueError:
        print("Błędna wartość, ustawiam 50 epok domyślnie.")
        epochs = 50

    model_name = "yolov5s.pt"  # Możesz zmienić na 'yolov5m.pt', 'yolov5l.pt' itd.

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = "cuda:0"

    print(f"🔄 Rozpoczynam trening na zbiorze: {dataset_path} przez {epochs} epok...")

    run(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=640,
        workers=num_workers,
        device=device,
        batch_size=32,
        weights=model_name,
        project="runs/train",
        name="exp",
        exist_ok=True
    )

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    model_save_path_pt = os.path.join(desktop_path, "trained_yolo_model_YOLO5.pt")

    best_model_path = Path("runs/train/exp/weights/best.pt")
    if best_model_path.exists():
        best_model_path.rename(model_save_path_pt)
        print(f"✅ Model YOLOv5 zapisany: {model_save_path_pt}")
    else:
        print("⚠️ Nie znaleziono zapisanego modelu. Sprawdź katalog runs/train/exp/weights.")

    print("🎉 Trening zakończony sukcesem!")

if __name__ == "__main__":
    main()
