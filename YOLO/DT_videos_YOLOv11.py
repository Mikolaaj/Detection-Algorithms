import cv2
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
from ultralytics import YOLO  # Import YOLOv11
from collections import defaultdict
from scipy.interpolate import make_interp_spline

def choose_file(title, file_types):
    """Wybór pliku przez okno dialogowe"""
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=file_types)
    return file_path

def run_detection(video_path, model, device):
    """Przetwarzanie wideo i wykrywanie obiektów YOLOv11"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Nie można otworzyć pliku wideo.")
        return
    
    window_name = "Detekcja YOLOv11"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    frame_count = 0
    confidence_dict = defaultdict(list)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        results = model(frame, device=device)
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = model.names[class_id]
                confidence_dict[label].append((frame_count, confidence))
        
        frame = results[0].plot()
        frame_resized = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_LINEAR)
        
        cv2.imshow(window_name, frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    generate_plots(confidence_dict)

def generate_plots(confidence_dict):
    """Tworzenie wykresów na podstawie wyników detekcji"""
    if not confidence_dict:
        print("📉 Brak danych do wygenerowania wykresów.")
        return
    
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    
    for label, values in confidence_dict.items():
        frame_numbers, confidence_scores = zip(*values)
        
        plt.figure(figsize=(8, 5))
        plt.plot(frame_numbers, confidence_scores, marker='o', label=label)
        
        min_conf, max_conf = min(confidence_scores), max(confidence_scores)
        min_index, max_index = confidence_scores.index(min_conf), confidence_scores.index(max_conf)
        
        # Dodanie poziomych linii dla min i max
        plt.hlines(y=min_conf, xmin=min(frame_numbers), xmax=max(frame_numbers), colors='red', linestyles='dashed', label=f'Min: {min_conf:.2f}')
        plt.hlines(y=max_conf, xmin=min(frame_numbers), xmax=max(frame_numbers), colors='green', linestyles='dashed', label=f'Max: {max_conf:.2f}')
        
        plt.xlabel("Numer klatki")
        plt.ylabel("Prawdopodobieństwo")
        plt.title(f"Detekcja obiektów - {label}")
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(desktop_path, f"confidence_vs_frame_{label}.png")
        plt.savefig(plot_path)
        plt.show()
        print(f"📁 Wykres dla {label} zapisany: {plot_path}")
        
        # Aproksymacja danych do płynniejszej wizualizacji
        unique_dict = {f: c for f, c in sorted(values)}
        unique_frames, unique_confidences = list(unique_dict.keys()), list(unique_dict.values())
        
        if len(unique_frames) > 3:  # Sprawdzamy, czy wystarczy punktów do aproksymacji
            frame_smooth = np.linspace(min(unique_frames), max(unique_frames), 300)
            confidence_smooth = make_interp_spline(unique_frames, unique_confidences)(frame_smooth)
            
            plt.figure(figsize=(8, 5))
            plt.plot(frame_smooth, confidence_smooth, label=f"Aproksymacja - {label}", color='orange')
            plt.xlabel("Numer klatki")
            plt.ylabel("Prawdopodobieństwo")
            plt.title(f"Aproksymowana detekcja obiektów - {label}")
            plt.legend()
            plt.grid(True)
            
            approx_plot_path = os.path.join(desktop_path, f"confidence_approx_{label}.png")
            plt.savefig(approx_plot_path)
            plt.show()
            print(f"📁 Wykres aproksymacyjny dla {label} zapisany: {approx_plot_path}")

if __name__ == "__main__":
    print("📌 Wybierz plik modelu YOLOv11")
    model_path = choose_file("Wybierz model YOLO", [("Pliki PyTorch", "*.pt")])
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

    print("📂 Wybierz plik wideo")
    video_path = choose_file("Wybierz plik wideo", [("Pliki wideo", "*.mp4;*.avi;*.mov;*.mkv")])
    if not video_path:
        print("❌ Nie wybrano pliku wideo. Kończenie programu.")
        exit()

    run_detection(video_path, model, device)
