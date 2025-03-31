import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import filedialog
from ultralytics import YOLO  # Import YOLOv11
from threading import Thread
from scipy.interpolate import make_interp_spline

def choose_model():
    """Wybór modelu YOLOv11 przez okno dialogowe"""
    model_path = filedialog.askopenfilename(title="Wybierz model YOLOv11", filetypes=[("Pliki PyTorch", "*.pt")])
    return model_path

def run_yolo():
    """Uruchomienie YOLOv11 na streamie z kamery"""
    global running, frame_indices, confidence_dict, class_colors
    frame_indices = []  # Lista numerów klatek
    confidence_dict = {}  # Słownik przechowujący wartości prawdopodobieństwa dla każdej klasy
    class_colors = {}  # Słownik kolorów dla każdej klasy

    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < 10:  # Stream przez 10 sekund
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_time = time.time() - start_time
        frame_count += 1
        frame_indices.append(frame_count)

        results = model(frame, device=device)  # YOLOv11 wykrywa obiekty

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = model.names[class_id] if class_id < len(model.names) else f"Class {class_id}"
                
                if label not in confidence_dict:
                    confidence_dict[label] = []
                confidence_dict[label].append((frame_count, confidence))
                
                if label not in class_colors:
                    class_colors[label] = np.random.rand(3,)

                frame = result.plot()

        cv2.putText(frame, f"Czas trwania: {elapsed_time:.2f} s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Detekcja YOLOv11", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def remove_duplicates(x, y):
    """Usuwa duplikaty wartości x, uśredniając odpowiadające im wartości y."""
    unique_x, unique_y = [], {}
    for i in range(len(x)):
        if x[i] in unique_y:
            unique_y[x[i]].append(y[i])
        else:
            unique_y[x[i]] = [y[i]]

    for key in sorted(unique_y.keys()):
        unique_x.append(key)
        unique_y[key] = np.mean(unique_y[key])  # Średnia wartość y dla powtarzających się x

    return unique_x, list(unique_y.values())

def generate_plots():
    """Generowanie wykresów pewności detekcji"""
    if not confidence_dict:
        print("Brak danych do wygenerowania wykresów.")
        return
    
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    
    for label, values in confidence_dict.items():
        frame_numbers, confidence_scores = zip(*values)

        # Usuń duplikaty przed interpolacją
        frame_numbers, confidence_scores = remove_duplicates(frame_numbers, confidence_scores)

        plt.figure(figsize=(8, 5))
        plt.plot(frame_numbers, confidence_scores, marker='o', color=class_colors[label], label=label)
        
        # Dodanie poziomych linii dla min/max
        min_conf = min(confidence_scores)
        max_conf = max(confidence_scores)
        plt.axhline(min_conf, color='red', linestyle='--', label=f'Min: {min_conf:.2f}')
        plt.axhline(max_conf, color='green', linestyle='--', label=f'Max: {max_conf:.2f}')
        
        plt.xlabel("Frame Index")
        plt.ylabel("Confidence Score")
        plt.title(f"Confidence vs Frame Index - {label}")
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(desktop_path, f"confidence_vs_frame_{label}.png")
        plt.savefig(plot_path)
        plt.show()
        print(f"📁 Wykres dla {label} zapisany: {plot_path}")
        
        # Tworzenie wykresu z aproksymacją
        if len(frame_numbers) > 3:  
            frame_smooth = np.linspace(min(frame_numbers), max(frame_numbers), 300)
            confidence_smooth = make_interp_spline(frame_numbers, confidence_scores)(frame_smooth)
            
            plt.figure(figsize=(8, 5))
            plt.plot(frame_smooth, confidence_smooth, color=class_colors[label], label=f"Aproksymacja - {label}")
            plt.xlabel("Frame Index")
            plt.ylabel("Confidence Score")
            plt.title(f"Confidence Approximation - {label}")
            plt.legend()
            plt.grid(True)
            
            approx_plot_path = os.path.join(desktop_path, f"confidence_approx_{label}.png")
            plt.savefig(approx_plot_path)
            plt.show()
            print(f"📁 Wykres aproksymacji dla {label} zapisany: {approx_plot_path}")

if __name__ == "__main__":
    print("Wybierz plik modelu YOLOv11")
    model_path = choose_model()
    if not model_path:
        print("Nie wybrano modelu. Kończenie programu.")
        exit()

    # Wybór urządzenia (GPU lub CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Używane urządzenie: {device.upper()}")

    try:
        model = YOLO(model_path)  # YOLOv11 model
    except Exception as e:
        print(f"❌ Błąd podczas ładowania modelu: {e}")
        exit()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nie można otworzyć kamery")
        exit()

    running = True  
    yolo_thread = Thread(target=run_yolo)
    yolo_thread.start()
    yolo_thread.join()

    generate_plots()
