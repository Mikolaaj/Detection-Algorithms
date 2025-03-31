import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import filedialog
from threading import Thread
from scipy.interpolate import make_interp_spline
from yolov5 import YOLOv5  # Import YOLOv5

def choose_model():
    model_path = filedialog.askopenfilename(title="Wybierz model YOLO", filetypes=[("Pliki PyTorch", "*.pt")])
    return model_path

def run_yolo():
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

        results = model.predict(frame)
        detections = results.xyxy[0].cpu().numpy()

        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            class_id = int(class_id)
            confidence = float(conf)
            label = model.model.names[class_id]
            
            if label not in confidence_dict:
                confidence_dict[label] = []
            confidence_dict[label].append((frame_count, confidence))
            
            if label not in class_colors:
                class_colors[label] = np.random.rand(3,)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.putText(frame, f"Czas trwania: {elapsed_time:.2f} s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Detekcja YOLOv5", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def remove_duplicates(x, y):
    unique_x, unique_y = [], {}
    for i in range(len(x)):
        if x[i] in unique_y:
            unique_y[x[i]].append(y[i])
        else:
            unique_y[x[i]] = [y[i]]

    for key in sorted(unique_y.keys()):
        unique_x.append(key)
        unique_y[key] = np.mean(unique_y[key])

    return unique_x, list(unique_y.values())

def generate_plots():
    if not confidence_dict:
        print("Brak danych do wygenerowania wykresów.")
        return
    
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    
    for label, values in confidence_dict.items():
        frame_numbers, confidence_scores = zip(*values)

        frame_numbers, confidence_scores = remove_duplicates(frame_numbers, confidence_scores)

        plt.figure(figsize=(8, 5))
        plt.plot(frame_numbers, confidence_scores, marker='o', color=class_colors[label], label=label)
        
        # Dodanie poziomej linii dla minimalnej i maksymalnej wartości
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
    print("Wybierz plik modelu YOLOv5")
    model_path = choose_model()
    if not model_path:
        print("Nie wybrano modelu. Kończenie programu.")
        exit()

    model = YOLOv5(model_path)  
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nie można otworzyć kamery")
        exit()

    running = True  
    yolo_thread = Thread(target=run_yolo)
    yolo_thread.start()
    yolo_thread.join()

    generate_plots()
