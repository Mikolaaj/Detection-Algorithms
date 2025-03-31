import cv2
import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from ultralytics import YOLO  # Import YOLO
from collections import defaultdict
from scipy.interpolate import make_interp_spline

def choose_model():
    model_path = filedialog.askopenfilename(title="Wybierz model YOLO", filetypes=[("Pliki PyTorch", "*.pt")])
    return model_path

def choose_video():
    video_path = filedialog.askopenfilename(title="Wybierz plik wideo", filetypes=[("Pliki wideo", "*.mp4;*.avi;*.mov;*.mkv")])
    return video_path

def run_detection(video_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Nie można otworzyć pliku wideo")
        return
    
    window_name = "Detekcja YOLOv8"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    frame_count = 0
    confidence_dict = defaultdict(list)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        results = model(frame)
        
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
    if not confidence_dict:
        print("Brak danych do wygenerowania wykresów.")
        return
    
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    
    for label, values in confidence_dict.items():
        frame_numbers, confidence_scores = zip(*values)
        
        plt.figure(figsize=(8, 5))
        plt.plot(frame_numbers, confidence_scores, marker='o', label=label)
        
        min_conf = min(confidence_scores)
        max_conf = max(confidence_scores)
        min_index = confidence_scores.index(min_conf)
        max_index = confidence_scores.index(max_conf)
        
        plt.annotate(f'Min: {min_conf:.2f}', xy=(frame_numbers[min_index], min_conf), 
                     xytext=(frame_numbers[min_index], min_conf - 0.1),
                     arrowprops=dict(facecolor='red', shrink=0.05))
        plt.annotate(f'Max: {max_conf:.2f}', xy=(frame_numbers[max_index], max_conf), 
                     xytext=(frame_numbers[max_index], max_conf + 0.1),
                     arrowprops=dict(facecolor='green', shrink=0.05))
        
        plt.xlabel("Frame Index")
        plt.ylabel("Confidence Score")
        plt.title(f"Confidence vs Frame Index - {label}")
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(desktop_path, f"confidence_vs_frame_{label}.png")
        plt.savefig(plot_path)
        plt.show()
        print(f"📁 Wykres dla {label} zapisany: {plot_path}")
        
        # Aproksymacja wykresu z usunięciem duplikatów
        unique_dict = {}
        for f, c in sorted(values):
            unique_dict[f] = c  # Przypisanie wartości, co usunie duplikaty
        
        unique_frames = list(unique_dict.keys())
        unique_confidences = list(unique_dict.values())
        
        if len(unique_frames) > 3:  # Sprawdzamy, czy wystarczy punktów do aproksymacji
            frame_smooth = np.linspace(min(unique_frames), max(unique_frames), 300)
            confidence_smooth = make_interp_spline(unique_frames, unique_confidences)(frame_smooth)
            
            plt.figure(figsize=(8, 5))
            plt.plot(frame_smooth, confidence_smooth, label=f"Approximated {label}", color='orange')
            plt.xlabel("Frame Index")
            plt.ylabel("Confidence Score")
            plt.title(f"Approximated Confidence vs Frame Index - {label}")
            plt.legend()
            plt.grid(True)
            
            approx_plot_path = os.path.join(desktop_path, f"confidence_approx_{label}.png")
            plt.savefig(approx_plot_path)
            plt.show()
            print(f"📁 Wykres aproksymacyjny dla {label} zapisany: {approx_plot_path}")

if __name__ == "__main__":
    print("Wybierz plik modelu YOLOv8")
    model_path = choose_model()
    if not model_path:
        print("Nie wybrano modelu. Kończenie programu.")
        exit()
    
    print("Wybierz plik wideo")
    video_path = choose_video()
    if not video_path:
        print("Nie wybrano pliku wideo. Kończenie programu.")
        exit()
    
    model = YOLO(model_path)
    run_detection(video_path, model)
