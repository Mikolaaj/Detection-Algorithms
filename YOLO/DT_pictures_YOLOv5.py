import cv2
import torch
import os
import numpy as np
from tkinter import filedialog, Tk

def choose_model():
    root = Tk()
    root.withdraw()
    model_path = filedialog.askopenfilename(title="Wybierz model YOLO", filetypes=[("Pliki PyTorch", "*.pt")])
    return model_path

def choose_image_folder():
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Wybierz folder ze zdjęciami")
    return folder_path

def process_images(model, folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not image_files:
        print("Brak obrazów w wybranym folderze.")
        return
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Nie można załadować obrazu: {image_file}")
            continue
        
        # Detekcja obiektów
        results = model(image)  
        detected_image = results.render()[0]  # Rysowanie wykrytych obiektów
        
        cv2.imshow("Detekcja YOLOv5", detected_image)
        print(f"Wyświetlanie: {image_file}")
        if cv2.waitKey(0) & 0xFF == ord('q'):  # Zamknij okno po naciśnięciu 'q'
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Wybierz plik modelu YOLOv5")
    model_path = choose_model()
    if not model_path:
        print("Nie wybrano modelu. Kończenie programu.")
        exit()
    
    # Załaduj model YOLOv5 z pliku .pt
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # Załaduj model YOLOv5
    
    print("Wybierz folder ze zdjęciami")
    folder_path = choose_image_folder()
    if not folder_path:
        print("Nie wybrano folderu. Kończenie programu.")
        exit()
    
    process_images(model, folder_path)
