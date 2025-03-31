import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
from matplotlib.widgets import Button

# Załaduj procesor i model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Folder ze zdjęciami
image_folder = "/path/to/your/images"  # Podaj ścieżkę do swojego folderu ze zdjęciami
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Indeks bieżącego obrazu
current_index = 0

# Funkcja do wyświetlania obrazu i predykcji
def show_image(index):
    global current_index
    image_path = os.path.join(image_folder, image_files[index])
    image = Image.open(image_path)
    
    # Wykonaj detekcję
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    
    # Tworzenie wykresu
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    # Rysowanie prostokątów wokół wykrytych obiektów
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color="red", linewidth=3))
        ax.text(box[0], box[1], f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}", fontsize=8, color="red")
    
    ax.axis("off")
    plt.show()

# Funkcja nawigacyjna do zmiany obrazu
def next_image(event):
    global current_index
    current_index = (current_index + 1) % len(image_files)
    show_image(current_index)

def previous_image(event):
    global current_index
    current_index = (current_index - 1) % len(image_files)
    show_image(current_index)

# Tworzenie interaktywnego interfejsu
fig, ax = plt.subplots(figsize=(12, 9))
plt.subplots_adjust(bottom=0.2)

# Przyciski do przewijania
ax_prev = plt.axes([0.1, 0.01, 0.1, 0.075])
ax_next = plt.axes([0.8, 0.01, 0.1, 0.075])
btn_prev = Button(ax_prev, 'Previous')
btn_next = Button(ax_next, 'Next')

btn_prev.on_clicked(previous_image)
btn_next.on_clicked(next_image)

# Pokaż pierwszy obraz
show_image(current_index)

# Wyświetl wykres
plt.show()