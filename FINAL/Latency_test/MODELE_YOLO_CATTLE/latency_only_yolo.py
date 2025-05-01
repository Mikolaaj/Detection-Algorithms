import torch
import time
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import sys
import cv2
from torchvision import transforms
import re

MODEL_FOLDER = "/mnt/d/home/miko/Latency_test/MODELE_YOLO_CATTLE/"
IMAGE_FOLDER = "/mnt/d/home/miko/Latency_test/MODELE_YOLO_CATTLE/test_images_cattle/"
MAP_FILE_PATH = "/mnt/d/home/miko/Latency_test/MODELE_YOLO_CATTLE/dane_mAP_yolo.txt"
YOLOV5_PATH = "/mnt/d/home/miko/Latency_test/yolov5"
OUTPUT_RESULTS_PATH = "/mnt/d/home/miko/Latency_test/MODELE_YOLO_CATTLE/wyniki_latency_map.txt"

sys.path.append(YOLOV5_PATH)
from models.experimental import attempt_load


def measure_latency_on_images(model_path, image_paths, batch_size=8):
    print(f"Ładowanie modelu z: {model_path}")
    model_name = os.path.basename(model_path)
    is_yolov5 = 'yolov5' in model_name.lower()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_yolov5:
        model = attempt_load(model_path)
        model.to(device)
        model.eval()
    else:
        model = YOLO(model_path)
        model.to(device)

    total_time = 0
    num_images = len(image_paths)

    if is_yolov5:
        img = cv2.imread(image_paths[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        model(img.to(device))
    else:
        model.predict(image_paths[0], device=device, verbose=False)

    for i in range(0, num_images, batch_size):
        batch_imgs = [Image.open(p).convert("RGB") for p in image_paths[i:i+batch_size]]
        batch_tensors = [transforms.ToTensor()(img).unsqueeze(0) for img in batch_imgs]
        batch_tensor = torch.cat(batch_tensors)

        start = time.time()
        if is_yolov5:
            model(batch_tensor.to(device))
        else:
            model.predict(batch_imgs, device=device, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        total_time += (end - start)

    avg_latency = (total_time / num_images) * 1000
    return avg_latency


def get_image_paths(folder):
    valid_exts = ('.jpg', '.jpeg', '.png')
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid_exts)]


def parse_map_file(filepath):
    map_dict = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip().strip(';')
        if not line or line.lower().startswith("model"):
            continue
        parts = [x.strip() for x in line.split(',')]
        if len(parts) >= 3:
            model_name = parts[0]
            try:
                map_50 = float(parts[1])
                map_5090 = float(parts[2])
                map_dict[model_name] = {"map50": map_50, "map5090": map_5090}
            except ValueError:
                continue
    return map_dict


def plot_graph(latencies, map_values, model_names, ylabel, title, filename, model_groups, colors):
    plt.figure(figsize=(10, 5))

    for group_idx, (group, color) in enumerate(zip(model_groups, colors)):
        group_latencies = [latencies[i] for i, name in enumerate(model_names) if name in group]
        group_map_values = [map_values[i] for i, name in enumerate(model_names) if name in group]

        plt.plot(group_latencies, group_map_values, '-o', color=color, markerfacecolor='white', markeredgewidth=2)
        for i, name in enumerate(model_names):
            if name in group:
                match = re.search(r'-(\d+)([a-zA-Z]+)', name)
                if match:
                    number_part = match.group(1)
                    letters_part = match.group(2)
                    if int(number_part) in range(1, 5):
                        label = letters_part[0].upper()
                    elif int(number_part) == 5:
                        label = letters_part[:2].upper()
                    plt.text(latencies[i] + 0.2, map_values[i], label, fontsize=9)

    plt.xlabel('Latency (ms/img)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True)
    legend_labels = [f"{group[0].split('-')[0]} models" for group in model_groups if group]
    plt.legend(legend_labels, loc='lower right')
    plt.tight_layout()

    output_path = os.path.join(os.getcwd(), filename)
    plt.savefig(output_path, dpi=300)
    print(f"✅ Wykres zapisany jako: {output_path}")
    plt.show()


def main():
    model_files = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".pt")]
    if not model_files:
        print("❌ Brak plików .pt w folderze.")
        return

    image_paths = get_image_paths(IMAGE_FOLDER)
    if not image_paths:
        print("❌ Brak zdjęć (.jpg/.png) w folderze.")
        return

    map_data = parse_map_file(MAP_FILE_PATH)

    try:
        batch_size = int(input("\n🧪 Podaj batch size (np. 1, 4, 8): "))
    except ValueError:
        print("⚠️ Nieprawidłowy batch size. Używam domyślnego: 1")
        batch_size = 16

    latencies = []
    map50_list = []
    map5090_list = []
    model_names = []

    for model_file in model_files:
        model_name = model_file.replace(".pt", "")
        if model_name not in map_data:
            print(f"⚠️  Brak wpisu mAP dla modelu: {model_name}. Pomijam.")
            continue

        full_path = os.path.join(MODEL_FOLDER, model_file)
        print(f"\n➡️ Model: {model_file}")
        latency = measure_latency_on_images(full_path, image_paths, batch_size=batch_size)
        print(f"   🚀 Średnia latencja: {latency:.2f} ms/img")

        map50 = map_data[model_name]["map50"]
        map5090 = map_data[model_name]["map5090"]

        model_names.append(model_name)
        latencies.append(latency)
        map50_list.append(map50)
        map5090_list.append(map5090)

    yolov8_models = [name for name in model_names if 'yolov8' in name.lower()]
    yolov5_models = [name for name in model_names if 'yolov5' in name.lower()]
    v11_models = [name for name in model_names if 'v11' in name.lower()]
    v12_models = [name for name in model_names if 'v12' in name.lower()]

    def sort_key(name):
        match = re.search(r'-(\d+)', name)
        return int(match.group(1)) if match else float('inf')

    yolov8_models.sort(key=sort_key)
    yolov5_models.sort(key=sort_key)
    v11_models.sort(key=sort_key)
    v12_models.sort(key=sort_key)

    model_groups = [yolov8_models, yolov5_models, v11_models, v12_models]
    colors = ['blue', 'green', 'orange', 'purple']

    plot_graph(
        latencies,
        map5090_list,
        model_names,
        ylabel='COCO mAP 50:95',
        title='Metryka wydajności YOLO i EfficientDet na GPU',
        filename="DT_graphs_latency_vs_map50_90.png",
        model_groups=model_groups,
        colors=colors
    )

    plot_graph(
        latencies,
        map50_list,
        model_names,
        ylabel='COCO mAP 50',
        title='Metryka wydajności YOLO i EfficientDet na GPU',
        filename="DT_graphs_latency_vs_map50.png",
        model_groups=model_groups,
        colors=colors
    )

    with open(OUTPUT_RESULTS_PATH, 'w') as f:
        f.write("Model, Latency (FPS), mAP50, mAP50:95\n")
        for name, lat, m50, m5090 in zip(model_names, latencies, map50_list, map5090_list):
            f.write(f"{name}, {lat:.2f}, {m50:.3f}, {m5090:.3f}\n")

    print(f"\n📁 Wyniki zapisane do pliku: {OUTPUT_RESULTS_PATH}")


if __name__ == "__main__":
    main()
