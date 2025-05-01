# import os
# import time
# import torch
# from torch.backends import cudnn
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from backbone import EfficientDetBackbone
# from efficientdet.utils import BBoxTransform, ClipBoxes
# from utils.utils import preprocess, invert_affine, postprocess, plot_one_box

# # Ustawienia
# compound_coef = 2  # Zmień na wartość zgodną z Twoim modelem (np. D2)
# weights_path = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/logs/cattle_coco/efficientdet-d2_49_8500.pth'
# input_dir = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/cattle_test/'  # Ścieżka do katalogu z obrazami
# output_dir = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/DETEC_PICS/'  # Ścieżka do katalogu wyników
# plot_dir = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/GRAPHS_EFFDET/'  # Katalog na wykresy
# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(plot_dir, exist_ok=True)

# # Konfiguracja detekcji
# threshold = 0.2  # Próg wykrywania
# iou_threshold = 0.2  # Próg IoU
# use_cuda = True  # Używanie GPU
# num_classes = 1  # Liczba klas (np. 1 jeśli wykrywa tylko butelki)
# obj_list = ['bottle']  # Lista obiektów

# # Ładowanie modelu
# model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=num_classes)
# model.load_state_dict(torch.load(weights_path, map_location='cpu'))
# model.requires_grad_(False)
# model.eval()

# if use_cuda:
#     model = model.cuda()

# # Przetwarzanie obrazów
# image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# # Zmienne do wykresu mAP i czasu detekcji
# map_scores = []
# detection_times = []

# for idx, img_path in enumerate(image_paths):
#     print(f"Przetwarzanie obrazu {img_path}...")

#     # Przetwarzanie obrazu
#     ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=1024)

#     if use_cuda:
#         x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
#     else:
#         x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

#     x = x.float().permute(0, 3, 1, 2)

#     # Detekcja obiektów
#     start_time = time.time()  # Czas rozpoczęcia detekcji
#     with torch.no_grad():
#         features, regression, classification, anchors = model(x)

#         regressBoxes = BBoxTransform()
#         clipBoxes = ClipBoxes()

#         out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)
#         out = invert_affine(framed_metas, out)
#     end_time = time.time()  # Czas zakończenia detekcji

#     # Obliczanie czasu detekcji
#     detection_time = (end_time - start_time) * 1000  # Czas w milisekundach
#     detection_times.append(detection_time)

#     # Rysowanie detekcji na obrazie
#     detected_image = ori_imgs[0].copy()

#     for j in range(len(out[0]['rois'])):
#         x1, y1, x2, y2 = out[0]['rois'][j].astype(int)
#         obj = obj_list[out[0]['class_ids'][j]]
#         score = float(out[0]['scores'][j])
#         plot_one_box(detected_image, [x1, y1, x2, y2], label=obj, score=score, color=(0, 255, 0))

#     # Zapisz przetworzony obraz
#     output_img_path = os.path.join(output_dir, f'detected_{idx+1}.jpg')
#     cv2.imwrite(output_img_path, detected_image)
#     print(f"Zapisano obraz: {output_img_path}")

#     # Obliczanie mAP (tutaj można zaimplementować odpowiednią funkcję obliczania mAP)
#     map_scores.append(np.mean([score for score in out[0]['scores']]))  # Możesz dostosować tę metodę do własnych potrzeb

# # Tworzenie wykresu mAP
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(map_scores) + 1), map_scores, marker='o', linestyle='-', color='b')
# plt.title("mAP na obrazach")
# plt.xlabel("Numer obrazu")
# plt.ylabel("Współczynnik mAP")
# plt.xticks(range(1, len(map_scores) + 1))
# plt.grid(True)

# # Dodanie poziomych linii oznaczających maksymalne i minimalne wartości mAP
# min_map = np.min(map_scores)
# max_map = np.max(map_scores)
# plt.axhline(min_map, color='g', linestyle='--', label=f'Minimalny mAP: {min_map:.4f}')
# plt.axhline(max_map, color='r', linestyle='--', label=f'Maksymalny mAP: {max_map:.4f}')

# # Zapisz wykres mAP
# plot_path = os.path.join(plot_dir, 'mAP_plot.png')
# plt.legend(loc='lower right')  # Ustawienie legendy w prawym dolnym rogu
# plt.savefig(plot_path)
# plt.close()
# print(f"Zapisano wykres: {plot_path}")

# # Tworzenie wykresu czasu detekcji (pomijając pierwsze dwa obrazy)
# plt.figure(figsize=(10, 6))

# # Sprawdzamy, ile jest obrazów w detection_times
# total_images = len(detection_times)

# # Jeśli mamy 22 obrazy, to na wykresie zaczniemy od obrazu 3
# x_values_time = range(3, total_images + 1)  # Oś X - numery obrazów, zaczynając od 3

# # Pomijamy pierwsze dwa elementy w detection_times
# plt.plot(x_values_time, detection_times[2:], marker='o', linestyle='-', color='r', label='Czas detekcji')

# # Dodanie tytułu, etykiet osi i legendy
# plt.title("Czas detekcji na obrazach")
# plt.xlabel("Numer obrazu")
# plt.ylabel("Czas detekcji (ms)")
# plt.xticks(x_values_time)

# # Dodanie poziomych linii oznaczających maksymalne i minimalne wartości
# min_time = np.min(detection_times[2:])
# max_time = np.max(detection_times[2:])
# plt.axhline(min_time, color='g', linestyle='--', label=f'Minimalny czas: {min_time:.2f} ms')
# plt.axhline(max_time, color='r', linestyle='--', label=f'Maksymalny czas: {max_time:.2f} ms')

# # Zapisz wykres czasu detekcji
# plot_path_time = os.path.join(plot_dir, 'Detection_time_plot.png')
# plt.legend(loc='lower right')  # Ustawienie legendy w prawym dolnym rogu
# plt.grid(True)
# plt.savefig(plot_path_time)
# plt.close()
# print(f"Zapisano wykres czasu detekcji: {plot_path_time}")

# import os
# import time
# import torch
# from torch.backends import cudnn
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from backbone import EfficientDetBackbone
# from efficientdet.utils import BBoxTransform, ClipBoxes
# from utils.utils import preprocess, invert_affine, postprocess, plot_one_box

# # --- Funkcja do usuwania outlierów ---
# def remove_outliers(data):
#     data = np.array(data)
#     if len(data) < 4:
#         return data.tolist()
#     q1 = np.percentile(data, 25)
#     q3 = np.percentile(data, 75)
#     iqr = q3 - q1
#     lower_bound = q1 - 3.5 * iqr
#     upper_bound = q3 + 3.5 * iqr
#     filtered = [x for x in data if lower_bound <= x <= upper_bound]
#     return filtered

# # --- Ustawienia ---
# compound_coef = 2
# weights_path = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/logs/cattle_coco/efficientdet-d2_49_8500.pth'
# input_dir = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/cattle_test/'
# output_dir = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/DETEC_PICS/'
# plot_dir = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/GRAPHS_EFFDET/'
# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(plot_dir, exist_ok=True)

# threshold = 0.2
# iou_threshold = 0.2
# use_cuda = True
# num_classes = 1
# obj_list = ['cattle']

# # --- Ładowanie modelu ---
# model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=num_classes)
# model.load_state_dict(torch.load(weights_path, map_location='cpu'))
# model.requires_grad_(False)
# model.eval()
# if use_cuda:
#     model = model.cuda()

# # --- Przetwarzanie obrazów ---
# image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
# map_scores = []
# detection_times = []

# for idx, img_path in enumerate(image_paths):
#     print(f"Przetwarzanie obrazu {img_path}...")

#     ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=1024)

#     if use_cuda:
#         x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
#     else:
#         x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

#     x = x.float().permute(0, 3, 1, 2)

#     start_time = time.time()
#     with torch.no_grad():
#         features, regression, classification, anchors = model(x)
#         regressBoxes = BBoxTransform()
#         clipBoxes = ClipBoxes()
#         out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)
#         out = invert_affine(framed_metas, out)
#     end_time = time.time()

#     detection_time = (end_time - start_time) * 1000
#     detection_times.append(detection_time)

#     detected_image = ori_imgs[0].copy()
#     for j in range(len(out[0]['rois'])):
#         x1, y1, x2, y2 = out[0]['rois'][j].astype(int)
#         obj = obj_list[out[0]['class_ids'][j]]
#         score = float(out[0]['scores'][j])
#         plot_one_box(detected_image, [x1, y1, x2, y2], label=obj, score=score, color=(0, 255, 0))

#     output_img_path = os.path.join(output_dir, f'detected_{idx+1}.jpg')
#     cv2.imwrite(output_img_path, detected_image)
#     print(f"Zapisano obraz: {output_img_path}")

#     if len(out[0]['scores']) > 0:
#         map_scores.append(np.mean([score for score in out[0]['scores']]))

# # --- Wykres mAP bez outlierów ---
# filtered_map_scores = remove_outliers(map_scores)

# # Jeśli wszystko zostało odfiltrowane — użyj oryginalnych
# if not filtered_map_scores:
#     print("Wszystkie wartości mAP uznane za outliery – używam oryginalnych danych.")
#     filtered_map_scores = map_scores
#     x_values_map = list(range(1, len(map_scores) + 1))
# else:
#     x_values_map = [i + 1 for i, val in enumerate(map_scores) if val in filtered_map_scores]

# if filtered_map_scores:
#     y_values_map = filtered_map_scores
#     min_map = np.min(y_values_map)
#     max_map = np.max(y_values_map)

#     plt.figure(figsize=(10, 6))
#     plt.plot(x_values_map, y_values_map, marker='o', linestyle='-', color='b', label='mAP (bez outliers)')
#     plt.title("mAP na obrazach")
#     plt.xlabel("Numer obrazu")
#     plt.ylabel("Współczynnik mAP")
#     plt.xticks(x_values_map)
#     plt.grid(True)

#     if min_map == max_map:
#         plt.axhline(min_map, color='orange', linestyle='--', label=f'mAP: {min_map:.4f}')
#         plt.ylim(min_map - 0.01, max_map + 0.01)
#     else:
#         plt.axhline(min_map, color='g', linestyle='--', label=f'Min mAP: {min_map:.4f}')
#         plt.axhline(max_map, color='r', linestyle='--', label=f'Max mAP: {max_map:.4f}')

#     plt.legend(loc='lower right')
#     plot_path = os.path.join(plot_dir, 'mAP_plot.png')
#     plt.savefig(plot_path)
#     plt.close()
#     print(f"Zapisano wykres: {plot_path}")
# else:
#     print("Brak danych do wykresu mAP.")

# # --- Wykres czasu detekcji bez outlierów (pomijamy pierwsze dwa obrazy) ---
# filtered_detection_times = remove_outliers(detection_times[2:])

# if filtered_detection_times:
#     x_values_time = [i + 3 for i, val in enumerate(detection_times[2:]) if val in filtered_detection_times]
#     y_values_time = filtered_detection_times
#     min_time = np.min(y_values_time)
#     max_time = np.max(y_values_time)

#     plt.figure(figsize=(10, 6))
#     plt.plot(x_values_time, y_values_time, marker='o', linestyle='-', color='r', label='Czas detekcji (bez outliers)')
#     plt.title("Czas detekcji na obrazach")
#     plt.xlabel("Numer obrazu")
#     plt.ylabel("Czas detekcji (ms)")
#     plt.xticks(x_values_time)
#     plt.grid(True)

#     if min_time == max_time:
#         plt.axhline(min_time, color='orange', linestyle='--', label=f'Czas detekcji: {min_time:.2f} ms')
#         plt.ylim(min_time - 1, max_time + 1)
#     else:
#         plt.axhline(min_time, color='g', linestyle='--', label=f'Min czas: {min_time:.2f} ms')
#         plt.axhline(max_time, color='r', linestyle='--', label=f'Max czas: {max_time:.2f} ms')

#     plt.legend(loc='lower right')
#     plot_path_time = os.path.join(plot_dir, 'Detection_time_plot.png')
#     plt.savefig(plot_path_time)
#     plt.close()
#     print(f"Zapisano wykres czasu detekcji: {plot_path_time}")
# else:
#     print("Brak danych do wykresu czasu detekcji po odfiltrowaniu outliers.")
# ####################################################################
# import os
# import time
# import torch
# from torch.backends import cudnn
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from backbone import EfficientDetBackbone
# from efficientdet.utils import BBoxTransform, ClipBoxes
# from utils.utils import preprocess, invert_affine, postprocess, plot_one_box

# # --- Funkcja do usuwania outlierów ---
# def remove_outliers(data):
#     data = np.array(data)
#     if len(data) < 4:
#         return data.tolist()
#     q1 = np.percentile(data, 25)
#     q3 = np.percentile(data, 75)
#     iqr = q3 - q1
#     lower_bound = q1 - 3.5 * iqr
#     upper_bound = q3 + 3.5 * iqr
#     filtered = [x for x in data if lower_bound <= x <= upper_bound]
#     return filtered

# # --- Ustawienia ---
# compound_coef = 4
# weights_path = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/TRAINED_EFFDET/CATTLE/cattle_50ep_d4/efficientdet-d4_49_34500.pth'
# input_dir = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/cattle_test/'
# output_dir = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/DETEC_PICS/'
# plot_dir = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/GRAPHS_EFFDET/'
# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(plot_dir, exist_ok=True)

# threshold = 0.2
# iou_threshold = 0.2
# use_cuda = True
# num_classes = 1
# obj_list = ['cattle']

# # --- Ładowanie modelu ---
# model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=num_classes)
# model.load_state_dict(torch.load(weights_path, map_location='cpu'))
# model.requires_grad_(False)
# model.eval()
# if use_cuda:
#     model = model.cuda()

# # --- Przetwarzanie obrazów ---
# image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
# map_scores = []
# detection_times = []

# for idx, img_path in enumerate(image_paths):
#     print(f"Przetwarzanie obrazu {img_path}...")

#     ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=1024)

#     if use_cuda:
#         x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
#     else:
#         x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

#     x = x.float().permute(0, 3, 1, 2)

#     start_time = time.time()
#     with torch.no_grad():
#         features, regression, classification, anchors = model(x)
#         regressBoxes = BBoxTransform()
#         clipBoxes = ClipBoxes()
#         out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)
#         out = invert_affine(framed_metas, out)
#     end_time = time.time()

#     detection_time = (end_time - start_time) * 1000
#     detection_times.append(detection_time)

#     detected_image = ori_imgs[0].copy()
#     for j in range(len(out[0]['rois'])):
#         x1, y1, x2, y2 = out[0]['rois'][j].astype(int)
#         obj = obj_list[out[0]['class_ids'][j]]
#         score = float(out[0]['scores'][j])
#         plot_one_box(detected_image, [x1, y1, x2, y2], label=obj, score=score, color=(0, 255, 0))

#     output_img_path = os.path.join(output_dir, f'detected_{idx+1}.jpg')
#     cv2.imwrite(output_img_path, detected_image)
#     print(f"Zapisano obraz: {output_img_path}")

#     if len(out[0]['scores']) > 0:
#         map_scores.append(np.mean([score for score in out[0]['scores']]))

# # --- Wykres mAP bez outlierów ---
# filtered_map_scores = remove_outliers(map_scores)

# if not filtered_map_scores:
#     print("Wszystkie wartości mAP uznane za outliery – używam oryginalnych danych.")
#     filtered_map_scores = map_scores
#     x_values_map = list(range(1, len(map_scores) + 1))
# else:
#     x_values_map = [i + 1 for i, val in enumerate(map_scores) if val in filtered_map_scores]

# if filtered_map_scores:
#     y_values_map = filtered_map_scores
#     min_map = np.min(y_values_map)
#     max_map = np.max(y_values_map)

#     plt.figure(figsize=(10, 6))
#     plt.plot(x_values_map, y_values_map, marker='o', linestyle='-', color='b', label='mAP')
#     plt.title(f"mAP na obrazach efficientdet {compound_coef} dla klasy cattle")
#     plt.xlabel("Numer obrazu")
#     plt.ylabel("Współczynnik mAP")
#     plt.xticks(x_values_map)
#     plt.grid(True)

#     if min_map == max_map:
#         plt.axhline(min_map, color='orange', linestyle='--', label=f'mAP: {min_map:.4f}')
#         plt.ylim(min_map - 0.01, max_map + 0.01)
#     else:
#         plt.axhline(min_map, color='g', linestyle='--', label=f'Min mAP: {min_map:.4f}')
#         plt.axhline(max_map, color='r', linestyle='--', label=f'Max mAP: {max_map:.4f}')

#     plt.legend(loc='lower right')
#     plot_path = os.path.join(plot_dir, 'mAP_plot.png')
#     plt.savefig(plot_path)
#     plt.close()
#     print(f"Zapisano wykres: {plot_path}")
# else:
#     print("Brak danych do wykresu mAP.")

# # --- Wykres czasu detekcji bez outlierów (pomijamy pierwsze dwa obrazy) ---
# filtered_detection_times = remove_outliers(detection_times[2:])

# if filtered_detection_times:
#     x_values_time = [i + 3 for i, val in enumerate(detection_times[2:]) if val in filtered_detection_times]
#     y_values_time = filtered_detection_times
#     min_time = np.min(y_values_time)
#     max_time = np.max(y_values_time)

#     plt.figure(figsize=(10, 6))
#     plt.plot(x_values_time, y_values_time, marker='o', linestyle='-', color='r', label='Czas detekcji')
#     plt.title(f"Czas detekcji na obrazach efficientdet {compound_coef} dla klasy cattle")
#     plt.xlabel("Numer obrazu")
#     plt.ylabel("Czas detekcji (ms)")
#     plt.xticks(x_values_time)
#     plt.grid(True)

#     if min_time == max_time:
#         plt.axhline(min_time, color='orange', linestyle='--', label=f'Czas detekcji: {min_time:.2f} ms')
#         plt.ylim(min_time - 1, max_time + 1)
#     else:
#         plt.axhline(min_time, color='g', linestyle='--', label=f'Min czas: {min_time:.2f} ms')
#         plt.axhline(max_time, color='r', linestyle='--', label=f'Max czas: {max_time:.2f} ms')

#     plt.legend(loc='lower right')
#     plot_path_time = os.path.join(plot_dir, 'Detection_time_plot.png')
#     plt.savefig(plot_path_time)
#     plt.close()
#     print(f"Zapisano wykres czasu detekcji: {plot_path_time}")
# else:
#     print("Brak danych do wykresu czasu detekcji po odfiltrowaniu outliers.")

# # --- Wyświetlenie średniego mAP ---
# if filtered_map_scores:
#     avg_map = np.mean(filtered_map_scores)
#     print(f"\nŚrednia wartość mAP (po usunięciu outliers): {avg_map:.4f}")
# else:
#     print("\nBrak wystarczających danych do obliczenia średniego mAP.")

# if map_scores:
#     raw_avg_map = np.mean(map_scores)
#     print(f"Średnia wartość mAP (oryginalna, bez filtrowania): {raw_avg_map:.4f}")

# # --- Wyświetlenie średniego opóźnienia ---
# if detection_times:
#     avg_detection_time = np.mean(detection_times)
#     print(f"\nŚredni czas detekcji (wszystkie obrazy): {avg_detection_time:.2f} ms")

# # --- Wyświetlenie liczby przetworzonych zdjęć ---
# total_images = len(image_paths)
# print(f"Łączna liczba przetworzonych zdjęć: {total_images}")

# # --- FPS i wartość niestandardowa ---
# if detection_times and total_images > 0:
#     custom_value = avg_detection_time / total_images  # Twój wzór

#     print(f"Wartość (ms/img): {custom_value:.4f}")

import os
import time
import torch
from torch.backends import cudnn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, plot_one_box

def remove_outliers(data):
    data = np.array(data)
    if len(data) < 4:
        return data.tolist()
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 3.5 * iqr
    upper_bound = q3 + 3.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]

# === ŚCIEŻKI I USTAWIENIA ===
compound_coef = 0
weights_path = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/TRAINED_EFFDET/CATTLE/cattle_50ep_d0/efficientdet-d0_49_8650.pth'
input_dir = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/cattle_test/'
output_dir = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/DETEC_PICS/'
plot_dir = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/GRAPHS_EFFDET/'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

threshold = 0.2
iou_threshold = 0.2
use_cuda = True
num_classes = 1
obj_list = ['cattle']

# === MODELOWANIE ===
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=num_classes)
model.load_state_dict(torch.load(weights_path, map_location='cpu'))
model.requires_grad_(False)
model.eval()
if use_cuda:
    model = model.cuda()

# === DETEKCJA OBRAZÓW ===
image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
map_scores = []
detection_times = []

for idx, img_path in enumerate(image_paths):
    print(f"Przetwarzanie obrazu {img_path}...")

    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=1024)
    x = torch.stack([torch.from_numpy(fi).cuda() if use_cuda else torch.from_numpy(fi) for fi in framed_imgs], 0)
    x = x.float().permute(0, 3, 1, 2)

    start_time = time.time()
    with torch.no_grad():
        features, regression, classification, anchors = model(x)
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)
        out = invert_affine(framed_metas, out)
    end_time = time.time()

    detection_time = (end_time - start_time) * 1000
    detection_times.append(detection_time)

    detected_image = ori_imgs[0].copy()
    for j in range(len(out[0]['rois'])):
        x1, y1, x2, y2 = out[0]['rois'][j].astype(int)
        obj = obj_list[out[0]['class_ids'][j]]
        score = float(out[0]['scores'][j])
        plot_one_box(detected_image, [x1, y1, x2, y2], label=obj, score=score, color=(0, 255, 0))

    output_img_path = os.path.join(output_dir, f'detected_{idx+1}.jpg')
    cv2.imwrite(output_img_path, detected_image)
    print(f"Zapisano obraz: {output_img_path}")

    if len(out[0]['scores']) > 0:
        map_scores.append(np.mean(out[0]['scores']))
    else:
        map_scores.append(0.0)

# === WYKRES mAP ===
filtered_map_scores = remove_outliers(map_scores)
x_values_map = list(range(1, len(map_scores) + 1)) if not filtered_map_scores else [i + 1 for i, val in enumerate(map_scores) if val in filtered_map_scores]

if filtered_map_scores:
    y_values_map = filtered_map_scores
    min_map, max_map = np.min(y_values_map), np.max(y_values_map)

    plt.figure(figsize=(10, 6))
    plt.plot(x_values_map, y_values_map, marker='o', color='b', label='mAP')
    plt.title(f"mAP dla EfficientDet-D{compound_coef} - klasa: cattle")
    plt.xlabel("Numer obrazu")
    plt.ylabel("mAP")
    plt.grid(True)
    if min_map == max_map:
        plt.axhline(min_map, color='orange', linestyle='--', label=f'mAP: {min_map:.4f}')
        plt.ylim(min_map - 0.01, max_map + 0.01)
    else:
        plt.axhline(min_map, color='g', linestyle='--', label=f'Min mAP: {min_map:.4f}')
        plt.axhline(max_map, color='r', linestyle='--', label=f'Max mAP: {max_map:.4f}')
    plt.legend()
    plot_path = os.path.join(plot_dir, 'mAP_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Zapisano wykres: {plot_path}")

# === WYKRES CZASU DETEKCJI ===
filtered_detection_times = remove_outliers(detection_times[2:])
x_values_time = [i + 3 for i, val in enumerate(detection_times[2:]) if val in filtered_detection_times]

if filtered_detection_times:
    y_values_time = filtered_detection_times
    min_time, max_time = np.min(y_values_time), np.max(y_values_time)

    plt.figure(figsize=(10, 6))
    plt.plot(x_values_time, y_values_time, marker='o', color='r', label='Czas detekcji (ms)')
    plt.title(f"Czas detekcji dla EfficientDet-D{compound_coef}")
    plt.xlabel("Numer obrazu")
    plt.ylabel("Czas (ms)")
    plt.grid(True)
    if min_time == max_time:
        plt.axhline(min_time, color='orange', linestyle='--', label=f'Czas: {min_time:.2f} ms')
        plt.ylim(min_time - 1, max_time + 1)
    else:
        plt.axhline(min_time, color='g', linestyle='--', label=f'Min czas: {min_time:.2f} ms')
        plt.axhline(max_time, color='r', linestyle='--', label=f'Max czas: {max_time:.2f} ms')
    plt.legend()
    plot_path_time = os.path.join(plot_dir, 'Detection_time_plot.png')
    plt.savefig(plot_path_time)
    plt.close()
    print(f"Zapisano wykres czasu detekcji: {plot_path_time}")

# === ŚREDNIE WARTOŚCI I STATYSTYKI ===
if filtered_map_scores:
    avg_map = np.mean(filtered_map_scores)
    print(f"\nŚrednia wartość mAP (po filtracji): {avg_map:.4f}")
if map_scores:
    print(f"Średnia wartość mAP (oryginalna): {np.mean(map_scores):.4f}")
if detection_times:
    avg_detection_time = np.mean(detection_times)
    print(f"Średni czas detekcji: {avg_detection_time:.2f} ms")

total_images = len(image_paths)
print(f"Łączna liczba zdjęć: {total_images}")
if detection_times:
    custom_value = avg_detection_time / total_images
    print(f"Wartość (ms/img): {custom_value:.4f}")

# === ZAPIS CSV W FORMACIE: mAP, opóźnienie, numer zdjęcia ===
csv_output_path = os.path.join(plot_dir, "map_czas_zdjecie.csv")
with open(csv_output_path, "w") as f:
    f.write("mAP,opoznienie_ms,nr_zdjecia\n")
    for i, (map_val, delay) in enumerate(zip(map_scores, detection_times)):
        f.write(f"{map_val:.4f},{delay:.2f},{i+1}\n")
print(f"✅ Dane zapisane w CSV: {csv_output_path}")
