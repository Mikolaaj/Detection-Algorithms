# import os
# import time
# import torch
# from torch.backends import cudnn
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from backbone import EfficientDetBackbone
# from efficientdet.utils import BBoxTransform, ClipBoxes
# from utils.utils import preprocess_video, invert_affine, postprocess, plot_one_box

# # === ŚCIEŻKI ===
# video_src = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/cattle_side_wide_view.mp4'
# output_video_path = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/DETEC_PICS/output_video.mp4'
# plot_dir = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/GRAPHS_EFFDET/'
# os.makedirs(plot_dir, exist_ok=True)

# # === PARAMETRY ===
# compound_coef = 2
# torch_model_path = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/TRAINED_EFFDET/cattle_50ep_d2/efficientdet-d2_49_8500.pth'
# threshold = 0.2
# iou_threshold = 0.2
# use_cuda = True
# use_float16 = False
# obj_list = ['bottle']
# input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
# input_size = input_sizes[compound_coef]

# # === CUDA i model ===
# cudnn.fastest = True
# cudnn.benchmark = True
# model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
# model.load_state_dict(torch.load(torch_model_path, map_location='cuda' if use_cuda else 'cpu'))
# model.requires_grad_(False)
# model.eval()
# if use_cuda:
#     model = model.cuda()
# if use_float16:
#     model = model.half()

# regressBoxes = BBoxTransform()
# clipBoxes = ClipBoxes()

# # === WIDEO: otwarcie wejścia i ustawienie wyjścia ===
# cap = cv2.VideoCapture(video_src)
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# # === LISTY DLA WYKRESÓW ===
# map_scores = []
# detection_times = []

# frame_index = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame_index += 1

#     ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)
#     if use_cuda:
#         x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
#     else:
#         x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
#     x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

#     # === DETEKCJA ===
#     start_time = time.time()
#     with torch.no_grad():
#         features, regression, classification, anchors = model(x)
#         detections = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)
#         detections = invert_affine(framed_metas, detections)
#     end_time = time.time()
#     detection_time = (end_time - start_time) * 1000
#     detection_times.append(detection_time)

#     # === mAP DLA TEJ KLATKI ===
#     if detections and len(detections[0]['scores']) > 0:
#         map_scores.append(np.mean(detections[0]['scores']))
#     else:
#         map_scores.append(0.0)

#     # === RYSOWANIE DETEKCJI ===
#     detected_image = ori_imgs[0].copy()
#     for j in range(len(detections[0]['rois'])):
#         x1, y1, x2, y2 = detections[0]['rois'][j].astype(int)
#         score = float(detections[0]['scores'][j])
#         obj = obj_list[detections[0]['class_ids'][j]]
#         plot_one_box(detected_image, [x1, y1, x2, y2], label=obj, score=score, color=(255, 255, 0))

#     # === SKALOWANIE DO WYMIARÓW ORYGINALNEGO WIDEO ===
#     scale = min(frame_width / detected_image.shape[1], frame_height / detected_image.shape[0])
#     new_width, new_height = int(detected_image.shape[1] * scale), int(detected_image.shape[0] * scale)
#     detected_image = cv2.resize(detected_image, (new_width, new_height))
#     top_left_x = (frame_width - new_width) // 2
#     top_left_y = (frame_height - new_height) // 2
#     final_frame = np.zeros_like(frame)
#     final_frame[top_left_y:top_left_y+new_height, top_left_x:top_left_x+new_width] = detected_image

#     video_writer.write(final_frame)
#     print(f"Przetworzono klatkę {frame_index} | mAP: {map_scores[-1]:.4f} | Czas: {detection_time:.2f} ms")

# # === ZAMKNIĘCIE ===
# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()

# # === FILTROWANIE SKRAJNYCH WARTOŚCI ===
# def filter_extreme_values(values, threshold=0.1):
#     mean_value = np.mean(values)
#     std_dev = np.std(values)
#     return [value for value in values if abs(value - mean_value) <= threshold * std_dev]

# # === WYKRES mAP ===
# plt.figure(figsize=(10, 6))
# x_vals_map = list(range(1, len(map_scores) + 1))
# filtered_map_scores = filter_extreme_values(map_scores)  # Filtrujemy wartości

# # Oś X zaczyna się od 1, 2, 3, ...
# plt.plot(x_vals_map[:len(filtered_map_scores)], filtered_map_scores, marker='o', linestyle='-', color='b', label='Średni mAP')
# plt.title("mAP na klatkach wideo")
# plt.xlabel("Numer klatki")
# plt.ylabel("Współczynnik mAP")
# plt.grid(True)

# min_map = np.min(filtered_map_scores)
# max_map = np.max(filtered_map_scores)
# plt.axhline(min_map, color='g', linestyle='--', label=f'Min mAP: {min_map:.4f}')
# plt.axhline(max_map, color='r', linestyle='--', label=f'Max mAP: {max_map:.4f}')
# plt.legend(loc='lower right')  # PRAWY DOLNY RÓG
# plt.xticks(x_vals_map[:len(filtered_map_scores)])
# plt.savefig(os.path.join(plot_dir, 'mAP_plot.png'))
# plt.close()
# print(f"Zapisano wykres mAP: {os.path.join(plot_dir, 'mAP_plot.png')}")

# # === WYKRES CZASU DETEKCJI ===
# plt.figure(figsize=(10, 6))
# x_vals_time = list(range(3, len(detection_times) + 1))
# filtered_detection_times = filter_extreme_values(detection_times)  # Filtrujemy wartości

# plt.plot(x_vals_time[:len(filtered_detection_times)], filtered_detection_times, marker='o', linestyle='-', color='r', label='Czas detekcji (ms)')
# min_time = np.min(filtered_detection_times)
# max_time = np.max(filtered_detection_times)
# plt.axhline(min_time, color='g', linestyle='--', label=f'Min czas: {min_time:.2f} ms')
# plt.axhline(max_time, color='b', linestyle='--', label=f'Max czas: {max_time:.2f} ms')
# plt.title("Czas detekcji na klatkach")
# plt.xlabel("Numer klatki")
# plt.ylabel("Czas detekcji (ms)")
# plt.legend(loc='lower right')  # PRAWY DOLNY RÓG
# plt.grid(True)
# plt.savefig(os.path.join(plot_dir, 'Detection_time_plot.png'))
# plt.close()
# print(f"Zapisano wykres czasu detekcji: {os.path.join(plot_dir, 'Detection_time_plot.png')}")

# import os
# import time
# import torch
# from torch.backends import cudnn
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from backbone import EfficientDetBackbone
# from efficientdet.utils import BBoxTransform, ClipBoxes
# from utils.utils import preprocess_video, invert_affine, postprocess, plot_one_box

# # === ŚCIEŻKI ===
# #video_src = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/water_in_factory.mp4'
# video_src = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/cattle_side_wide_view.mp4'

# output_video_path = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/DETEC_PICS/output_video.mp4'
# plot_dir = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/GRAPHS_EFFDET/'
# os.makedirs(plot_dir, exist_ok=True)

# # === PARAMETRY ===
# compound_coef = 2
# #torch_model_path = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/logs/bottle_coco/efficientdet-d2_30_1526.pth'
# torch_model_path = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/TRAINED_EFFDET/cattle_50ep_d2/efficientdet-d2_49_8500.pth'
# threshold = 0.2
# iou_threshold = 0.2
# use_cuda = True
# use_float16 = False
# obj_list = ['bottle']
# input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
# input_size = input_sizes[compound_coef]

# # === CUDA i model ===
# cudnn.fastest = True
# cudnn.benchmark = True
# model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
# model.load_state_dict(torch.load(torch_model_path, map_location='cuda' if use_cuda else 'cpu'))
# model.requires_grad_(False)
# model.eval()
# if use_cuda:
#     model = model.cuda()
# if use_float16:
#     model = model.half()

# regressBoxes = BBoxTransform()
# clipBoxes = ClipBoxes()

# # === WIDEO: otwarcie wejścia i ustawienie wyjścia ===
# cap = cv2.VideoCapture(video_src)
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# # === LISTY DLA WYKRESÓW ===
# map_scores = []
# detection_times = []

# frame_index = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame_index += 1

#     ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)
#     if use_cuda:
#         x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
#     else:
#         x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
#     x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

#     # === DETEKCJA ===
#     start_time = time.time()
#     with torch.no_grad():
#         features, regression, classification, anchors = model(x)
#         detections = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)
#         detections = invert_affine(framed_metas, detections)
#     end_time = time.time()
#     detection_time = (end_time - start_time) * 1000
#     detection_times.append(detection_time)

#     # === mAP DLA TEJ KLATKI ===
#     if detections and len(detections[0]['scores']) > 0:
#         map_scores.append(np.mean(detections[0]['scores']))
#     else:
#         map_scores.append(0.0)

#     # === RYSOWANIE DETEKCJI ===
#     detected_image = ori_imgs[0].copy()
#     for j in range(len(detections[0]['rois'])):
#         x1, y1, x2, y2 = detections[0]['rois'][j].astype(int)
#         score = float(detections[0]['scores'][j])
#         obj = obj_list[detections[0]['class_ids'][j]]
#         plot_one_box(detected_image, [x1, y1, x2, y2], label=obj, score=score, color=(255, 255, 0))

#     # === SKALOWANIE DO WYMIARÓW ORYGINALNEGO WIDEO ===
#     scale = min(frame_width / detected_image.shape[1], frame_height / detected_image.shape[0])
#     new_width, new_height = int(detected_image.shape[1] * scale), int(detected_image.shape[0] * scale)
#     detected_image = cv2.resize(detected_image, (new_width, new_height))
#     top_left_x = (frame_width - new_width) // 2
#     top_left_y = (frame_height - new_height) // 2
#     final_frame = np.zeros_like(frame)
#     final_frame[top_left_y:top_left_y+new_height, top_left_x:top_left_x+new_width] = detected_image

#     video_writer.write(final_frame)
#     print(f"Przetworzono klatkę {frame_index} | mAP: {map_scores[-1]:.4f} | Czas: {detection_time:.2f} ms")

# # === ZAMKNIĘCIE ===
# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()

# # === FILTROWANIE SKRAJNYCH WARTOŚCI ===
# def filter_extreme_values(values, threshold=1.5):
#     mean_value = np.mean(values)
#     std_dev = np.std(values)
#     return [value for value in values if abs(value - mean_value) <= threshold * std_dev]

# # === WYKRES mAP ===
# # === WYKRES mAP ===
# plt.figure(figsize=(10, 6))
# x_vals_map = list(range(1, len(map_scores) + 1))
# filtered_map_scores = filter_extreme_values(map_scores)

# if filtered_map_scores:
#     plt.plot(x_vals_map[:len(filtered_map_scores)], filtered_map_scores, marker='o', linestyle='-', color='b', label='Średni mAP')
#     min_map = np.min(filtered_map_scores)
#     max_map = np.max(filtered_map_scores)
#     plt.axhline(min_map, color='g', linestyle='--', label=f'Min mAP: {min_map:.4f}')
#     plt.axhline(max_map, color='r', linestyle='--', label=f'Max mAP: {max_map:.4f}')
#     plt.legend(loc='lower right')
#     plt.xticks(x_vals_map[:len(filtered_map_scores)])
#     plt.title("mAP na klatkach wideo")
#     plt.xlabel("Numer klatki")
#     plt.ylabel("Współczynnik mAP")
#     plt.grid(True)
#     plt.savefig(os.path.join(plot_dir, 'mAP_plot.png'))
#     print(f"Zapisano wykres mAP: {os.path.join(plot_dir, 'mAP_plot.png')}")
# else:
#     print("Brak danych do stworzenia wykresu mAP – wszystkie wartości zostały odfiltrowane.")
# plt.close()


# # === WYKRES CZASU DETEKCJI ===
# plt.figure(figsize=(10, 6))
# x_vals_time = list(range(3, len(detection_times) + 1))
# filtered_detection_times = filter_extreme_values(detection_times)

# plt.plot(x_vals_time[:len(filtered_detection_times)], filtered_detection_times, marker='o', linestyle='-', color='r', label='Czas detekcji (ms)')
# min_time = np.min(filtered_detection_times)
# max_time = np.max(filtered_detection_times)
# plt.axhline(min_time, color='g', linestyle='--', label=f'Min czas: {min_time:.2f} ms')
# plt.axhline(max_time, color='b', linestyle='--', label=f'Max czas: {max_time:.2f} ms')
# plt.title("Czas detekcji na klatkach")
# plt.xlabel("Numer klatki")
# plt.ylabel("Czas detekcji (ms)")
# plt.legend(loc='lower right')
# plt.grid(True)
# plt.savefig(os.path.join(plot_dir, 'Detection_time_plot.png'))
# plt.close()
# print(f"Zapisano wykres czasu detekcji: {os.path.join(plot_dir, 'Detection_time_plot.png')}")

# # === PODSUMOWANIE ===
# print("\n--- PODSUMOWANIE ---")

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

#     # --- FPS ---
#     avg_fps = 1000.0 / avg_detection_time
#     print(f"Średnie FPS (na podstawie czasu detekcji): {avg_fps:.2f}")

# # --- Wyświetlenie liczby przetworzonych zdjęć ---
# total_images = frame_index
# print(f"\nŁączna liczba przetworzonych klatek: {total_images}")

# # --- Wartość niestandardowa ---
# if detection_times and total_images > 0:
#     custom_value = avg_detection_time / total_images
#     print(f"Wartość (ms/img): {custom_value:.4f}")
###########################################################
# import os
# import time
# import torch
# from torch.backends import cudnn
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from backbone import EfficientDetBackbone
# from efficientdet.utils import BBoxTransform, ClipBoxes
# from utils.utils import preprocess_video, invert_affine, postprocess, plot_one_box

# # === ŚCIEŻKI ===
# video_src = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/cattle_cut.mp4'
# output_video_path = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/DETEC_PICS/output_video.mp4'
# plot_dir = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/GRAPHS_EFFDET/'
# os.makedirs(plot_dir, exist_ok=True)

# # === PARAMETRY ===
# compound_coef = 4
# torch_model_path = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/TRAINED_EFFDET/CATTLE/cattle_50ep_d4/efficientdet-d4_49_34500.pth'
# threshold = 0.2
# iou_threshold = 0.2
# use_cuda = True
# use_float16 = False
# obj_list = ['cattle']
# input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
# input_size = input_sizes[compound_coef]

# # === CUDA i model ===
# cudnn.fastest = True
# cudnn.benchmark = True
# model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
# model.load_state_dict(torch.load(torch_model_path, map_location='cuda' if use_cuda else 'cpu'))
# model.requires_grad_(False)
# model.eval()
# if use_cuda:
#     model = model.cuda()
# if use_float16:
#     model = model.half()

# regressBoxes = BBoxTransform()
# clipBoxes = ClipBoxes()

# # === WIDEO: otwarcie wejścia i ustawienie wyjścia ===
# cap = cv2.VideoCapture(video_src)
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# # === LISTY DLA WYKRESÓW ===
# map_scores = []
# detection_times = []

# frame_index = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame_index += 1

#     ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)
#     if use_cuda:
#         x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
#     else:
#         x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
#     x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

#     # === DETEKCJA ===
#     start_time = time.time()
#     with torch.no_grad():
#         features, regression, classification, anchors = model(x)
#         detections = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)
#         detections = invert_affine(framed_metas, detections)
#     end_time = time.time()
#     detection_time = (end_time - start_time) * 1000
#     detection_times.append(detection_time)

#     # === mAP DLA TEJ KLATKI ===
#     if detections and len(detections[0]['scores']) > 0:
#         map_scores.append(np.mean(detections[0]['scores']))
#     else:
#         map_scores.append(0.0)

#     # === RYSOWANIE DETEKCJI ===
#     detected_image = ori_imgs[0].copy()
#     for j in range(len(detections[0]['rois'])):
#         x1, y1, x2, y2 = detections[0]['rois'][j].astype(int)
#         score = float(detections[0]['scores'][j])
#         obj = obj_list[detections[0]['class_ids'][j]]
#         plot_one_box(detected_image, [x1, y1, x2, y2], label=obj, score=score, color=(255, 255, 0))

#     # === SKALOWANIE DO WYMIARÓW ORYGINALNEGO WIDEO ===
#     scale = min(frame_width / detected_image.shape[1], frame_height / detected_image.shape[0])
#     new_width, new_height = int(detected_image.shape[1] * scale), int(detected_image.shape[0] * scale)
#     detected_image = cv2.resize(detected_image, (new_width, new_height))
#     top_left_x = (frame_width - new_width) // 2
#     top_left_y = (frame_height - new_height) // 2
#     final_frame = np.zeros_like(frame)
#     final_frame[top_left_y:top_left_y+new_height, top_left_x:top_left_x+new_width] = detected_image

#     video_writer.write(final_frame)
#     print(f"Przetworzono klatkę {frame_index} | mAP: {map_scores[-1]:.4f} | Czas: {detection_time:.2f} ms")

# # === ZAMKNIĘCIE ===
# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()

# # === FILTROWANIE SKRAJNYCH WARTOŚCI ===
# def filter_extreme_values(values, threshold=1.0):
#     mean_value = np.mean(values)
#     std_dev = np.std(values)
#     return [value for value in values if abs(value - mean_value) <= threshold * std_dev]

# # === WYKRES mAP (pomija zera) ===
# plt.figure(figsize=(10, 6))
# filtered_map_scores = filter_extreme_values([score for score in map_scores if score > 0])
# x_vals_map = list(range(1, len(filtered_map_scores) + 1))

# if filtered_map_scores:
#     plt.plot(x_vals_map, filtered_map_scores, marker='o', linestyle='-', color='b', label='Średni mAP')
#     min_map = np.min(filtered_map_scores)
#     max_map = np.max(filtered_map_scores)
#     plt.axhline(min_map, color='g', linestyle='--', label=f'Min mAP: {min_map:.4f}')
#     plt.axhline(max_map, color='r', linestyle='--', label=f'Max mAP: {max_map:.4f}')
#     plt.legend(loc='lower right')
#     xticks_map = list(range(1, len(x_vals_map)+1, 25))
#     plt.xticks(xticks_map)
#     plt.title(f"mAP na klatkach wideo efficeintdet {compound_coef} dla klasy cattle")
#     plt.xlabel("Numer klatki")
#     plt.ylabel("Współczynnik mAP")
#     plt.grid(True)
#     plt.savefig(os.path.join(plot_dir, 'mAP_plot.png'))
#     print(f"Zapisano wykres mAP: {os.path.join(plot_dir, 'mAP_plot.png')}")
# else:
#     print("Brak danych do stworzenia wykresu mAP – wszystkie wartości zostały odfiltrowane.")
# plt.close()

# # === WYKRES CZASU DETEKCJI ===
# plt.figure(figsize=(10, 6))

# # Filtrowanie wartości odstających
# filtered_detection_times = filter_extreme_values(detection_times)
# x_vals_time = list(range(1, len(filtered_detection_times) + 1))

# # Oś X co 25 klatek
# xticks_time = list(range(1, len(x_vals_time) + 1, 25))

# # Rysowanie wykresu
# plt.plot(x_vals_time, filtered_detection_times, marker='o', linestyle='-', color='r', label='Czas detekcji (ms)')

# # Linie min/max
# min_time = np.min(filtered_detection_times)
# max_time = np.max(filtered_detection_times)
# plt.axhline(min_time, color='g', linestyle='--', label=f'Min czas: {min_time:.2f} ms')
# plt.axhline(max_time, color='b', linestyle='--', label=f'Max czas: {max_time:.2f} ms')

# # Oś X z krokiem co 25
# plt.xticks(xticks_time)

# # Opisy i zapis
# plt.title(f"Czas detekcji na klatkach efficientdet {compound_coef} dla klasy cattle")
# plt.xlabel("Numer klatki")
# plt.ylabel("Czas detekcji (ms)")
# plt.legend(loc='lower right')
# plt.grid(True)
# plt.savefig(os.path.join(plot_dir, 'Detection_time_plot.png'))
# plt.close()
# print(f"Zapisano wykres czasu detekcji: {os.path.join(plot_dir, 'Detection_time_plot.png')}")

# # === PODSUMOWANIE ===
# print("\n--- PODSUMOWANIE ---")

# # --- Wyświetlenie średniego mAP ---
# if filtered_map_scores:
#     avg_map = np.mean(filtered_map_scores)
#     print(f"\nŚrednia wartość mAP (po usunięciu outliers i zer): {avg_map:.4f}")
# else:
#     print("\nBrak wystarczających danych do obliczenia średniego mAP.")

# if map_scores:
#     raw_avg_map = np.mean(map_scores)
#     print(f"Średnia wartość mAP (oryginalna, bez filtrowania): {raw_avg_map:.4f}")

# # --- Wyświetlenie średniego opóźnienia ---
# if detection_times:
#     avg_detection_time = np.mean(detection_times)
#     print(f"\nŚredni czas detekcji (wszystkie obrazy): {avg_detection_time:.2f} ms")

#     # --- FPS ---
#     avg_fps = 1000.0 / avg_detection_time
#     print(f"Średnie FPS (na podstawie czasu detekcji): {avg_fps:.2f}")

# # --- Wyświetlenie liczby przetworzonych zdjęć ---
# total_images = frame_index
# print(f"\nŁączna liczba przetworzonych klatek: {total_images}")

# # --- Wartość niestandardowa ---
# if detection_times and total_images > 0:
#     custom_value = avg_detection_time / total_images
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
from utils.utils import preprocess_video, invert_affine, postprocess, plot_one_box

# === ŚCIEŻKI ===
video_src = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/cattle_cut.mp4'
output_video_path = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/DETEC_PICS/output_video.mp4'
plot_dir = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/GRAPHS_EFFDET/'
os.makedirs(plot_dir, exist_ok=True)

# === PARAMETRY ===
compound_coef = 0
torch_model_path = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/TRAINED_EFFDET/CATTLE/cattle_50ep_d0/efficientdet-d0_49_8650.pth'
threshold = 0.2
iou_threshold = 0.2
use_cuda = True
use_float16 = False
obj_list = ['cattle']
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef]

# === CUDA i model ===
cudnn.fastest = True
cudnn.benchmark = True
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
model.load_state_dict(torch.load(torch_model_path, map_location='cuda' if use_cuda else 'cpu'))
model.requires_grad_(False)
model.eval()
if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

# === WIDEO: otwarcie wejścia i ustawienie wyjścia ===
cap = cv2.VideoCapture(video_src)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# === LISTY DLA WYKRESÓW ===
map_scores = []
detection_times = []

frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_index += 1

    ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)
    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    # === DETEKCJA ===
    start_time = time.time()
    with torch.no_grad():
        features, regression, classification, anchors = model(x)
        detections = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)
        detections = invert_affine(framed_metas, detections)
    end_time = time.time()
    detection_time = (end_time - start_time) * 1000
    detection_times.append(detection_time)

    # === mAP DLA TEJ KLATKI ===
    if detections and len(detections[0]['scores']) > 0:
        map_scores.append(np.mean(detections[0]['scores']))
    else:
        map_scores.append(0.0)

    # === RYSOWANIE DETEKCJI ===
    detected_image = ori_imgs[0].copy()
    for j in range(len(detections[0]['rois'])):
        x1, y1, x2, y2 = detections[0]['rois'][j].astype(int)
        score = float(detections[0]['scores'][j])
        obj = obj_list[detections[0]['class_ids'][j]]
        plot_one_box(detected_image, [x1, y1, x2, y2], label=obj, score=score, color=(255, 255, 0))

    # === SKALOWANIE DO WYMIARÓW ORYGINALNEGO WIDEO ===
    scale = min(frame_width / detected_image.shape[1], frame_height / detected_image.shape[0])
    new_width, new_height = int(detected_image.shape[1] * scale), int(detected_image.shape[0] * scale)
    detected_image = cv2.resize(detected_image, (new_width, new_height))
    top_left_x = (frame_width - new_width) // 2
    top_left_y = (frame_height - new_height) // 2
    final_frame = np.zeros_like(frame)
    final_frame[top_left_y:top_left_y+new_height, top_left_x:top_left_x+new_width] = detected_image

    video_writer.write(final_frame)
    print(f"Przetworzono klatkę {frame_index} | mAP: {map_scores[-1]:.4f} | Czas: {detection_time:.2f} ms")

# === ZAMKNIĘCIE ===
cap.release()
video_writer.release()
cv2.destroyAllWindows()

# === FILTROWANIE SKRAJNYCH WARTOŚCI ===
def filter_extreme_values(values, threshold=1.0):
    mean_value = np.mean(values)
    std_dev = np.std(values)
    return [value for value in values if abs(value - mean_value) <= threshold * std_dev]

# === WYKRES mAP (pomija zera) ===
plt.figure(figsize=(10, 6))
filtered_map_scores = filter_extreme_values([score for score in map_scores if score > 0])
x_vals_map = list(range(1, len(filtered_map_scores) + 1))

if filtered_map_scores:
    plt.plot(x_vals_map, filtered_map_scores, marker='o', linestyle='-', color='b', label='Średni mAP')
    min_map = np.min(filtered_map_scores)
    max_map = np.max(filtered_map_scores)
    plt.axhline(min_map, color='g', linestyle='--', label=f'Min mAP: {min_map:.4f}')
    plt.axhline(max_map, color='r', linestyle='--', label=f'Max mAP: {max_map:.4f}')
    plt.legend(loc='lower right')
    xticks_map = list(range(1, len(x_vals_map)+1, 25))
    plt.xticks(xticks_map)
    plt.title(f"mAP na klatkach wideo efficeintdet {compound_coef} dla klasy cattle")
    plt.xlabel("Numer klatki")
    plt.ylabel("Współczynnik mAP")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'mAP_plot.png'))
    print(f"Zapisano wykres mAP: {os.path.join(plot_dir, 'mAP_plot.png')}")
else:
    print("Brak danych do stworzenia wykresu mAP – wszystkie wartości zostały odfiltrowane.")
plt.close()

# === WYKRES CZASU DETEKCJI ===
plt.figure(figsize=(10, 6))

# Filtrowanie wartości odstających
filtered_detection_times = filter_extreme_values(detection_times)

# Zaczynamy od 10. klatki (indeks 9), więc wybieramy elementy od indeksu 9
filtered_detection_times_from_10th_frame = filtered_detection_times[9:]

# Oś X co 25 klatek, zaczynając od 10. klatki
x_vals_time_from_10th = list(range(10, len(filtered_detection_times) + 1))

# Rysowanie wykresu
plt.plot(x_vals_time_from_10th, filtered_detection_times_from_10th_frame, marker='o', linestyle='-', color='r', label='Czas detekcji (ms)')

# Linie min/max
min_time = np.min(filtered_detection_times_from_10th_frame)
max_time = np.max(filtered_detection_times_from_10th_frame)
plt.axhline(min_time, color='g', linestyle='--', label=f'Min czas: {min_time:.2f} ms')
plt.axhline(max_time, color='b', linestyle='--', label=f'Max czas: {max_time:.2f} ms')

# Oś X z krokiem co 25
plt.xticks(range(10, len(x_vals_time_from_10th) + 1, 25))

# Opisy i zapis
plt.title(f"Czas detekcji na klatkach efficientdet {compound_coef} dla klasy cattle")
plt.xlabel("Numer klatki")
plt.ylabel("Czas detekcji (ms)")
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(os.path.join(plot_dir, 'Detection_time_plot_from_10th.png'))
plt.close()
print(f"Zapisano wykres czasu detekcji (od 10. klatki): {os.path.join(plot_dir, 'Detection_time_plot_from_10th.png')}")

# === PODSUMOWANIE ===
print("\n--- PODSUMOWANIE ---")

# --- Wyświetlenie średniego mAP ---
if filtered_map_scores:
    avg_map = np.mean(filtered_map_scores)
    print(f"\nŚrednia wartość mAP (po usunięciu outliers i zer): {avg_map:.4f}")
else:
    print("\nBrak wystarczających danych do obliczenia średniego mAP.")

if map_scores:
    raw_avg_map = np.mean(map_scores)
    print(f"Średnia wartość mAP (oryginalna, bez filtrowania): {raw_avg_map:.4f}")

# --- Wyświetlenie średniego opóźnienia ---
if detection_times:
    avg_detection_time = np.mean(detection_times)
    print(f"\nŚredni czas detekcji (wszystkie obrazy): {avg_detection_time:.2f} ms")

    # --- FPS ---
    avg_fps = 1000.0 / avg_detection_time
    print(f"Średnie FPS (na podstawie czasu detekcji): {avg_fps:.2f}")

# --- Wyświetlenie liczby przetworzonych zdjęć ---
total_images = frame_index
print(f"\nŁączna liczba przetworzonych klatek: {total_images}")

# --- Wartość niestandardowa ---
if detection_times and total_images > 0:
    custom_value = avg_detection_time / total_images
    print(f"Wartość (ms/img): {custom_value:.4f}")

# === ZAPIS CSV W FORMACIE: mAP, opóźnienie, numer klatki ===
csv_output_path = os.path.join(plot_dir, "map_czas_zdjecie.csv")
with open(csv_output_path, "w") as f:
    f.write("mAP,opoznienie_ms,nr_klatki\n")
    for i, (map_val, delay) in enumerate(zip(map_scores, detection_times)):
        f.write(f"{map_val:.4f},{delay:.2f},{i+1}\n")
print(f"✅ Dane zapisane w CSV: {csv_output_path}")
