# # Author: Zylo117

# """
# Simple Inference Script of EfficientDet-Pytorch
# """
# import time
# import torch
# from torch.backends import cudnn
# from matplotlib import colors

# from backbone import EfficientDetBackbone
# import cv2
# import numpy as np

# from efficientdet.utils import BBoxTransform, ClipBoxes
# from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

# compound_coef = 0
# force_input_size = None  # set None to use default size
# #img_path = 'test/img.png'
# img_path = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/bottle_img.jpg'
# # replace this part with your project's anchor config
# anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
# anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

# threshold = 0.2
# iou_threshold = 0.2

# use_cuda = True
# use_float16 = False
# cudnn.fastest = True
# cudnn.benchmark = True

# obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#             'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
#             'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
#             'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#             'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
#             'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
#             'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
#             'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#             'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
#             'toothbrush']


# color_list = standard_to_bgr(STANDARD_COLORS)
# # tf bilinear interpolation is different from any other's, just make do
# input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
# input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
# ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

# if use_cuda:
#     x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
# else:
#     x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

# x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

# model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
#                              ratios=anchor_ratios, scales=anchor_scales)
# model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth', map_location='cpu'))
# model.requires_grad_(False)
# model.eval()

# if use_cuda:
#     model = model.cuda()
# if use_float16:
#     model = model.half()

# with torch.no_grad():
#     features, regression, classification, anchors = model(x)

#     regressBoxes = BBoxTransform()
#     clipBoxes = ClipBoxes()

#     out = postprocess(x,
#                       anchors, regression, classification,
#                       regressBoxes, clipBoxes,
#                       threshold, iou_threshold)

# def display(preds, imgs, imshow=True, imwrite=False):
#     for i in range(len(imgs)):
#         if len(preds[i]['rois']) == 0:
#             continue

#         imgs[i] = imgs[i].copy()

#         for j in range(len(preds[i]['rois'])):
#             #x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
#             x1, y1, x2, y2 = preds[i]['rois'][j].astype(int)  # lub np.int64 jeśli potrzebujesz konkretnej precyzji

#             obj = obj_list[preds[i]['class_ids'][j]]
#             score = float(preds[i]['scores'][j])
#             plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])


#         if imshow:
#             cv2.imshow('img', imgs[i])
#             cv2.waitKey(0)

#         if imwrite:
#             cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])


# out = invert_affine(framed_metas, out)
# display(out, ori_imgs, imshow=False, imwrite=True)

# print('running speed test...')
# with torch.no_grad():
#     print('test1: model inferring and postprocessing')
#     print('inferring image for 10 times...')
#     t1 = time.time()
#     for _ in range(10):
#         _, regression, classification, anchors = model(x)

#         out = postprocess(x,
#                           anchors, regression, classification,
#                           regressBoxes, clipBoxes,
#                           threshold, iou_threshold)
#         out = invert_affine(framed_metas, out)

#     t2 = time.time()
#     tact_time = (t2 - t1) / 10
#     print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

#     # uncomment this if you want a extreme fps test
#     # print('test2: model inferring only')
#     # print('inferring images for batch_size 32 for 10 times...')
#     # t1 = time.time()
#     # x = torch.cat([x] * 32, 0)
#     # for _ in range(10):
#     #     _, regression, classification, anchors = model(x)
#     #
#     # t2 = time.time()
#     # tact_time = (t2 - t1) / 10
#     # print(f'{tact_time} seconds, {32 / tact_time} FPS, @batch_size 32')

#MOJE
import time
import torch
from torch.backends import cudnn
import cv2
import numpy as np

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, plot_one_box

# Ustawienia
compound_coef = 2  # Zmień na wartość zgodną z Twoim modelem (np. D2)
weights_path = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/logs/bottle_coco/efficientdet-d2_30_1526.pth'
img_path = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/bottle_img_1.jpg'

# Konfiguracja detekcji
threshold = 0.2  # Próg wykrywania
iou_threshold = 0.2  # Próg IoU
use_cuda = True  # Używanie GPU
num_classes = 1  # Liczba klas (np. 1 jeśli wykrywa tylko butelki)
obj_list = ['bottle']  # Lista obiektów

# Ładowanie modelu
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=num_classes)
model.load_state_dict(torch.load(weights_path, map_location='cpu'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()

# Przetwarzanie obrazu
ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=1024)

if use_cuda:
    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
else:
    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

x = x.float().permute(0, 3, 1, 2)

# Detekcja obiektów
with torch.no_grad():
    features, regression, classification, anchors = model(x)

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)
    out = invert_affine(framed_metas, out)

# Rysowanie detekcji na obrazie
for i in range(len(ori_imgs)):
    if len(out[i]['rois']) == 0:
        continue

    img = ori_imgs[i].copy()

    for j in range(len(out[i]['rois'])):
        x1, y1, x2, y2 = out[i]['rois'][j].astype(int)
        obj = obj_list[out[i]['class_ids'][j]]
        score = float(out[i]['scores'][j])
        plot_one_box(img, [x1, y1, x2, y2], label=obj, score=score, color=(0, 255, 0))

    cv2.imwrite(f'detected_{i}.jpg', img)
    print(f"Zapisano obraz: detected_{i}.jpg")
