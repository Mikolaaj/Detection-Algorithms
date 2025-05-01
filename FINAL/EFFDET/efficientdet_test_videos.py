# # Core Author: Zylo117
# # Script's Author: winter2897 

# """
# Simple Inference Script of EfficientDet-Pytorch for detecting objects on webcam
# """
# import time
# import torch
# import cv2
# import numpy as np
# from torch.backends import cudnn
# from backbone import EfficientDetBackbone
# from efficientdet.utils import BBoxTransform, ClipBoxes
# from utils.utils import preprocess, invert_affine, postprocess, preprocess_video

# # Video's path
# video_src = 'water_in_factory.mp4'  # set int to use webcam, set str to read from a video file

# compound_coef = 0
# force_input_size = None  # set None to use default size

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

# # tf bilinear interpolation is different from any other's, just make do
# input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
# input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

# # load model
# model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
# #model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
# model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth', weights_only=True))
# model.requires_grad_(False)
# model.eval()

# if use_cuda:
#     model = model.cuda()
# if use_float16:
#     model = model.half()

# # function for display
# def display(preds, imgs):
#     for i in range(len(imgs)):
#         if len(preds[i]['rois']) == 0:
#             return imgs[i]

#         for j in range(len(preds[i]['rois'])):
#             #(x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
#             (x1, y1, x2, y2) = preds[i]['rois'][j].astype(int)
#             cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
#             obj = obj_list[preds[i]['class_ids'][j]]
#             score = float(preds[i]['scores'][j])

#             cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
#                         (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                         (255, 255, 0), 1)
        
#         return imgs[i]
# # Box
# regressBoxes = BBoxTransform()
# clipBoxes = ClipBoxes()

# # Video capture
# cap = cv2.VideoCapture(video_src)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # frame preprocessing
#     ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)

#     if use_cuda:
#         x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
#     else:
#         x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

#     x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

#     # model predict
#     with torch.no_grad():
#         features, regression, classification, anchors = model(x)

#         out = postprocess(x,
#                         anchors, regression, classification,
#                         regressBoxes, clipBoxes,
#                         threshold, iou_threshold)

#     # result
#     out = invert_affine(framed_metas, out)
#     img_show = display(out, ori_imgs)
    
#     # Pobranie rozdzielczości ekranu
#     screen_width = 1920  # Zmień na cv2.getWindowImageRect('frame')[2] jeśli działa na Twoim systemie
#     screen_height = 1080  # Zmień na cv2.getWindowImageRect('frame')[3] jeśli działa na Twoim systemie

#     # Pobranie wymiarów wideo
#     frame_height, frame_width = img_show.shape[:2]

#     # Obliczenie współczynnika skalowania, aby dopasować obraz do ekranu
#     scale = min(screen_width / frame_width, screen_height / frame_height)

#     # Zmiana rozmiaru okna wyświetlania
#     new_width = int(frame_width * scale)
#     new_height = int(frame_height * scale)
#     img_show = cv2.resize(img_show, (new_width, new_height))

#     # show frame by frame
#     cv2.imshow('frame',img_show)
#     if cv2.waitKey(1) & 0xFF == ord('q'): 
#         break

# cap.release()
# cv2.destroyAllWindows()

import time
import torch
import cv2
import numpy as np
from torch.backends import cudnn
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video

# Video's path
video_src = 'water_in_factory.mp4'  # set int to use webcam, set str to read from a video file

compound_coef = 2  # Zmienione na zgodne z Twoim modelem
torch_model_path = '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/logs/bottle_coco/efficientdet-d2_30_1526.pth'
force_input_size = None  # set None to use default size

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

# Zakładam, że model wykrywa butelki, zmień jeśli masz inne klasy
obj_list = ['bottle']

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

# Wczytanie modelu
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
model.load_state_dict(torch.load(torch_model_path, map_location='cuda' if use_cuda else 'cpu'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

# Funkcja do wyświetlania wyników
def display(preds, imgs):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            return imgs[i]

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], f'{obj}, {score:.3f}',
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
        
        return imgs[i]

regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

cap = cv2.VideoCapture(video_src)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = model(x)
        out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)

    out = invert_affine(framed_metas, out)
    img_show = display(out, ori_imgs)
    
    screen_width, screen_height = 1920, 1080  # Możesz zmienić na dynamiczne pobieranie rozdzielczości
    frame_height, frame_width = img_show.shape[:2]
    scale = min(screen_width / frame_width, screen_height / frame_height)
    new_width, new_height = int(frame_width * scale), int(frame_height * scale)
    img_show = cv2.resize(img_show, (new_width, new_height))

    cv2.imshow('frame', img_show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
