python train.py -c 2 -p bottle_coco --batch_size 8 --lr 1e-3 --num_epochs 10 --load_weights weights/efficientdet-d2.pth
python train.py -c 2 -p bottle_coco --batch_size 8 --lr 1e-3 --num_epochs 30 --load_weights weights/efficientdet-d2.pth

python train.py -c 2 -p bottle_coco --batch_size 8 --lr 1e-3 --num_epochs 4 --load_weights weights/efficientdet-d2.pth --head_only True
python train.py -c 2 -p bottle_coco --batch_size 8 --lr 1e-3 --num_epochs 30 --load_weights weights/efficientdet-d2.pth --head_only True

python coco_eval.py -p bottle_coco -c 2 -w logs/bottle_coco/efficientdet-d2_9_500.pth

python coco_eval.py -p bottle_coco -c 2 -w logs/bottle_coco/efficientdet-d2_29_1500.pth

 python coco_eval.py -p bottle_coco -c 2 -w weights/efficientdet-d2.pth
 python coco_eval.py -p bottle_coco -c 2 -w logs/bottle_coco/efficientdet-d2_9_500.pth
 python coco_eval.py -p bottle_coco -c 2 -w logs/bottle_coco/efficientdet-d2_30_1526.pth

python coco_eval.py -p datasets/cattle_coco/ -c 2 -w TRAINED_EFFDET/cattle_50ep_d2/efficientdet-d2_49_8500.pth
python coco_eval.py -p cattle_coco -c 2 -w TRAINED_EFFDET/cattle_50ep_d2/efficientdet-d2_49_8500.pth --nms_threshold 0.5 --cuda True --device 0 --float16 True --override True

python coco_eval.py --project cattle_coco --weights /mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/TRAINED_EFFDET/cattle_50ep_d2/efficientdet-d2_49_8500.pth

 #Check onnx model website : https://netron.app

zeby dodstac sie do tensor borda (trzeba byc w YET-...): tensorboard --logdir logs/


python train.py -c 2 -p bottle_coco --batch_size 8 --lr 1e-3 --num_epochs 50 --load_weights weights/efficientdet-d2.pth
python train.py -c 2 -p bottle_coco --batch_size 8 --lr 1e-3 --num_epochs 30 --load_weights weights/efficientdet-d1.pth
python train.py -c 1 -p cattle_coco --batch_size 4 --lr 1e-3 --num_epochs 50 --load_weights weights/efficientdet-d1.pth
python train.py -c 4 -p cattle_coco --batch_size 1 --lr 1e-3 --num_epochs 50 --load_weights weights/efficientdet-d4.pth

python train.py -c 2 -p cattle_coco --batch_size 4 --lr 1e-3 --num_epochs 150 --load_weights weights/efficientdet-d2.pth


python train.py -c 2 -p cattle_coco --batch_size 1 --lr 1e-3 --num_epochs 30 --load_weights weights/efficientdet-d2.pth --head_only True

 python coco_eval.py -p bottle_coco -c 2 -w logs/bottle_coco/efficientdet-d2_30_1526.pth
python coco_eval.py -p cattle_coco -c 2 -w TRAINED_EFFDET/cattle_50ep_d0/efficientdet-d0_49_8500.pth


python train.py -c 2 -p CATTLE_coco --batch_size 8 --lr 1e-3 --num_epochs 50 --load_weights weights/efficientdet-d2.pth
python train.py -c 2 -p cattle_coco --batch_size 4 --lr 1e-3 --num_epochs 50 --load_weights weights/efficientdet-d2.pth
python train.py -c 0 -p cattle_coco --batch_size 4 --lr 1e-3 --num_epochs 50 --load_weights weights/efficientdet-d0.pth

/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/TRAINED_EFFDET/cattle_50ep_d2/efficientdet-d2_49_8500.pth


python train.py -c 2 -p car_coco --batch_size 4 --lr 1e-3 --num_epochs 50 --load_weights weights/efficientdet-d2.pth


##########################################################################################################################
tensorboard --logdir TRAINED_EFFDET/CATTLE/cattle_50ep_d0/
tensorboard --logdir TRAINED_EFFDET/CATTLE/cattle_50ep_d1/
tensorboard --logdir TRAINED_EFFDET/CATTLE/cattle_50ep_d2/
tensorboard --logdir TRAINED_EFFDET/CATTLE/cattle_50ep_d3/
tensorboard --logdir TRAINED_EFFDET/CATTLE/cattle_50ep_d4/


python coco_eval.py -p cattle_coco -c 0 -w /mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/TRAINED_EFFDET/CATTLE/cattle_50ep_d0/efficientdet-d0_49_8650.pth
python coco_eval.py -p cattle_coco -c 1 -w /mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/TRAINED_EFFDET/CATTLE/cattle_50ep_d1/efficientdet-d1_49_8650.pth
python coco_eval.py -p cattle_coco -c 2 -w /mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/TRAINED_EFFDET/CATTLE/cattle_50ep_d2/efficientdet-d2_49_8500.pth
python coco_eval.py -p cattle_coco -c 3 -w /mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/TRAINED_EFFDET/CATTLE/cattle_50ep_d3/efficientdet-d3_48_17000.pth
python coco_eval.py -p cattle_coco -c 4 -w /mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/TRAINED_EFFDET/CATTLE/cattle_50ep_d4/efficientdet-d4_49_34500.pth
