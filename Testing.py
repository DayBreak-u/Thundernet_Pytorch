import torch
import shutil
from torchvision.transforms import ToTensor, Compose

from PIL import Image
import time
import numpy as np
import cv2
from config import Configs
from Utils.py_cpu_nms import  py_cpu_nms
from versions.nn.Snet import  get_thundernet
import  os

CLASSES = Configs.get("CLASSES")
VOCROOT = Configs.get("VOCROOT")


# img_path = "/mnt/data1/yanghuiyu/dlmodel/Fd/Face-Detector-1MB-with-landmark/images/input/10.jpg"
if os.path.exists("./result"):
    shutil.rmtree("./result")
os.mkdir("./result")



model = get_thundernet()

model.cuda()
model.load_state_dict(torch.load('./save_weights/efficient_rcnn_17.pth'))
model.eval()


datas = open("data_pre/VOC2012_test.txt").readlines()
for img_path in datas[:10]:

    # img_path =  "VOC2012/JPEGImages/2008_000039.jpg"
    img_path =  os.path.join(VOCROOT,img_path.strip()[1:])
    imge = Image.open(img_path)
    testtransform = Compose([ToTensor()])
    img = testtransform(imge)


    start = time.time()
    print(img.size())
    results = model([img.cuda()])
    open_cv_image = np.array(imge)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    boxes = []
    for box,label,score,  in zip(results[0]['boxes'],results[0]['labels'],results[0]["scores"]):
        boxes.append(box[:4].tolist() + [label] + [score])


    boxes = np.array(boxes)
    if boxes.shape[0] != 0:
        keep = py_cpu_nms(boxes, 0.35)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        boxes = boxes[keep, :]


    for box in boxes:
        if box[4] < 0.4:
            continue
        box = box.tolist()
        score = float(box[5])

        label_id  = int(box[4])
        label = CLASSES[label_id]
        cv2.rectangle(open_cv_image, (int(box[0]), int(box[1]), int(box[2]) - int(box[0]), int(box[3]) - int(box[1])), (255, 225, 0), 2)
        cx = box[0]
        cy = box[1] + 12
        cv2.putText(open_cv_image, "{}:{:.2f}".format(label,score), (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    # cv2.imshow("sd", open_cv_image)
    cv2.imwrite("result/{}".format(img_path.split("/")[-1]), open_cv_image)
# cv2.waitKey(30000)

