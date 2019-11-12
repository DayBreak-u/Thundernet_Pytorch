import  cv2
import  sys
sys.path.insert(0,"..")
from config import  Configs
import os
import numpy as np
import time
CLASSES = Configs.get("CLASSES")
VOCROOT = Configs.get("VOCROOT")

txt_path = "./VOC2007_trainval.txt"
isFirst = True
image_dat = []


datas = open(txt_path, "r").readlines()
for line in datas:
    line = line.rstrip()
    if line.startswith('#'):
        if isFirst:
            boxes = []
            labels = []
            annotation = {}
            isFirst = False
        else:
            if len(boxes) == 0:
                continue
            annotation["filepath"] = path
            annotation["bboxes"] = boxes
            annotation["labels"] = labels
            debug = 1
            if debug :
                im = cv2.imread(path)


                cv2.imwrite("test.jpg", im)
                im = cv2.imread("test.jpg")
                for box, label in zip(boxes,labels):
                    # print(label)
                    label = CLASSES[label]

                    cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cx,cy = box[0] , box[1] +10
                    cv2.putText(im, "{}".format(label), (int(cx), int(cy)),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
                cv2.imwrite("temp.jpg", im)
                time.sleep(1)



            image_dat.append(annotation)
            boxes = []
            labels = []
            annotation = {}
        path = line[1:].strip()
        # path = txt_path.replace('label.txt','images/') + path
        path = os.path.join(VOCROOT, path)
    else:
        line = line.split(' ')
        label = [float(x) for x in line]

        x1, y1, w, h = label[:4]
        x2 = x1 + w
        y2 = y1 + h
        class_id = label[4]

        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        labels.append(int(class_id))

