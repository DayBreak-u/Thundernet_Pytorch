from collections import  Counter

txts = ["./VOC2007_trainval.txt", "./VOC2012_trainval.txt"]

CLASSES =  ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

all_lines  = open(txts[0]).readlines() + open(txts[1]).readlines()
all_class = []
for line in all_lines:
    if line.startswith("#"):
        # print(line)
        continue
    line = line.strip().split()

    all_class.append(CLASSES[int(line[-1])])

print(Counter(all_class))