# import the necessary packages
from imutils.video import FileVideoStream, VideoStream, WebcamVideoStream
from imutils.video import FPS
import imutils
import time
from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter, Resize
import cv2
import numpy as np
from EfficientLight_RCNN import model
import torch
from torchvision.ops import nms
from torchvision.ops.boxes import remove_small_boxes






testtransform = Compose([ToTensor()])


vs = cv2.VideoCapture("/home/princemerveil/Desktop/zozo.mp4")
height = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)

width = vs.get(cv2.CAP_PROP_FRAME_WIDTH)

time.sleep(1.0)


frame_count = 150
model.cuda()
model.load_state_dict(torch.load('./checkpoint/efficient_model_L_7.pth'))
model.eval()

fps = FPS().start()

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('video.avi', fourcc, 25, (int(width), int(height)))

while True:
    ret, frame = vs.read()

    if frame is None:
        break

    if ret:
        tensor = testtransform(frame)
        start = time.time()
        predictions = model([tensor.cuda()])
        end = time.time()

        keeps = nms(boxes=predictions[0]['boxes'], scores=predictions[0]['scores'], iou_threshold=0.5)
        keeps = keeps.cpu().detach().numpy()
        predictions = predictions[0]['boxes'].cpu().detach().numpy()
        predictions = predictions[keeps]

        for box in predictions:
            cv2.rectangle(frame, (int(box[0]), int(box[1]), int(box[2]) - int(box[0]), int(box[3]) - int(box[1])), (255, 225, 0), 2)
        fps_time = 1. / (end - start)

        cv2.putText(frame, "NVIDIA GTX 1060 6G", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, 'FPS: %.2f' % fps_time, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.imshow("fdf", frame)


        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
           break
        fps.update()





fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup

vs.stream.release()
cv2.destroyAllWindows()
vs.stop()
out.release()