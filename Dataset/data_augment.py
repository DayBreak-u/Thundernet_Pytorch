import cv2
import numpy as np
import random


def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def _crop(image, boxes, labels, landm, img_dim):


    height, width, _ = image.shape
    pad_image_flag = True

    if random.uniform(0, 1) <= 0.05:
        image_aray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        for  i in range(3):
            image[...,i] = image_aray

    for _ in range(250):
        if height == 480 or width == 480:
            if random.uniform(0, 1) <= 0.5:
                scale = 1.0
            else:
                scale = random.uniform(0.85, 1.0)

        elif random.uniform(0, 1) <= 0.4:
            scale = 1.0
        else:
            scale = random.uniform(0.3, 1.0)
        # PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
        # scale = random.choice(PRE_SCALES)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 1)
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()

        landms_t = landm[mask_a].copy()

        landms_t = landms_t.reshape([-1, 5, 2])

        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

        # landm
        landms_t[:, :, :2] = landms_t[:, :, :2] - roi[:2]

        # landms_t[:, :, :2] = np.maximum(landms_t[:, :, :2], np.array([0, 0]))
        # landms_t[:, :, :2] = np.minimum(landms_t[:, :, :2], roi[2:] - roi[:2])

        landms_t = landms_t.reshape([-1, 10])


	# make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > 5
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]
        landms_t = landms_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        return image_t, boxes_t, labels_t, landms_t, pad_image_flag

    b_w_t = (boxes[:, 2] - boxes[:, 0] + 1) / width * img_dim
    b_h_t = (boxes[:, 3] - boxes[:, 1] + 1) / height * img_dim
    mask_b = np.minimum(b_w_t, b_h_t) > 5
    boxes = boxes[mask_b]
    labels = labels[mask_b]
    landm = landm[mask_b]

    return image, boxes, labels, landm, pad_image_flag


def _distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t


def _mirror(image, boxes, landms):
    height, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]

        # landm

        landms = landms.copy()
        landms = landms.reshape([-1, 5, 2])
        for i  in range(len(landms)):
            landm = landms[i]
            if np.any(landm < 0  ) or np.any(landm[:,0]>width ) or np.any(landm[:,1]>height ):
                landms[i] = np.ones([5,2]) * -1
            else:
                landms[i, :, 0] = width - landms[i, :, 0]
                tmp = landms[i, 1, :].copy()
                landms[i, 1, :] = landms[i, 0, :]
                landms[i, 0, :] = tmp
                tmp1 = landms[i, 4, :].copy()
                landms[i, 4, :] = landms[i, 3, :]
                landms[i, 3, :] = tmp1
        landms = landms.reshape([-1, 10])


    return image, boxes, landms


def _pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def _resize_subtract_mean(image, insize, rgb_mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_mean
    return image.transpose(2, 0, 1)

def _subtract_mean(image, rgb_mean):
    image = image.astype(np.float32)
    image -= rgb_mean
    return image.transpose(2, 0, 1)


# class preproc(object):
#
#     def __init__(self, img_dim, rgb_means):
#         self.img_dim = img_dim
#         self.rgb_means = rgb_means
#
#     def __call__(self, image, targets):
#         assert targets.shape[0] > 0, "this image does not have gt"
#
#         boxes = targets[:, :4].copy()
#         labels = targets[:, -1].copy()
#         landm = targets[:, 4:-1].copy()
#
#         image_t, boxes_t, labels_t, landm_t, pad_image_flag = _crop(image, boxes, labels, landm, self.img_dim)
#         image_t = _distort(image_t)
#         image_t = _pad_to_square(image_t,self.rgb_means, pad_image_flag)
#         image_t, boxes_t, landm_t = _mirror(image_t, boxes_t, landm_t)
#         height, width, _ = image_t.shape
#         image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)
#         boxes_t[:, 0::2] /= width
#         boxes_t[:, 1::2] /= height
#
#         landm_t[:, 0::2] /= width
#         landm_t[:, 1::2] /= height
#
#         labels_t = np.expand_dims(labels_t, 1)
#         targets_t = np.hstack((boxes_t, landm_t, labels_t))
#
#         return image_t, targets_t

class preproc(object):

    def __init__(self, img_dim ,  rgb_mean = (104, 117, 123) ):
        self.img_dim = img_dim
        self.rgb_means = rgb_mean

    def __call__(self, image, boxes , landm ,labels):
        # assert targets.shape[0] > 0, "this image does not have gt"


        image_t, boxes_t, labels_t, landm_t, pad_image_flag = _crop(image, boxes, labels, landm, self.img_dim)
        image_t = _distort(image_t)
        image_t = _pad_to_square(image_t,self.rgb_means, pad_image_flag)
        image_t, boxes_t, landm_t = _mirror(image_t, boxes_t, landm_t)
        height, width, _ = image_t.shape
        # image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)

        # boxes_t[:, 0::2] /= width
        # boxes_t[:, 1::2] /= height
        #
        # landm_t[:, 0::2] /= width
        # landm_t[:, 1::2] /= height

        debug = 0
        if debug:
            cv2.imwrite("test.jpg", image_t)
            image_t = cv2.imread("test.jpg")
            for box, landmark5 in zip(boxes_t, landm_t):
                landmark5 = landmark5.astype(np.int).reshape([-1, 2])
                box = box.astype(np.int)
                cv2.rectangle(image_t, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                # print(landmark.shape)
                for l in range(landmark5.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(image_t, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
            cv2.imwrite("temp.jpg", image_t)


        # image_t = image_t[...,::-1]

        # image_t = image_t.transpose(2, 0, 1)

        labels_t = np.expand_dims(labels_t, 1)
        # targets_t = np.hstack((boxes_t, landm_t, labels_t))
        image_t = np.ascontiguousarray(image_t, dtype=np.float32)
        return image_t  , boxes_t,labels_t,landm_t
        # return image_t / 255.0 , boxes_t,labels_t,landm_t