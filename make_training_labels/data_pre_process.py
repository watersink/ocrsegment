import cv2
import numpy as np
import os
import scipy.ndimage as ndi


def _get_text_line_image(image,line_label):
    x1 = line_label[0]
    y1 = line_label[1]
    x2 = line_label[2]
    y2 = line_label[3]
    h = y2-y1
    image_line = image[y1:y2, x1:x2]

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_line)

    image_connect = np.zeros(image_line.shape, dtype=np.uint8)


    for i in range(1, stats.shape[0]):
        x1 = stats[i, 0]
        y1 = stats[i, 1]
        x2 = stats[i, 2] + x1
        y2 = stats[i, 3] + y1
        image_connect[y1:y2, x1:x2] = 255


    image_anigauss = ndi.gaussian_filter(image_connect, sigma=[h/2, h], mode='constant', cval=0)
    image_line_center_index = np.argmax(image_anigauss, 0)

    image_center_gauss_index = ndi.gaussian_filter1d(image_line_center_index, sigma=h/3)
    for k in range(len(image_center_gauss_index)):
        image_anigauss[image_center_gauss_index[k], k] = 255

    image_bin2 = (np.uint8(image_anigauss)>254)*255.0

    kernel = np.ones((3, 3), np.uint8)
    image_dilation = cv2.dilate(image_bin2, kernel, iterations=3)
    temp = (np.uint8(image_dilation)>254)*255

    return temp




if __name__ == '__main__':
    image_name="K47LYBN.framed.png"
    image = cv2.imread(image_name, 0)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    image = 255-image
    label_image = np.zeros_like(image)
    text_file = os.path.splitext(image_name)[0]+'.txt'
    with open(text_file, "r", encoding="utf-8") as f:
        labels = f.readlines()
    labels=[labels[i].rstrip("\n").split(",") for i in range(len(labels))]
    label_arrays=np.asarray(labels,np.int32)
    for line in label_arrays:
        label_line_image = _get_text_line_image(image,line)
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]
        label_image[y1:y2, x1:x2] = label_line_image

    cv2.imwrite(image_name.replace("framed","lines"), label_image)

