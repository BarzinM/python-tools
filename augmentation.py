import numpy as np
import cv2

def resize(array, dimensions):
    a = np.empty((len(array), *dimensions), dtype=np.uint8)
    for i in range(len(array)):
        a[i] = cv2.resize(array[i], dimensions[:2])

    return a


def mirrorFrames(frames):
    return np.flip(frames, 2)


def mirrorLidar(readings):
    return np.flip(readings, 1)



def rotate(image, angle):
    rows, cols, ch = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, rotation_matrix, (cols, rows))


def scale(image, scale_x, scale_y):
    return cv2.resize(image, None, fx=scale_x, fy=scale_y)


def transform(image, rotation, translation_h, translation_v, scale_h, scale_v):
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(
        (cols / 2, rows / 2), rotation, 1)
    translation_matrix = np.float32(
        [[1, 0, translation_h], [0, 1, translation_v]])
    scaling_matrix = np.float32(
        [[scale_h, 0, (1 - scale_h) * rows], [0, scale_v, (1 - scale_v) * cols]])
    image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    image = cv2.warpAffine(image, scaling_matrix, (cols, rows))
    image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

    return image


def batchTransform(data, rotation, translation, scaling):
    size = len(data)
    rotations = np.random.uniform(-rotation, rotation, size)
    scalings = np.exp(np.random.uniform(- scaling, scaling, (size, 2)))
    translations = np.random.uniform(-translation, translation, (size, 2))
    for i in range(size):
        data[i] = transform(data[i], rotations[i], *translations[i], *scalings[i])

    return data
