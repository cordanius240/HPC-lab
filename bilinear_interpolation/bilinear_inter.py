import numpy as np
from numba import cuda, uint8
import cv2
import time
import math
@cuda.jit
def kern(img, result):
    idx, jdx = cuda.grid(2)
    if idx < result.shape[0] and jdx < result.shape[1]:
        x = idx // 2
        y = jdx // 2
        if x < img.shape[0] - 1 and y < img.shape[1] - 1:
            dx = idx % 2
            dy = jdx % 2
            x1 = min(x + 1, img.shape[0] - 1)
            y1 = min(y + 1, img.shape[1] - 1)
            result[idx, jdx] = (
                img[x, y] * (1 - dx) * (1 - dy) +
                img[x, y1] * (1 - dx) * dy +
                img[x1, y] * dx * (1 - dy) +
                img[x1, y1] * dx * dy
            )


def gpu_bilinear_interpolation(img):
    threads_per_block = (8, 8)
    blocks_per_grid = (
        math.ceil(img.shape[0] * 2 / threads_per_block[0]),
        math.ceil(img.shape[1] * 2 / threads_per_block[1])
    )

    startp = cuda.event()
    endp = cuda.event()
    startp.record()

    img_dev = cuda.to_device(img)
    result_dev = cuda.to_device(np.zeros((img.shape[0] * 2, img.shape[1] * 2), dtype=np.float32))

    kern[blocks_per_grid, threads_per_block](img_dev, result_dev)
    endp.record()
    endp.synchronize()
    elapsedp = cuda.event_elapsed_time(startp, endp)

    return result_dev.copy_to_host(), elapsedp / 1000

def cpu_bilinear_interpolation(img):
    result = np.zeros((img.shape[0] * 2, img.shape[1] * 2), dtype=np.float32)

    for idx in range(result.shape[0]):
        for jdx in range(result.shape[1]):
            x = idx // 2
            y = jdx // 2
            if x < img.shape[0] - 1 and y < img.shape[1] - 1:
                dx = idx % 2
                dy = jdx % 2
                x1 = min(x + 1, img.shape[0] - 1)
                y1 = min(y + 1, img.shape[1] - 1)
                result[idx, jdx] = (
                    img[x, y] * (1 - dx) * (1 - dy) +
                    img[x, y1] * (1 - dx) * dy +
                    img[x1, y] * dx * (1 - dy) +
                    img[x1, y1] * dx * dy
                )

    return result

def cpu_bilinear_interpolation_time(img):
    startp = time.time()

    result = cpu_bilinear_interpolation(img)

    elapsedp = time.time() - startp
    return result, elapsedp

image = cv2.imread('picture600.bmp')
gray_image = cv2.imread('picture600.bmp', cv2.IMREAD_GRAYSCALE)
print("Размеры входной картинки: ", gray_image.shape[0], gray_image.shape[1])
result_gpu, result_time_gpu=gpu_bilinear_interpolation(gray_image)
start_time = time.time()
result_сpu,result_time_cpu=cpu_bilinear_interpolation_time(gray_image)
end_time = time.time()
# Вычисляем время выполнения
execution_time = end_time - start_time
print("Затраченное время GPU: ", result_time_gpu)
print("Затраченное время CPU: ", execution_time)
print("Time cpu/gpu: ", execution_time / result_time_gpu)
cv2.imshow('Gray Image', gray_image)
cv2.imshow('Result GPU', result_gpu)
cv2.imwrite('picture_2xgpu600.bmp', result_gpu)
cv2.imwrite('picture_2xcpu600.bmp', result_сpu)
cv2.waitKey(0)
cv2.destroyAllWindows()