import cv2
import numpy as np
import glob
import time
from PIL import Image, ImageEnhance


def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

# ------ ZMENA KONTRASTU/SATURACIE/SVETLOSTI ---------
imageload = Image.open("final-cut.png")
contrast = ImageEnhance.Contrast(imageload).enhance(1.2)
saturation = ImageEnhance.Color(contrast).enhance(2)
brightness = ImageEnhance.Brightness(saturation).enhance(1.5)
brightness.save("final-cut-con.png")
# ------ ZMENA KONTRASTU/SATURACIE/SVETLOSTI ---------

image = cv2.imread("final-cut-con.png", 0)
image_zal = cv2.imread("final-cut.png")

cv2.namedWindow('original', cv2.WINDOW_AUTOSIZE)
cv2.imshow("original", image_zal)

y = 0
x = 0
h = 750
w = 1010

# ------ PERSPEKTIVA ---------
rows, cols = image.shape
print(rows, cols)
pts1 = np.float32([[0, 0], [0, 1010], [738, 20], [755, 983]])
pts2 = np.float32([[0, 0], [0, 1010], [750, 0], [750, 1010]])
wrap_per = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(image, wrap_per, (750, 1010))
# ------ PERSPEKTIVA ---------

# ------ OREZANIE ------------
imagecr = dst[y:y+w, x:x+h]
# ------ OREZANIE ------------

templates = glob.glob("tabulky-fotky/*")
final = [0.0, ""]

dt = time.time()

for t in templates:
    img = cv2.imread(t, 0)
    # ------ RESIZE TIME ------------
    reszt = time.time()
    dim = (750, 1010)
    resized = cv2.resize(img, dim)
    print(f"----------------------------------\nResize time: {time.time()-reszt}")
    # ------ RESIZE TIME ------------

    res = cv2.matchTemplate(imagecr, resized, cv2.TM_CCORR_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    print (f"Subor: {t} \nZhoda: {max_val}%")
    if final[0] < max_val:
        final[0] = max_val
        final[1] = t
    bottom_right = (top_left[0] + dim[0], top_left[1] + dim[1])
    # ------ FOR TIME ------------
    print(f"FOR Time: {time.time() - dt}")

print(f"----------------------------------\nCELKOVY CAS: {time.time() - dt}")
print(f"SUBOR: {final[1]}\nZHODA: {round(final[0]*100, 2)}%")

fimage = cv2.imread(final[1])
perc = round(final[0], 3)
cv2.imshow("hmm {}%".format(perc), cv2.resize(fimage, (750, 1010)))
cv2.waitKey(0)
