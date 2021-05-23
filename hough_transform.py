import cv2
import numpy as np

img = cv2.imread('./image2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 150, 300, apertureSize = 3) # threshold1, threshold2, apertureSize
lines = cv2.HoughLines(edges, 1, np.pi/180, 150) # r정밀도(1), theta정밀도(1라디안), threshold
# threshold 값이 작을수록 디테일해짐

for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)

cv2.imshow('edges', edges)
cv2.imshow('result', img)
cv2.waitKey()
cv2.imwrite('./image2-hough.jpg', img)
cv2.destroyAllWindows()