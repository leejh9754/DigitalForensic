import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 150, 300, apertureSize = 3) # threshold1, threshold2, apertureSize
lines = cv2.HoughLines(edges, 1, np.pi/180, 150) # r정밀도(1), theta정밀도(1라디안), threshold
# threshold 값이 작을수록 디테일해짐

real_lines = []
W, H, C = img.shape
if lines is not None:
    for line in lines:
        rho,theta = line[0]

        for r,t in real_lines:
            if ((r-rho)/W)**2 + ((t-theta)/np.pi)**2 < 0.01: # threshold 값을 키울수록 선이 많이 사라짐
                break
        else:
            real_lines.append((rho, theta))

if lines is not None:
    for line in real_lines:
        rho,theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)

cross_point=[]

if lines is not None:
    for line in real_lines:
        for line2 in real_lines:
            rho1,theta1 = line
            a1 = np.cos(theta1)
            b1 = np.sin(theta1)
            x10 = a*rho1
            y10 = b*rho1
            rho2,theta2 = line2
            a2 = np.cos(theta2)
            b2 = np.sin(theta2)
            m1 = np.tan(theta1)
            m2 = np.tan(theta2)
            x20 = a*rho2
            y20 = b*rho2
            if m1==m2:
                break
            x3=float((m1*x10-y10-m2*x20+y20)/(m1-m2))
            y3=float(m1*(x3-x10)+y10)
            cross_point.append((x3,y3))   

survived_point = [ ]
for point in cross_point:
    count=0
    for point2 in cross_point:
        p1,p2=point
        q1,q2=point2
        distance=(p1-q1)*(p1-q1)+(p2-q2)*(p2-q2)
        if distance<30000:
            count=count+1
    if count>=3:
        survived_point.append(point)
        

if cross_point is not None:
    for point in survived_point:
        z,w=point
        plt.plot(z, w, 'ro')

plt.axis([0, 1000, 0, 1000])
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('matplotlib sample')
plt.show()
#cv2.imshow('result', img)
cv2.waitKey()
#cv2.imwrite('./image2-hough.jpg', img)
cv2.destroyAllWindows()