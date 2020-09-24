# 作业二 相机校正

<p align='right'>——518021910971 裴奕博</p>

### 实验原理

- 根据上课的推导，我们得到在相机成时，物点坐标（三维）、像点坐标（二维）和相机内参的关系为
  $$
  P^{\prime}=\left[\begin{array}{l}
  x^{\prime} \\
  y^{\prime} \\
  z
  \end{array}\right]=\left[\begin{array}{llll}
  \alpha & 0 & c_{x} & 0 \\
  0 & \beta & c_{y} & 0 \\
  0 & 0 & 1 & 0
  \end{array}\right]\left[\begin{array}{l}
  x \\
  y \\
  z \\
  1
  \end{array}\right]=\left[\begin{array}{llll}
  \alpha & 0 & c_{x} & 0 \\
  0 & \beta & c_{y} & 0 \\
  0 & 0 & 1 & 0
  \end{array}\right] P=K\left [ \begin{array}{ll} 
    I& 0
  \end{array}\right]  P
  $$

- 其中，$P^{\prime}$为像点坐标，$P$为物点坐标，$K$为相机的内参矩阵。

- 因此，我们可以通过取足够多组的$P$和$P^{\prime}$来求出$K$中的$\alpha,\beta,c_x,c_y$四个参数。

### OpenCV-Python代码实现

```python
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*10,3), np.float32)
objp[:,:2] = 5*np.mgrid[:7,:10].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('data/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray,cmap='gray')
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,10),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,10), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
printa(mtx)

# Code Reference:https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
```

### 工作流程

<img src="G:\MS332\homework\Week2_Camera_Calibration\workflow.png" style="zoom: 80%;" />

### 计算结果

![](G:\MS332\homework\Week2_Camera_Calibration\result.png)