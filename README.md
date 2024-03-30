# Optical pose estimation in robotics applications
Nowadays, Machine Vision has become very common and popular in the field of robotics, especially when it comes down to precise measurements. Combining these two fields has made many previously difficult or insoluble tasks feasible. There are many factors that can affect the exact measurement of different components. These include, for example, various distortions (e.g. barrel, pincushion) on the images, handling the case of occluded points, poor lighting conditions and poor image quality. All this must be considered and eliminated during implementation.

Camera calibration and image undistortion are standard operations within the field of Computer Vision. All of these are the basis of fundamental tasks such as pose estimation and optical measurements of objects. The common solution to find certain key points on images is to use different feature matching algorithms, such as SIFT (Scale Invariant Feature Transform) or SURF (Speeded Up Robust Features). For pose estimation, the classic way is the PnP (Perspective-n-Point) algorithm.

The goal of this thesis is to design a measurement process for the pose estimation of different electrical components such as motherboards, memory modules, CPUs, etc. To facilitate the implementation of the task, the OpenCV software library should be used. This directory contains all the essential image processing functions necessary to accomplish this task. The final step in the work is testing the system with different objects and then thoroughly evaluating the measured data.


## Requriments
| Package               | Version    |
|-----------------------|------------|
| colorama              | 0.4.6      |
| colorlog              | 6.8.2      |
| numpy                 | 1.26.4     |
| opencv-contrib-python | 4.9.0.80   |
| opencv-python         | 4.9.0.80   |
| pandas                | 2.2.1      |
| pip                   | 23.2.1     |
| python-dateutil       | 2.9.0.post0|
| pytz                  | 2024.1     |
| six                   | 1.16.0     |
| tqdm                  | 4.66.2     |
| tzdata                | 2024.1     |
