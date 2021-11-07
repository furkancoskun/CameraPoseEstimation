# M. Furkan Coskun
# Estimate 6 Dof Camera Pose Test
# 07.11.2021

import numpy as np
import cv2

image_coordinates = np.load("vr2d.npy")
image_coordinates = np.expand_dims(image_coordinates, axis=0)
world_coordinates = np.load("vr3d.npy")
world_coordinates = np.expand_dims(world_coordinates, axis=0)

img1 = cv2.imread("img1.png")
img2 = cv2.imread("img2.png")
img3 = cv2.imread("img3.png")

initial_camera_matrix = np.array([[100, 0, 960], [0, 100, 540], [0, 0, 1]], np.float32)
_, camera_matrix, _, _, _ = cv2.calibrateCamera(world_coordinates, image_coordinates, 
                           (img1.shape[1],img1.shape[0]), initial_camera_matrix, None,
                           flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT +
                           cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST )

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
kp3, des3 = orb.detectAndCompute(img3,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

img1_img2_matches = bf.knnMatch(des1, des2, k=2)
img1_img3_matches = bf.knnMatch(des1, des3, k=2)

img1_img2_matches_ratio_test = []
for m,n in img1_img2_matches:
    if m.distance < 0.8*n.distance:
        img1_img2_matches_ratio_test.append([m])
img1_img2_matches_ratio_test = sorted(img1_img2_matches_ratio_test, key=lambda x: x[0].distance)
if (len(img1_img2_matches_ratio_test) < 50 ):
    img1_img2_good_matches = img1_img2_matches_ratio_test
else :
    img1_img2_good_matches = img1_img2_matches_ratio_test[:50]

img1_img3_matches_ratio_test = []
for m,n in img1_img3_matches:
    if m.distance < 0.8*n.distance:
        img1_img3_matches_ratio_test.append([m])
img1_img3_matches_ratio_test = sorted(img1_img3_matches_ratio_test, key=lambda x: x[0].distance)
if (len(img1_img3_matches_ratio_test) < 50 ):
    img1_img3_good_matches = img1_img3_matches_ratio_test
else :
    img1_img3_good_matches = img1_img3_matches_ratio_test[:50]

img1_wrt2_points = []
img2_points = []
for match in img1_img2_good_matches:
  img1_wrt2_points.append(kp1[match[0].queryIdx].pt)
  img2_points.append(kp2[match[0].trainIdx].pt)
img1_wrt2_points = np.float32(img1_wrt2_points)
img2_points = np.float32(img2_points)

img1_wrt3_points = []
img3_points = []
for match in img1_img3_good_matches:
  img1_wrt3_points.append(kp1[match[0].queryIdx].pt)
  img3_points.append(kp3[match[0].trainIdx].pt)    
img1_wrt3_points = np.float32(img1_wrt3_points)
img3_points = np.float32(img3_points)

essential_matrix_img2_wrt_img1, _ = cv2.findEssentialMat(img2_points, img1_wrt2_points,
                                                         method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, RotMat_img2_wrt_img1, T_img2_wrt_img1, _ = cv2.recoverPose(essential_matrix_img2_wrt_img1, img2_points, 
                                                              img1_wrt2_points, camera_matrix)
RotVec_img2_wrt_img1 = cv2.Rodrigues(RotMat_img2_wrt_img1)

essential_matrix_img3_wrt_img1, _ = cv2.findEssentialMat(img3_points, img1_wrt3_points, 
                                                         method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, RotMat_img3_wrt_img1, T_img3_wrt_img1, _ = cv2.recoverPose(essential_matrix_img3_wrt_img1, img3_points, 
                                                              img1_wrt3_points, camera_matrix)
RotVec_img3_wrt_img1 = cv2.Rodrigues(RotMat_img3_wrt_img1)

print("------ Camera Pose in img2 w.r.t. img1 ------")
print("- Rotation(Rx, Ry, Rz) in Radians -")
print(RotVec_img2_wrt_img1[0])
print("- Rotation(Rx, Ry, Rz) in Degrees -")
print(np.rad2deg(RotVec_img2_wrt_img1[0]))
print("- Unit Translation(Tx, Ty, Tz) Vector -")
print(T_img2_wrt_img1)

origin_point = np.float32([0, 0, 0])
projected_point, _ = cv2.projectPoints(origin_point, np.float32(RotVec_img2_wrt_img1[0]), 
                                       np.float32(T_img2_wrt_img1), camera_matrix, None)
projected_point = projected_point/100000
img2_trajectory_on_img1 = img1.copy()
img2_trajectory_on_img1 = cv2.arrowedLine(img2_trajectory_on_img1, (960,1080), 
                                          (int(960+projected_point[0][0][0]),int(1080-projected_point[0][0][1])), 
                                          (0,0,255), thickness=5)
cv2.imwrite("img2_trajectory_on_img1.png",img2_trajectory_on_img1)

print("\n---------------------------------------------\n")

print("------ Camera Pose in img3 w.r.t. img1 ------")
print("- Rotation(Rx, Ry, Rz) in Radians -")
print(RotVec_img3_wrt_img1[0])
print("- Rotation(Rx, Ry, Rz) in Degrees -")
print(np.rad2deg(RotVec_img3_wrt_img1[0]))
print("- Unit Translation(Tx, Ty, Tz) Vector -")
print(T_img3_wrt_img1)

origin_point = np.float32([0, 0, 0])
projected_point, _ = cv2.projectPoints(origin_point, np.float32(RotVec_img3_wrt_img1[0]), 
                                       np.float32(T_img3_wrt_img1), camera_matrix, None)
projected_point = projected_point/5000
img3_trajectory_on_img1 = img1.copy()
img3_trajectory_on_img1 = cv2.arrowedLine(img3_trajectory_on_img1, (960,1080), 
                                          (int(960+projected_point[0][0][0]),int(1080-projected_point[0][0][1])), 
                                          (0,0,255), thickness=5)
cv2.imwrite("img3_trajectory_on_img1.png",img3_trajectory_on_img1)