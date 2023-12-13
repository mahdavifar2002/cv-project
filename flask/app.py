import os

from flask import Flask, redirect, jsonify, request, url_for, render_template, flash, send_from_directory
from flask_cors import CORS, cross_origin
import torch
import cv2
import json
import msgpack
import math
from PIL import Image
import numpy as np
from numpy.linalg import inv
from scipy import optimize
from gltf_builder import mesh_to_glb
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["IMAGE_UPLOADS"] = "./images"
app.config["MODEL_UPLOADS"] = "./models"
cameras = []


# Download the MiDaS
# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() and model_type == "MiDaS_small" else torch.device('cpu')
# device = torch.device('cpu')
midas.to(device)
midas.eval()

# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = transforms.dpt_transform
else:
    transform = transforms.small_transform


def applyMatrix4(mat, vec):
    ans = np.matmul(mat, vec.reshape(4, 1))
    ans /= ans[3]
    return ans[:3]

def reconstructPixel(camPose, unprojectMatrix, height, width, row, column):
    aspectRatioRoot = math.sqrt(width / height)
    ndc_x = (column / width * 2 - 1) * aspectRatioRoot
    ndc_y = (1 - row / height * 2) / aspectRatioRoot
    ndc_z = 1
    ndc_w = 1
    ndc = np.array([[ndc_x], [ndc_y], [ndc_z], [ndc_w]])
    rec = applyMatrix4(unprojectMatrix, ndc)
    rec = rec - camPose
    rec /= np.linalg.norm(rec)
    
    return rec

def reconstruct(camPose, unprojectMatrix, height, width):
    # Generate NDC for grid of pixels
    ndc_xy = np.flip(np.dstack(np.meshgrid(np.linspace(1, -1, height), np.linspace(-1, 1, width), indexing="ij")), axis=2)
    
    # Convert square NDC to rectangular, with respect to aspect ratio
    aspectRatioRoot = math.sqrt(width / height)
    ndc_xy[:,:,0] *= aspectRatioRoot
    ndc_xy[:,:,1] /= aspectRatioRoot

    # Add two homogeneous coordinates to prepare for applying unprojection
    ndc_xyzw = np.zeros((height, width, 4, 1))
    ndc_xyzw[:,:,0,0] = ndc_xy[:,:,0]
    ndc_xyzw[:,:,1,0] = ndc_xy[:,:,1]
    ndc_xyzw[:,:,2,0] = 1
    ndc_xyzw[:,:,3,0] = 1

    # Apply unprojection
    ndc_xyzw = ndc_xyzw.reshape(-1, 4).T
    points = unprojectMatrix @ ndc_xyzw
    points /= points[3,:]

    # Remove 4th homogeneous coordinate
    points = points[:3,:]

    # Reshape back to grid
    points = points.T.reshape(height, width, 3)

    # Make reconstructed points relative to camera center
    points -= camPose.reshape(-1)

    # Normalize distance from camera center
    points /= np.linalg.norm(points, axis=2, keepdims=True)

    return points

def loss(x, A, b):
    result = A @ x.reshape((2,1)) - b
    # result = result / np.linalg.norm(result)
    eps = 0.1
    # (abs(result) > eps).sum()
    err = np.sqrt(abs(result)).sum()
    # print(f"x={x} --> loss={err}")
    return err

def add_triangle_if_small(i, j, k, points, triangles):
    # treshhold = 0.3
    # if np.linalg.norm(points[i] - points[j]) > treshhold:
    #     return
    # if np.linalg.norm(points[i] - points[k]) > treshhold:
    #     return
    # if np.linalg.norm(points[j] - points[k]) > treshhold:
    #     return
    triangles.append(np.array([i, j, k], dtype="uint8"))
    triangles.append(np.array([k, j, i], dtype="uint8"))

def add_triangles(ind, points, triangles):
    add_triangle_if_small(ind[0][0], ind[0][1], ind[1][1], points, triangles)
    add_triangle_if_small(ind[0][0], ind[1][0], ind[1][1], points, triangles)

def saveCameraDetails(id, camera):
    with open(f"./images/archive/capture_{id}.txt", "w") as file:
        file.write(f"camPose =\n{camera.camPose}\n\n")
        file.write(f"unprojectMatrix =\n{camera.unprojectMatrix}\n\n")
        file.write(f"camMatrixWorld =\n{camera.camMatrixWorld}\n\n")
        file.write(f"camProjectionMatrix =\n{camera.camProjectionMatrix}\n\n")
        file.write(f"K =\n{camera.K}\n\n")


class Camera:
    def __init__(self, camPose, unprojectMatrix, camMatrixWorld, camProjectionMatrix, K):
        self.camPose = camPose
        self.unprojectMatrix = unprojectMatrix
        self.camMatrixWorld = camMatrixWorld
        self.camProjectionMatrix = camProjectionMatrix
        self.K = K
    
    def T(self):
        return self.camMatrixWorld[0:3, 3]
    def R(self):
        return self.camMatrixWorld[0:3, 0:3]
    def K3(self):
        return self.K[0:3, 0:3]
    
    def F(self, cam2):
        cam1 = self
        T = cam2.T() - cam1.T()
        T_hat = np.cross(np.eye(3), T.reshape(-1))
        R = cam2.R() @ cam1.R().T
        return inv(cam2.K3().T) @ T_hat @ R @ inv(cam1.K3())


# def matchPoints():
#     # Load images
#     image1 = cv2.imread("./images/archive/capture_0.jpg", 0)
#     image2 = cv2.imread("./images/archive/capture_1.jpg", 0)

#     # Provide fundamental matrix
#     F = cameras[0].F(cameras[1])

#     # Detect and match features (You may use your preferred method)
#     # For example, using ORB feature detector and matcher
#     detector = cv2.ORB_create()
#     keypoints1, descriptors1 = detector.detectAndCompute(image1, None)
#     keypoints2, descriptors2 = detector.detectAndCompute(image2, None)

#     matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = matcher.match(descriptors1, descriptors2)

#     # # Selecting good matches based on the fundamental matrix
#     # good_matches = []
#     # for match in matches:
#     #     point1 = keypoints1[match.queryIdx].pt
#     #     point2 = keypoints2[match.trainIdx].pt
#     #     point1 = np.array([point1[0], point1[1], 1.0])
#     #     point2 = np.array([point2[0], point2[1], 1.0])

#     #     # Check the epipolar constraint
#     #     error = abs(np.matmul(point2, np.matmul(F, point1)))
#     #     if error < 0.1:  # Adjust this threshold based on your requirements
#     #         good_matches.append(match)
    
#     # Extract coordinates of matches
#     points1 = np.float32([keypoints1[match.queryIdx].pt for match in matches])
#     points2 = np.float32([keypoints2[match.trainIdx].pt for match in matches])

#     # Use RANSAC to estimate good matches based on fundamental matrix
#     model, inliers = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, ransacReprojThreshold=3.0)

#     # Convert OpenCV's inliers format to Python boolean array
#     inliers = inliers.ravel().tolist()

#     # Filter matches based on RANSAC inliers
#     good_matches = [matches[i] for i in range(len(matches)) if inliers[i] == 1]

#     print(f"{len(good_matches)} good matches out of {len(matches)}")
    
#     # Plot matches and images in original sizes
#     fig, ax = plt.subplots(figsize=(20, 10))

#     # Display the first image
#     ax.imshow(image1, cmap='gray')
#     ax.axis('off')

#     # Calculate the offset for the second image
#     height, width = image1.shape[:2]
#     offset = np.array([width, 0])

#     # Display the second image shifted to the right
#     ax.imshow(image2, cmap='gray', extent=[offset[0], offset[0] + width, height, 0])
#     ax.axis('off')

#     # Plot matches with offsets for the second image
#     for match in good_matches:
#         idx1 = match.queryIdx
#         idx2 = match.trainIdx
#         point1 = keypoints1[idx1].pt
#         point2 = keypoints2[idx2].pt + offset  # Apply offset for the second image

#         # Draw lines connecting the matches
#         ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'c-', lw=1)

#     # Save the image
#     plt.savefig('./images/archive/matches_visualization.png', bbox_inches='tight', pad_inches=0)
#     plt.close()


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload_image", methods=["POST", "GET"])
@cross_origin()
def upload_image():
    if request.method == "POST":
        # if request.form:
        #     image = request.form["image"]
        if request.files:
            captureCount = int(request.form["captureCount"])
            image = request.files["image"]
            camPose = np.array(json.loads(request.form["camPose"])).reshape((3,1))
            unprojectMatrix = np.array(json.loads(request.form["unprojectMatrix"])).reshape((4,4))
            camMatrixWorld = np.array(json.loads(request.form["camMatrixWorld"])).reshape((4,4))
            camProjectionMatrix = np.array(json.loads(request.form["camProjectionMatrix"])).reshape((4,4))
            K = np.array(json.loads(request.form["K"])).reshape((3,4))
            
            camera = Camera(camPose, unprojectMatrix, camMatrixWorld, camProjectionMatrix, K)
            # saveCameraDetails(captureCount, camera)
            cameras.append(camera)

            print("valid image")
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            # image.save(f"./images/archive/capture_{captureCount}.jpg")
            img = cv2.imread(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            imgbatch = transform(img).to(device)

            # if captureCount > 0:
            #     matchPoints()

            with torch.no_grad():
                prediction = midas(imgbatch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size = img.shape[:2], 
                    mode='bicubic', 
                    align_corners=False
                ).squeeze()

                depth = prediction.cpu().numpy()
                depth = 1/depth
                depth = (depth - depth.min()) / (depth.max() - depth.min())

                I8 = (depth * 255.9).astype(np.uint8)
                img = Image.fromarray(I8)
                img.save("./images/depth.jpg")

            # return render_template("upload_image.html", upload_image=image.filename)

            step = 30
            height, width = depth.shape
            
            A = []
            b = []

            # points = np.zeros((height, width, 3), dtype="float32")
            points = reconstruct(camPose, unprojectMatrix, height, width)

            for row in range(0, height):
                for column in range(0, width):
                    if row%step == 0 and column%step == 0:
                        rec = points[row][column].reshape(3,1)                    
                        A.append([rec[1][0] * depth[row][column], rec[1][0]])
                        b.append([-camPose[1][0]])

            # equal initial weighting
            x0 = [5, 5]

            # linear constraint: no points be reconstructed below floor
            lower_bound = np.array(b).reshape(-1)
            upper_bound = np.inf*np.ones(lower_bound.shape)
            constraint = optimize.LinearConstraint(A, lower_bound, upper_bound)

            print("starting optimization")
            result = optimize.minimize(loss, x0, args=(A,b), method="Nelder-Mead", bounds=((0, None), (None, None)))
            m, t = result.x
            print(f"m={m}, t={t} success={result.success} message={result.message} nit={result.nit}")
            depth = m*depth + t

            # Apply depth to point cloud
            depth_repeated = np.repeat(depth[:, :, np.newaxis], 3, axis=2)
            points = np.multiply(points, depth_repeated)

            # Convert from camera frame to world frame
            points += camPose.reshape(-1)

            # Sampling to make the point clould smaller
            samling_rate = 10
            points = points[::samling_rate,::samling_rate].astype("float32")

            points = points.reshape(-1, 3)
            salmpled_points = points.tolist()

            # sampling_height = math.ceil(height / samling_rate)
            # sampling_width = math.ceil(width / samling_rate)

            # triangles = []
            # for i in range(0, sampling_height-1):
            #     for j in range(0, sampling_width-1):
            #         base = i * sampling_width + j
            #         ind = np.array([[base, base + 1],
            #                         [base + sampling_width, base + sampling_width + 1]])

            #         add_triangles(ind, points, triangles)
            
            # triangles = np.stack(triangles, axis=0)
            # mesh_to_glb(points, triangles)

            encoded_data = msgpack.packb(salmpled_points)
            return encoded_data
    return render_template("upload_image.html")
    
@app.route("/images/<filename>")
def send_uploaded_file(filename=""):
    return send_from_directory(app.config["IMAGE_UPLOADS"], filename)
    
@app.route("/models/<filename>")
def send_uploaded_model(filename=""):
    return send_from_directory(app.config["MODEL_UPLOADS"], filename)

if __name__ == '__main__':
    context = ("./keys/fullchain.pem", "./keys/privkey.pem")
    app.run(host = '0.0.0.0', ssl_context = context, port = 5000, debug = True)
