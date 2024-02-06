import os
import time

from flask import Flask, redirect, jsonify, request, url_for, render_template, flash, send_from_directory
from flask_cors import CORS, cross_origin
import torch
from torchvision.transforms import Compose
import cv2
import json
import msgpack
import math
from PIL import Image
import numpy as np
from numpy.linalg import inv
from scipy import optimize
from gltf_builder import mesh_to_glb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# MIDAS
import midas.utils
from midas.models.midas_net import MidasNet
from midas.models.transforms import Resize, NormalizeImage, PrepareForNet

from camera import Camera


# Global variables
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["IMAGE_UPLOADS"] = "./images"
app.config["MODEL_UPLOADS"] = "./models"
cameras = {}
midasmodel = None
device = None
zoemodel = None


# # Download the MiDaS
# # model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

# midas = torch.hub.load("intel-isl/MiDaS", model_type)

# # Move model to GPU if available
# device = torch.device('cuda') if torch.cuda.is_available() and model_type == "MiDaS_small" else torch.device('cpu')
# # device = torch.device('cpu')
# midas.to(device)
# midas.eval()

# # Input transformation pipeline
# transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

# if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
#     transform = transforms.dpt_transform
# else:
#     transform = transforms.small_transform

def init_midas():
    global midasmodel, device

    # select device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device is {device}")

    midas_model_path = "midas/model.pt"
    # midas_model_path = "midas/midas_v21_small_256.pt"
    midasmodel = MidasNet(midas_model_path, non_negative=True)
    midasmodel.to(device)
    midasmodel.eval()
    return


def init_zoe():
    global zoemodel

    print("started init_zoe()")
    zoemodel = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True)
    device = torch.device("cpu")
    zoemodel = zoemodel.to(device)
    print("finished init_zoe()")


def estimate_midas(img, msize=512):
    # MiDas -v2 forward pass script adapted from https://github.com/intel-isl/MiDaS/tree/v2

    transform = Compose(
        [
            Resize(
                msize,
                msize,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    img_input = transform({"image": img})["image"]

    # Forward pass
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        prediction = midasmodel.forward(sample)

    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # # Normalization
    # depth_min = prediction.min()
    # depth_max = prediction.max()

    # if depth_max - depth_min > np.finfo("float").eps:
    #     prediction = (prediction - depth_min) / (depth_max - depth_min)
    # else:
    #     prediction = 0

    return prediction


def estimate_zoe(img_path):
    image = Image.open("capture.jpg").convert("RGB")  # load
    depth_numpy = zoemodel.infer_pil(image)  # as numpy
    depth_pil = zoemodel.infer_pil(image, output_type="pil")  # as 16-bit PIL Image
    depth_tensor = zoemodel.infer_pil(image, output_type="tensor")  # as torch tensor
    return depth_tensor.numpy()


def read_image(path):
    img = cv2.imread(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img


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
        # file.write(f"depth =\n{camera.depth}\n\n")
        # file.write(camera.toJSON())

def lines_intersection_loss(x, c1, p1, c2, p2):
    direction1 = p1 - c1
    direction2 = p2 - c2
    direction1 /= np.linalg.norm(direction1)
    direction2 /= np.linalg.norm(direction2)
    
    distance1 = np.cross(x-c1, direction1)
    distance2 = np.cross(x-c2, direction2)

    return np.linalg.norm(distance1) + np.linalg.norm(distance2)
    
    err1 = (x[0] - c1[0]) * (c1[1] - p1[1]) - (x[1] - c1[1]) * (c1[0] - p1[0])
    err2 = (x[1] - c1[1]) * (c1[2] - p1[2]) - (x[2] - c1[2]) * (c1[1] - p1[1])
    err3 = (x[0] - c2[0]) * (c2[1] - p2[1]) - (x[1] - c2[1]) * (c2[0] - p2[0])
    err4 = (x[1] - c2[1]) * (c2[2] - p2[2]) - (x[2] - c2[2]) * (c2[1] - p2[1])

    return abs(err1) + abs(err2) + abs(err3) + abs(err4)

def line_parametric(p1, p2):
    def line(t):
        return p1 + t * (p2 - p1)
    return line

def distance_squared(t, line1, line2):
    x1 = line1(t[0])
    x2 = line2(t[1])
    return np.sum((x1 - x2)**2)

def closest_point_skew_lines(p1, p2, p3, p4):
    line1 = line_parametric(p1, p2)
    line2 = line_parametric(p3, p4)

    # Initial guess for t and t' parameters
    initial_guess = [0.0, 0.0]

    # Minimize the distance function
    result = optimize.minimize(distance_squared, initial_guess, args=(line1, line2), method='Powell')

    t_min, t_prime_min = result.x
    closest_point_line1 = line1(t_min)
    closest_point_line2 = line2(t_prime_min)

    midpoint = 0.5 * (closest_point_line1 + closest_point_line2)
    
    return midpoint


def matchPoints(ind_1, ind_2):
    # Load images
    image1 = cv2.imread(f"./images/archive/capture_{ind_1}.jpg", 0)
    image2 = cv2.imread(f"./images/archive/capture_{ind_2}.jpg", 0)

    # Provide fundamental matrix
    F = cameras[ind_1].F(cameras[ind_2])

    # Detect and match features (You may use your preferred method)
    # For example, using ORB feature detector and matcher
    detector = cv2.ORB_create()
    keypoints1, descriptors1 = detector.detectAndCompute(image1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(image2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

    # # Selecting good matches based on the fundamental matrix
    # good_matches = []
    # for match in matches:
    #     point1 = keypoints1[match.queryIdx].pt
    #     point2 = keypoints2[match.trainIdx].pt
    #     point1 = np.array([point1[0], point1[1], 1.0])
    #     point2 = np.array([point2[0], point2[1], 1.0])

    #     # Check the epipolar constraint
    #     error = abs(np.matmul(point2, np.matmul(F, point1)))
    #     if error < 0.1:  # Adjust this threshold based on your requirements
    #         good_matches.append(match)
    
    # Extract coordinates of matches
    points1 = np.float32([keypoints1[match.queryIdx].pt for match in matches])
    points2 = np.float32([keypoints2[match.trainIdx].pt for match in matches])

    # Use RANSAC to estimate good matches based on fundamental matrix
    model, inliers = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, ransacReprojThreshold=1.0)

    # Convert OpenCV's inliers format to Python boolean array
    inliers = inliers.ravel().tolist()

    # Filter matches based on RANSAC inliers
    good_matches = [matches[i] for i in range(len(matches)) if inliers[i] == 1]

    print(f"{len(good_matches)} good matches out of {len(matches)}")
    
    # Plot matches and images in original sizes
    fig, ax = plt.subplots(figsize=(20, 10))

    # Display the first image
    ax.imshow(image1, cmap='gray')
    ax.axis('off')

    # Calculate the offset for the second image
    height, width = image1.shape[:2]
    offset = np.array([width, 0])

    # Display the second image shifted to the right
    ax.imshow(image2, cmap='gray', extent=[offset[0], offset[0] + width, height, 0])
    ax.axis('off')

    # Array to store reconstructed points for good matches
    keypoints = []
    keypoints_coords = []

    # Plot matches with offsets for the second image
    for match in good_matches:
        idx1 = match.queryIdx
        idx2 = match.trainIdx
        point1 = keypoints1[idx1].pt
        point2 = keypoints2[idx2].pt + offset  # Apply offset for the second image

        # Draw lines connecting the matches
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], color=np.random.rand(3), lw=1)


        # Let's do some math here
        point2 -= offset
        rec1 = reconstructPixel(cameras[ind_1].camPose, cameras[ind_1].unprojectMatrix, height, width, point1[1], point1[0]) + cameras[ind_1].camPose
        rec2 = reconstructPixel(cameras[ind_2].camPose, cameras[ind_2].unprojectMatrix, height, width, point2[1], point2[0]) + cameras[ind_2].camPose
        
        # line1 : camPose1 ___ rec1
        # line2 : camPose2 ___ rec2
        c1 = cameras[ind_1].camPose.copy().reshape(3)
        p1 = rec1.reshape(3)
        c2 = cameras[ind_2].camPose.copy().reshape(3)
        p2 = rec2.reshape(3)

        # x0 = [1, 1, 1]
        # result = optimize.minimize(lines_intersection_loss, x0, args=(c1,p1,c2,p2), method="Nelder-Mead")
        # keypoints.append(result.x.tolist())
        midpoint = closest_point_skew_lines(c1, p1, c2, p2)
        keypoints.append(midpoint.tolist())
        keypoints_coords.append([math.floor(point2[1]), math.floor(point2[0])])

        # print(f"merge  -> {result.x}")
        # print(f"midas1 -> {cameras[ind_1].points[math.floor(point1[1])][math.floor(point1[0])]}")
        # print(f"midas2 -> {cameras[ind_2].points[math.floor(point2[1])][math.floor(point2[0])]}")


    # Save the image
    plt.savefig('./images/archive/matches_visualization.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    return keypoints, keypoints_coords


def keypoints_reporojection_loss(x, midas_points, keypoints):
    M = x.reshape(4, 4)
    error_sum = 0

    for i in range(len(keypoints)):
        error = np.linalg.norm(keypoints[i] - M @ midas_points[i])
        error_sum += error ** 2
    
    return error_sum


def refinePoints(points, keypoints, keypoints_coords):
    midas_points = []
    
    # Grab midas points corresponding to keypoints and convert them to homogeneous coordinates
    for i, keypoints_coord in enumerate(keypoints_coords):
        row, column = keypoints_coord
        keypoints[i].append(1)
        midas_points.append(np.append(points[row][column].copy(), 1))
    

    # Initial guess: identity matrix
    x0 = np.eye(4).reshape(-1)

    # Define the constraint: sum of parameters = 1
    constraint_matrix = [[1] * 16]
    constraint_rhs = [1]
    constraint = optimize.LinearConstraint(constraint_matrix, constraint_rhs, constraint_rhs)


    result = optimize.minimize(keypoints_reporojection_loss, x0, args=(midas_points, keypoints), constraints=constraint)
    M = result.x.reshape(4, 4)
    
    height = points.shape[0]
    width = points.shape[1]

    # Add a homogeneous coordinate to prepare for applying transformation matrix M
    points_homogeneous = np.zeros((height, width, 4))
    points_homogeneous[:,:,:3] = points[:,:,:3]
    points_homogeneous[:,:,3] = 1

    # Apply transformation matrix M
    points_homogeneous = points_homogeneous.reshape(-1, 4).T
    points_homogeneous = M @ points_homogeneous
    points_homogeneous /= points_homogeneous[3,:]

    # Remove 4th homogeneous coordinate
    points = points_homogeneous[:3,:]

    # Reshape back to grid
    points = points.T.reshape(height, width, 3)

    return points

    # for row in range(height):
    #     for column in range(width):
    #         point_homo = M @ np.append(points[row][column], 1)
    #         point_homo /= point_homo[3]
    #         points[row][column] = point_homo[:3]


def reportOptimizationResult(result):
    print()
    print("  Optimization Result:")
    print(f"        Optimal value: {result.fun}")
    print(f"   Optimal parameters: {result.x}")
    print(f"        Success state: {result.success}")
    try:
        print(f" Number of iterations: {result.nit}")
    except AttributeError:
        print(f"Number of evaluations: {result.nfev}")
    print(f"              Message: {result.message}")
    print()


def keypoints_depth_loss(x, midas_depth, midas_points, keypoints, camPose):
    error_sum = 0
    m, t = x
    refined_depth = m * np.array(midas_depth) + t

    # Apply depth to point cloud
    depth_repeated = np.repeat(refined_depth[:, np.newaxis], 3, axis=1)
    refined_points = np.multiply(midas_points, depth_repeated)

    # Convert from camera frame to world frame
    refined_points += camPose.reshape(-1)

    for i in range(len(keypoints)):
        err = np.linalg.norm(keypoints[i] - refined_points[i])
        if err > 1:
            err = 1
        error_sum += err ** 2
    
    # print(f"{error_sum / len(keypoints)} | {x}")
    return error_sum


def keypoints_depth_errors(x, midas_depth, midas_points, keypoints, camPose):
    errors = []
    m, t = x
    refined_depth = m * np.array(midas_depth) + t

    # Apply depth to point cloud
    depth_repeated = np.repeat(refined_depth[:, np.newaxis], 3, axis=1)
    refined_points = np.multiply(midas_points, depth_repeated)

    # Convert from camera frame to world frame
    refined_points += camPose.reshape(-1)

    for i in range(len(keypoints)):
        err = np.linalg.norm(keypoints[i] - refined_points[i])
        errors.append(err)
    
    return np.array(errors)

def refineDepth(depth, points, keypoints, keypoints_coords, camPose):
    midas_depth = []
    midas_points = []

    # Grab midas depths corresponding to keypoints
    for keypoints_coord in keypoints_coords:
        row, column = keypoints_coord
        midas_depth.append(depth[row][column].copy())
        midas_points.append(points[row][column].copy())
    
    # Initial guess
    x0 = [1, 0]

    print("starting optimization")
    result = optimize.minimize(keypoints_depth_loss, x0, args=(midas_depth, midas_points, keypoints, camPose), method="Nelder-Mead", bounds=((0, None), (None, None)))
    # result = optimize.least_squares(keypoints_depth_errors, x0, args=(midas_depth, midas_points, keypoints, camPose), bounds=([0, -np.inf], [np.inf, np.inf]))
    m, t = result.x
    reportOptimizationResult(result)
    # print(f"m={m}, t={t} success={result.success} message={result.message} nit={result.nit}")
    depth = m*depth + t

    return depth


def fill_if_interest_dominant(mask, edges, interest_mask, interest_mask_relaxed, threshold_ratio=0.5):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flood_fill_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    the_ultimate_mask = cv2.bitwise_and(interest_mask_relaxed, 255 - edges)
    flood_fill_mask[1:-1, 1:-1] = 255 - the_ultimate_mask
    
    for contour in contours:
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        
        interest_area = cv2.bitwise_and(interest_mask, contour_mask)
        total_interest = np.sum(interest_area > 0)
        total_area = np.sum(contour_mask > 0)
        interest_ratio = total_interest / total_area if total_area > 0 else 0

        if interest_ratio >= threshold_ratio:
            moments = cv2.moments(contour)
            if moments['m00'] == 0:
                continue
            seed_x = int(moments['m10'] / moments['m00'])
            seed_y = int(moments['m01'] / moments['m00'])
            seed_point = (seed_x, seed_y)
            
            if 0 <= seed_x < mask.shape[1] and 0 <= seed_y < mask.shape[0]:
                cv2.floodFill(mask, flood_fill_mask, seed_point, 255)
    
    return mask

def connected_canny(image, sigma=0.13):
    # Compute the median of the single channel pixel intensities
    v = np.median(image)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # Dilate the edges to close the gaps
    kernel = np.ones((3,3), dtype=np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=1)

    return dilated

def detect_floor(rgb_image, gray_image, y, floor_y):
    # Threshold the grayscale image to create a binary mask of points of interest
    _, interest_mask = cv2.threshold(y, floor_y + 0.03, 255, cv2.THRESH_BINARY_INV)
    interest_mask = interest_mask.astype(np.uint8)

    # Threshold the grayscale image to create a binary mask of points of interest
    _, interest_mask_relaxed = cv2.threshold(y, floor_y + 0.06, 255, cv2.THRESH_BINARY_INV)
    interest_mask_relaxed = interest_mask_relaxed.astype(np.uint8)

    # Convert the RGB image to grayscale for Canny edge detection
    gray_rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    edges = connected_canny(gray_rgb_image)

    # Create an empty segmentation mask
    segmentation_mask_density = np.zeros_like(gray_image)

    # Fill the segmentation mask with white in areas where the points of interest are dominant
    segmentation_mask_density = fill_if_interest_dominant(
        segmentation_mask_density, edges, interest_mask, interest_mask_relaxed, threshold_ratio=0.5
    )

    # Vote between three masks
    vote = segmentation_mask_density.astype(int) + interest_mask.astype(int) + interest_mask_relaxed.astype(int)
    vote = (vote >= (2*255)).astype(int) * 255

    # Save the new segmentation mask to a file
    cv2.imwrite('0_edges.png', edges)
    cv2.imwrite('1_mask.png', interest_mask)
    cv2.imwrite('2_mask_relaxed.png', interest_mask_relaxed)
    cv2.imwrite('3_mask_segmentation.png', segmentation_mask_density)
    cv2.imwrite('4_mask_vote.png', vote)
    cv2.imwrite('5_rgb.png', rgb_image)

    # If you want to view the masks
    # cv2.imshow('Original Image', rgb_image)
    # cv2.imshow('Grayscale Image', gray_image)
    # cv2.imshow('Interest Points Mask', interest_mask)
    # cv2.imshow('Canny Edges', edges)
    # cv2.imshow('Segmentation Mask with Density Check', segmentation_mask_density)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # return segmentation_mask_density
    return vote


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
            floor_y = np.array(json.loads(request.form["floor_y"])).reshape(-1)[0]
            print(f"floor_y = {floor_y}")
            
            camera = Camera(camPose, unprojectMatrix, camMatrixWorld, camProjectionMatrix, K)
            cameras[captureCount] = camera

            print("_" * 60)
            # image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            image.save(f"./images/archive/capture_{captureCount}.jpg")

            ## old midas
            # img = cv2.imread(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            # imgbatch = transform(img).to(device)

            # with torch.no_grad():
            #     prediction = midas(imgbatch)
            #     prediction = torch.nn.functional.interpolate(
            #         prediction.unsqueeze(1),
            #         size = img.shape[:2], 
            #         mode='bicubic', 
            #         align_corners=False
            #     ).squeeze()

            #     depth = prediction.cpu().numpy()

            img = read_image(f"./images/archive/capture_{captureCount}.jpg")
            depth = estimate_midas(img, 384)
            
            # start = time.time()
            # depth = estimate_zoe(f"./images/archive/capture_{captureCount}.jpg")
            # end = time.time()
            # print(end - start)

            plt.imsave(f"./images/archive/disparity_{captureCount}.jpg", depth)

            depth = 1/depth
            depth = (depth - depth.min()) / (depth.max() - depth.min())

            # Save the depth map in gray scale
            I8 = (depth * 255.9).astype(np.uint8)
            img = Image.fromarray(I8)
            # img.save("./images/depth.jpg")
            img.save(f"./images/archive/depth_{captureCount}.jpg")

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
                        b.append([-camPose[1][0] + floor_y])

            # equal initial weighting
            x0 = [5, 5]

            # linear constraint: no points be reconstructed below floor
            lower_bound = np.array(b).reshape(-1)
            upper_bound = np.inf*np.ones(lower_bound.shape)
            constraint = optimize.LinearConstraint(A, lower_bound, upper_bound)

            # Refine midas depth map based WebXR floor detection
            print("starting optimization")
            result = optimize.minimize(loss, x0, args=(A,b), method="Nelder-Mead", bounds=((0, None), (None, None)))
            m, t = result.x
            # print(f"m={m}, t={t} success={result.success} message={result.message} nit={result.nit}")
            reportOptimizationResult(result)
            depth = m*depth + t

            # Refine midas depth map based on 3d reconstructed keypoints
            keypoints = []
            # if captureCount > 0:
            #     keypoints, keypoints_coords = matchPoints(captureCount-1, captureCount)
            #     depth = refineDepth(depth, points, keypoints, keypoints_coords, camPose)

            # Apply depth to point cloud
            depth_repeated = np.repeat(depth[:, :, np.newaxis], 3, axis=2)
            points = np.multiply(points, depth_repeated)

            # # Move points close to surface a little bit below surface
            # for row in range(0, height):
            #     for column in range(0, width):
            #         point = points[row][column]
            #         deltaY = point[1] + camPose.reshape(-1)[1]
            #         if deltaY < 0.:
            #             point *= 1 + (deltaY / point[1])
            #             point *= 1 + (0.05 / point[1])


            # Convert from camera frame to world frame
            points += camPose.reshape(-1)

            # Save the y-coordinate of each reconstructed pixel (zero or below resresents the surface)
            y = points[:,:,1].copy()
            y = y / (y.max() - y.min())
            y = np.clip(y, 0, None)

            gray_I8 = (y * 255.9).astype(np.uint8)
            gray_img = Image.fromarray(gray_I8)
            gray_img.save(f"./images/archive/y_{captureCount}.jpg")

            rgb = cv2.imread(f"./images/archive/capture_{captureCount}.jpg")
            # mask = detect_floor(rgb, gray_I8, y, floor_y)
            
            # for row in range(0, height):
            #     for column in range(0, width):
            #         if mask[row][column]:
            #             point = points[row][column]
            #             camPoseFlat = camPose.reshape(-1)
            #             vec = point - camPoseFlat
            #             points[row][column] =  vec * (1 - (point[1] - floor_y)/vec[1]) + camPoseFlat
            
            # Move points 6cm below floor
            points[:,:,1] -= 0.02

            # # Refine midas 3d points based on 3d reconstructed keypoints
            # if captureCount > 0:
            #     keypoints, keypoints_coords = matchPoints(captureCount-1, captureCount)
            #     points = refinePoints(points, keypoints, keypoints_coords)

            # Save full depth map and point cloud in camera object
            camera.depth = depth
            camera.points = points.copy()
            saveCameraDetails(captureCount, camera)

            # Sampling to make the point clould smaller
            sampling_rate = 10
            points = points[::sampling_rate,::sampling_rate].astype("float32")

            # # Flatten 2D array
            points_flat = points.reshape(-1, 3)
            
            # Convert numpy array to list
            salmpled_points = points.tolist()
            salmpled_points_flat = points_flat.tolist()

            # sampling_height = math.ceil(height / sampling_rate)
            # sampling_width = math.ceil(width / sampling_rate)

            # triangles = []
            # for i in range(0, sampling_height-1):
            #     for j in range(0, sampling_width-1):
            #         base = i * sampling_width + j
            #         ind = np.array([[base, base + 1],
            #                         [base + sampling_width, base + sampling_width + 1]])

            #         add_triangles(ind, points, triangles)
            
            # triangles = np.stack(triangles, axis=0)
            # mesh_to_glb(points, triangles)
            
            encoded_data = msgpack.packb([salmpled_points_flat, salmpled_points, keypoints])
            return encoded_data
    return render_template("upload_image.html")
    
@app.route("/images/<filename>")
def send_uploaded_file(filename=""):
    return send_from_directory(app.config["IMAGE_UPLOADS"], filename)
    
@app.route("/models/<filename>")
def send_uploaded_model(filename=""):
    return send_from_directory(app.config["MODEL_UPLOADS"], filename)

if __name__ == '__main__':
    init_midas()
    context = ("./keys/fullchain.pem", "./keys/privkey.pem")
    app.run(host = '0.0.0.0', ssl_context = context, port = 5000, debug = True)
