import os

from flask import Flask, redirect, jsonify, request, url_for, render_template, flash, send_from_directory
from flask_cors import CORS, cross_origin
import torch
import cv2
import json
import math
from PIL import Image
import numpy as np
from scipy import optimize
from gltf_builder import mesh_to_glb

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["IMAGE_UPLOADS"] = "./images"
app.config["MODEL_UPLOADS"] = "./models"


# Download the MiDaS
model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
    x = vec[0][0]
    y = vec[1][0]
    z = vec[2][0]
    e = list(mat.T.flatten())

    w = 1 / ( e[ 3 ] * x + e[ 7 ] * y + e[ 11 ] * z + e[ 15 ] )
    x_new = ( e[ 0 ] * x + e[ 4 ] * y + e[ 8 ] * z + e[ 12 ] ) * w
    y_new = ( e[ 1 ] * x + e[ 5 ] * y + e[ 9 ] * z + e[ 13 ] ) * w
    z_new = ( e[ 2 ] * x + e[ 6 ] * y + e[ 10 ] * z + e[ 14 ] ) * w

    return np.array([[x_new], [y_new], [z_new]]).tolist()

def reconstructPixel(camPose, unprojectMatrix, height, width, row, column):
    aspectRatioRoot = math.sqrt(width / height)
    ndc_x = (column / width * 2 - 1) * aspectRatioRoot
    ndc_y = (1 - row / height * 2) / aspectRatioRoot
    ndc_z = 1
    ndc = np.array([[ndc_x], [ndc_y], [ndc_z]])
    rec = applyMatrix4(unprojectMatrix, ndc)
    rec = rec - camPose
    rec /= np.linalg.norm(rec)
    return rec

def loss(x, A, b):
    result = A @ x.reshape((2,1)) - b
    # result = result / np.linalg.norm(result)
    eps = 0.1
    # (abs(result) > eps).sum()
    err = np.sqrt(abs(result)).sum()
    # print(f"x={x} --> loss={err}")
    return err

def add_triangle_if_small(i, j, k, points, triangles):
    treshhold = 0.3
    if np.linalg.norm(points[i] - points[j]) > treshhold:
        return
    if np.linalg.norm(points[i] - points[k]) > treshhold:
        return
    if np.linalg.norm(points[j] - points[k]) > treshhold:
        return
    triangles.append(np.array([i, j, k], dtype="uint8"))
    triangles.append(np.array([k, j, i], dtype="uint8"))

def add_triangles(ind, points, triangles):
    add_triangle_if_small(ind[0][0], ind[0][1], ind[1][1], points, triangles)
    add_triangle_if_small(ind[0][0], ind[1][0], ind[1][1], points, triangles)


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
            image = request.files["image"]
            camPose = np.array(json.loads(request.form["camPose"])).reshape((3,1))
            unprojectMatrix = np.array(json.loads(request.form["unprojectMatrix"])).reshape((4,4))
            
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            img = cv2.imread(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            imgbatch = transform(img).to(device)

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
                img.save("./images/depth.png")

            # return render_template("upload_image.html", upload_image=image.filename)
            print("valid image")

            step = 30
            height, width = depth.shape
            sampling_height = math.ceil(height / step)
            sampling_width = math.ceil(width / step)
            

            A = []
            b = []

            points = np.zeros((sampling_height, sampling_width, 3), dtype="float32")

            for row in range(0, height, step):
                for column in range(0, width, step):
                    rec = reconstructPixel(camPose, unprojectMatrix, height, width, row, column)
                    points[row//step][column//step] = rec.reshape(3)
                    A.append([rec[1][0] * depth[row][column], rec[1][0]])
                    b.append([-camPose[1][0]])

            # equal initial weighting
            x0 = [5, 5]
            result = optimize.minimize(loss, x0, args=(A,b), method="Nelder-Mead", bounds=((0, None), (None, None)))
            m, t = result.x
            print(f"m={m}, t={t} success={result.success} message={result.message} nit={result.nit}")
            depth = m*depth + t

            for row in range(0, height, step):
                for column in range(0, width, step):
                    points[row//step][column//step] *= depth[row][column]
                    points[row//step][column//step] += camPose.reshape(3)

            points = points.reshape(sampling_height*sampling_width, 3)

            # triangles = np.zeros((4*(sampling_height-1)*(sampling_width-1), 3), dtype="uint8")
            # triangles = [np.empty((0, 3), dtype="uint8")]
            triangles = []
            for i in range(0, sampling_height-1):
                for j in range(0, sampling_width-1):
                    base = i * sampling_width + j
                    ind = np.array([[base, base + 1],
                                    [base + sampling_width, base + sampling_width + 1]])

                    add_triangles(ind, points, triangles)
            
            triangles = np.stack(triangles, axis=0)
    
            # mesh_to_glb(points, triangles)

            return jsonify(depth[::step,::step].tolist())
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
