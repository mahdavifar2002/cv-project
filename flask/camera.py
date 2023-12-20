import json
import numpy as np
from numpy.linalg import inv

class Camera:
    depth = None
    points = None

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

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)