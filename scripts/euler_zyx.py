import math

import numpy as np

def eulerZYX(z=0, y=0, x=0):
    Ms = np.eye(3)
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms = np.array(
            [[1, 0, 0],
             [0, cosx, -sinx],
             [0, sinx, cosx]]).dot(Ms)
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms = np.array(
            [[cosy, 0, siny],
             [0, 1, 0],
             [-siny, 0, cosy]]).dot(Ms)
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms = np.array(
            [[cosz, -sinz, 0],
             [sinz, cosz, 0],
             [0, 0, 1]]).dot(Ms)
    return Ms

