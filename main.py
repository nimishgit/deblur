import numpy as np
import math
import cv2
from skimage.filters import roberts

# std dev along x axis
sigmaX = 0
# std dev along y axis
sigmaY = 0
t = 0
v = 3
f = 0
x = 0
y = 0

class Deblur:
    def find_equilibrium_factor(self):
        factor1 = v*f*t*math.pi*sigmaX*sigmaY
        val = 2*math.cos(factor1)/v
        self.c = math.log(val)* np.power(x**2+y**2, -6./7)
        return self.c

    def non_linear_deblur_first(self, blurred_image):
        A = [] # background intensity
        img = cv2.imread(blurred_image, 0)
        self.img = img
        x_len = img.shape[0]
        y_len = img.shape[1]
        factor1 = v*f*t*math.pi*sigmaX*sigmaY
        val = (v*math.pi)/math.cos(factor1)
        s = 0
        for x in range(x_len):
            for y in range(y_len):
                s+=self.c*img[x,y](np.exp((x**2+y**2)/2*sigmaX*sigmaY))
        return A + s 

    def robert_operator(self):
        edge_roberts = roberts(self.img)
        return edge_roberts

    def non_linear_deblur(self, img):
        return self.non_linear_deblur_first(img) + self.robert_operator()

