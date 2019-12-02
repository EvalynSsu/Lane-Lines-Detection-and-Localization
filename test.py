import numpy as np
import cv2
import matplotlib.pyplot as plt

a3 = np.array( [[[10,10],[100,10],[100,100],[10,100]]], dtype=np.int32 )
im = np.zeros([240,320],dtype=np.uint8)
cv2.fillPoly( im, a3, 255 )
print(a3.shape)
print(im.shape)

plt.imshow(im)
plt.show()