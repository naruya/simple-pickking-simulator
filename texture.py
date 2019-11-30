import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

a = np.ones((160, 160, 4)).astype(np.uint8) * 255
a[:,:,0] = 128
a[:,:,1] = 128
a[:,:,2] = 128
a[0:80, 20:140, 3] = 0
plt.imshow(a)
plt.show()
cv2.imwrite("src/robot.png", a)


import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

a = np.ones((600, 1000, 4)).astype(np.uint8) * 255
a[:,:,0] = 239
a[:,:,1] = 239
a[:300,:,0] = 191
a[:300,:,1] = 191
plt.imshow(a)
plt.show()
cv2.imwrite("src/shelf.png", a)