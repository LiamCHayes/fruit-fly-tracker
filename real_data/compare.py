import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


path = "grapes_frames/minpooled_resized/grapes_0135.jpeg"
path_output = "grapes_frames/Deep_results_prelim/frame000000.png"

img1 = Image.open(path)
img2 = Image.open(path_output)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
ax1.imshow(np.array(img2))
ax2.imshow(np.array(img1))

ax1.axis('off')
ax2.axis('off')

plt.show()
