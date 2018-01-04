import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import io 
from skimage.util import img_as_ubyte
from skimage.morphology import closing
from skimage import feature


# Generate noisy image of a square
#im = np.zeros((128, 128))
#im[32:-32, 32:-32] = 1
im = img_as_ubyte(io.imread('img_35.png', as_grey=True))

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(im)
edges2 = feature.canny(im, sigma=3)

# perform morpho "closing" on edges?
#closed = closing(edges2,selem)
closed = closing(edges2)

# display results
fig, (ax3, ax4) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
#                                    sharex=True, sharey=True)

#ax1.imshow(im, cmap=plt.cm.gray)
#ax1.axis('off')
#ax1.set_title('noisy image', fontsize=20)

#ax2.imshow(edges1, cmap=plt.cm.gray)
#ax2.axis('off')
#ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

ax4.imshow(closed, cmap=plt.cm.gray)
ax4.axis('off')

fig.tight_layout()

plt.show()
