# Import necessary libraries/functions
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from utils import gaussian_kernel, filter2d


def main():
    # Load the image
    im = imread('paint.jpg').astype('float')
    im = im / 255

    # Number of levels for down-sampling
    N_levels = 5

    # Make a copy of the original image
    im_subsample = im.copy()

    # Naive subsampling, visualize the results on the 1st row
    for i in range(N_levels):
        # Subsample image:
        im_subsample = im_subsample[::2, ::2, :]

        # Visualize the images:
        plt.subplot(2, N_levels, i+1)
        plt.imshow(im_subsample)
        plt.axis('off')

    # Subsampling without aliasing, visualize results on 2nd row
    im_subsample = im.copy()
    for i in range(N_levels):
        # Apply Gaussian filter to smooth the image before subsampling:
        kernel = gaussian_kernel()
        im_smoothed = filter2d(im_subsample, kernel)

        # Subsample the smoothed image:
        im_subsample = im_smoothed[::2, ::2, :]

        # Visualize the images:
        plt.subplot(2, N_levels, N_levels + i + 1)
        plt.imshow(im_subsample)
        plt.axis('off')

    plt.show()


if __name__ == "__main__":
    main()
