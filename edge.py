import numpy as np
import matplotlib.pylab as plt
from skimage import io
from utils import gaussian_kernel, filter2d, partial_x, partial_y

def main():
    # Load image
    img = io.imread('iguana.png', as_gray=True)

    ### YOUR CODE HERE

    # Smooth image with Gaussian kernel

    # Compute x and y derivate on smoothed image

    # Compute gradient magnitude

    # Visualize results
    
    ### END YOUR CODE
    
if __name__ == "__main__":
    main()

