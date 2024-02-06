import numpy as np
import matplotlib.pylab as plt
from skimage import io
from utils import gaussian_kernel, filter2d, partial_x, partial_y


def main():
    # Load image
    img = io.imread('iguana.png', as_gray=True)

    # Smooth image with Gaussian kernel
    kernel = gaussian_kernel()
    smoothed_im = filter2d(img, kernel)

    # Compute x and y derivate on smoothed image
    grad_x = partial_x(smoothed_im)
    grad_y = partial_y(smoothed_im)

    # Compute gradient magnitude
    grad_magnitude = np.sqrt(np.square(grad_x) + np.square(grad_y))

    # Visualize results
    plt.figure()

    plt.subplot(1, 3, 1)
    plt.title("Gradient in X direction")
    plt.imshow(grad_x)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Gradient in Y direction")
    plt.imshow(grad_y)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Gradient Magnitude")
    plt.imshow(grad_magnitude)
    plt.axis("off")

    plt.show()
    
if __name__ == "__main__":
    main()

