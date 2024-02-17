# Import necessary libraries/functions
import numpy as np
from utils import filter2d, partial_x, partial_y
from skimage.feature import peak_local_max
from skimage.io import imread
import matplotlib.pyplot as plt


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """
    # In order to compute the Harris Corner Detection Algorithm, there are some steps that must be followed:
    # STEP 1: Compute x and y derivatives of the image
    Ix = partial_x(img)
    Iy = partial_y(img)

    # STEP 2: Compute products of derivatives at every pixel
    Ixx = Ix**2
    Ixy = Ix*Iy
    Iyy = Iy**2

    # STEP 3: Apply a filter (uniform here for simplicity) to weighted derivatives to obtain a smooth representation
    # of gradient changes over the window
    kernel_value = 1 / (window_size*window_size)
    kernel = np.full((window_size, window_size), kernel_value)
    Sxx = filter2d(Ixx, kernel)
    Sxy = filter2d(Ixy, kernel)
    Syy = filter2d(Iyy, kernel)

    # STEP 4: Compute the Harris corner response at each pixel. The formula assesses the likelihood of a pixel being
    # a corner based on the local gradient distribution
    detM = Sxx * Syy - Sxy**2
    traceM = Sxx + Syy
    response = detM - k * traceM**2

    return response


def main():
    # Load image
    img = imread('building.jpg', as_gray=True)

    # Compute Harris corner response
    response = harris_corners(img)

    # Threshold on response
    threshold = np.max(response) * 0.001
    response_thresholded = response > threshold

    # Perform non-max suppression by finding peak local maximum
    coordinates = peak_local_max(response, threshold_abs=threshold)

    # Visualize results
    plt.figure(figsize=(15, 5))

    # Visualize response map before thresholding
    plt.subplot(1, 3, 1)
    plt.imshow(response, cmap='hot')
    plt.title('Harris Response Map')
    plt.axis('off')

    # Visualize response map after thresholding
    plt.subplot(1, 3, 2)
    plt.imshow(response_thresholded, cmap='hot')
    plt.title('Thresholded Response Map')
    plt.axis('off')

    # Visualize detected corners on the image
    plt.subplot(1, 3, 3)
    plt.imshow(img, cmap='gray')
    plt.scatter(coordinates[:, 1], coordinates[:, 0], s=10, color='red', marker='x')
    plt.title('Detected Corners')
    plt.axis('off')

    plt.show()


if __name__ == "__main__":
    main()
