# Import necessary libraries/functions
import numpy as np


def gaussian_kernel(l=5, sig=1.):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    # Create an array of values that spans the range needed to construct one axis of the kernel.
    # It centers the axis at 0, which is crucial for a symmetric Gaussian kernel.
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)

    # Applies the Gaussian function to every element of the array "ax". This operation generates the 1D Gaussian distribution.
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))

    # Calculates the outer product of the "gauss" array with itself to form a 2D Gaussian kernel. This step effectively
    # spreads the 1D Gaussian distribution across two dimensions.
    kernel = np.outer(gauss, gauss)

    # Finally, it normalizes the kernel so that its sum equals 1. This ensures that applying the kernel to an image does
    # not change the overall brightness of the image.
    return kernel / np.sum(kernel)

    
def zero_pad(image, pad_height, pad_width):
    """ 
    Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """
    # Zero-padding is a common preprocessing step in image processing, and it is used to adjust the spatial size of the
    # image. Here is a breakdown of the function and its components:

    # Extract the original height and width of the image.
    H, W = image.shape[:2]
    if len(image.shape) == 2:
        # If the image is grayscale, initialize an output array filled with zeros.
        # The output array's shape will be the original image's dimensions increased by double the padding dimensions.
        out = np.zeros((H+2*pad_height, W+2*pad_width))

        # Insert the original image in the center of the output array.
        out[pad_height: H+pad_height, pad_width: W+pad_width] = image
    else:
        # If the image is color, the process is similar but includes an additional dimension for the color channels.
        out = np.zeros((H+2*pad_height, W+2*pad_width, 3))
        out[pad_height: H+pad_height, pad_width: W+pad_width, :] = image        

    return out


def filter2d(image, filter):
    """ 
    A simple implementation of image filtering as correlation.
    For simplicity, let us assume the width/height of filter is odd number

    Args:
        image: numpy array of shape (Hi, Wi)
        filter: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    # Check if the image is grayscale or color:
    if len(image.shape) == 2:
        # If the length is two, then it is the case of grayscale image processing:
        Hi, Wi = image.shape  # Image dimensions
        Hk, Wk = filter.shape  # Filter dimensions
        out = np.zeros((Hi, Wi))  # Initialize the output array

        # Pad the image with zeros on all sides for boundary treatment. The padding size is half the filter size,
        # ensuring that the filter 'fits' when applied to the pixels at the image boundaries.
        image = zero_pad(image, Hk//2, Wk//2)

        # Apply the filter to each pixel in the image:
        for m in range(Hi):
            for n in range(Wi):
                out[m, n] = np.sum((image[m:m+Hk, n:n+Wk])*filter)
    else:
        # Otherwise, it is the case of color image processing:
        Hi, Wi, Ci = image.shape  # Image dimensions, including color channels
        out = np.zeros((Hi, Wi, Ci))  # Initialize the output array

        # In this case, it is highly important to process each color channel independently:
        for c in range(Ci):
            image_channel = image[:, :, c]  # Extract one color channel at a time
            image_padded = zero_pad(image_channel, filter.shape[0]//2, filter.shape[1]//2)  # Pad the channel

            # Apply the filter to each pixel in the image:
            for m in range(Hi):
                for n in range(Wi):
                    out[m, n, c] = np.sum(image_padded[m:m+filter.shape[0], n:n+filter.shape[1]] * filter)

    return out


def partial_x(img):
    """ Computes partial x-derivative of input img.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """
    # This function is designed to compute the partial derivative of an image with respect to the x-axis.
    # In this case, the selected filter is the Sobel operator. It is a discrete differentiation operator, computing an
    # approximation of the gradient of the image intensity function. It is applied as follows:

    # 1. Define the Sobel kernel for the x-direction. This kernel is designed to respond strongly to changes in pixel
    # intensity from left to right or right to left.
    kernel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    # 2. Apply this kernel to the image using the 2D filtering operation defined by the function "filter2d". This
    # function correlates the kernel with the image, effectively sliding the kernel across the image and computing the
    # sum of the element-wise product of the kernel and the image regions it covers. This process results in a new image
    # where each pixel value is the result of the kernel application, representing the gradient magnitude in the
    # x-direction at that point.
    out = filter2d(img, kernel_x)

    return out


def partial_y(img):
    """ Computes partial y-derivative of input img.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """
    # This function is designed to compute the partial derivative of an image with respect to the y-axis.
    # In this case, the selected filter is the Sobel operator. It is a discrete differentiation operator, computing an
    # approximation of the gradient of the image intensity function. It is applied as follows:

    # 1. Define the Sobel kernel for the t-direction. This kernel is specifically structured to be sensitive to
    # vertical changes in pixel intensity, enhancing the detection of horizontal edges.
    kernel_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])

    # 2. Apply this kernel to the image using the 2D filtering operation defined by the function "filter2d". This
    # function correlates the kernel with the image, effectively sliding the kernel across the image and computing the
    # sum of the element-wise product of the kernel and the image regions it covers. This process results in a new image
    # where each pixel value is the result of the kernel application, representing the gradient magnitude in the
    # y-direction at that point.
    out = filter2d(img, kernel_y)

    return out
