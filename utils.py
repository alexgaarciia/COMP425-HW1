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

    H, W = image.shape[:2]
    if len(image.shape) == 2:
        # gray image
        out = np.zeros((H+2*pad_height, W+2*pad_width))
        out[pad_height: H+pad_height, pad_width: W+pad_width] = image
    else:
        # color image
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

    out = None
    Hi, Wi = image.shape
    Hk, Wk = filter.shape
    out = np.zeros((Hi, Wi))

    image = zero_pad(image, Hk//2, Wk//2)
    for m in range(Hi):
        for n in range(Wi):
            out[m, n] = np.sum((image[m:m+Hk, n:n+Wk])*filter)
    return out
    

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """
    # Sobel x-axis kernel
    kernel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    # Apply the x-derivative filter using the filter2d function
    out = filter2d(img, kernel_x)
    return out


def partial_y(img):
    """ Computes partial y-derivative of input img.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """
    # Sobel y-axis kernel
    kernel_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])

    # Apply the y-derivative filter using the filter2d function
    out = filter2d(img, kernel_y)
    return out
