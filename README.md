# COMP425-HW1
## Overview
This project contains the implementation of various fundamental computer vision techniques as part of the Homework 1 assignment for COMP 425/COMP6341 Computer Vision, Winter 2024 session. The assignment focuses on image filtering, edge detection, corner detection, and image subsampling. Specific algorithms implemented include 2D image filtering (excluding convolution), gradient computation, the Harris corner detector, and image downsampling techniques with and without aliasing.

## Key Components
- Image Filtering: Implementation of filter2d function for 2D image filtering.
- Edge Detection: Calculation of gradient images in x and y directions and gradient magnitude using derivative filters.
- Corner Detection: Implementation of the Harris corner detector, including corner response calculation, thresholding, and non-maximum suppression.
- Image Subsampling: Implementation of naive and anti-aliasing downsampling techniques.

## Running the code
To run the individual components of the homework, navigate to the project directory in your terminal and execute the corresponding Python script. Ensure that the images iguana.png, building.jpg, and paint.png are placed in the correct directory as specified in the homework instructions.
- For image filtering: python utils.py
- For edge detection: python edge.py
- For corner detection: python corner.py
- For image subsampling: python downsample.py

Each script will output the results as specified in the homework instructions, including visualization figures and/or console outputs.
