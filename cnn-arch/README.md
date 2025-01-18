# Convolutional Neural Network (CNN) Architecture for Image Colorization

This is a simple CNN architecture for image colorization based on [assignment 2](http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/assignments/assignment2.pdf) of the Neural Networks and Deep Learning course of the University of Toronto. See more information in the course page [here](http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/).

The goal is to colorize grayscale images (e.g, given a black and white image, predict the colors in the image). This is not a straightforward problem because the mapping from grayscale to color is not one-to-one. For example, a grayscale image of a red apple could be colorized in many different ways, depending on the color of the apple.

It is insightful to check the model behavior by visualizing the predictions.

| ![Model Prediction at Epoch 0](examples/unet-0-ex1.png) | ![Model Prediction at Epoch 24](examples/unet-24-ex1.png) |
|:---------------------------------------------------------------:|:---------------------------------------------------------------:|

The image on the left (epoch 0) has a much lower color contrast than the image on the right (epoch 24). The model is learning to represent colors as the training progresses. Another example:

| ![Model Prediction at Epoch 0](examples/unet-0-ex2.png) | ![Model Prediction at Epoch 24](examples/unet-24-ex2.png) |
|:---------------------------------------------------------------:|:---------------------------------------------------------------:|

The image on the left is much lighter than the original image. At epoch 24, the image is more colorful and closer to the original image.

## Why are CNNs used for Image Processing and Computer Vision?

A few points:

- The layered structure of CNNs allows them to learn features hierarchically. The earlier layers learn simple features (e.g., edges, texture, basic shapes). The intermediate layers then combine these simple features to learn more complex features (e.g., corners, contours, parts of objects). The final layers capture high-level semantic information (e.g., entire objects, the background, etc..).
- Compared to a fully connected network, these contain fewer parameters, which makes training computationally more efficient.
- Also compared to a fully connected network, the nature of the convolution operation causes each unit to be connected to only a small region of the input image. This local connectivity allows the network to learn spatial hierarchies of features, described above.
- The same network can be used for multiple tasks, as the early layer features are general. In the assignment, the same network is used for image colorization and super-resolution (e.g, improve an image's resolution).
