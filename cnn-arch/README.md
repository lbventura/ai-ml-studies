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

- The architecture of CNNs - through the usage of both convolution and pooling layers - enables the hierarchical learning of features. Early layers detect simple elements such as edges, textures, and basic shapes. Intermediate layers combine these basic features to recognize more complex structures like corners, contours, and parts of objects. The deeper layers capture high-level semantic information, including entire objects and backgrounds.

- Compared to fully connected networks, CNNs have significantly fewer parameters. This makes the training process computationally more efficient and reduces the risk of overfitting.

- Convolution operations in CNNs ensure that each neuron is connected only to a small, localized region of the input image. This local connectivity allows the network to learn and preserve spatial hierarchies of features, enhancing its ability to understand spatial relationships within the image.

- (Point on pooling layers: 1. create a hierarchical representation by aggregating over features, 2. prioritize the most important features, 3. introduce translation invariance, which allows feature recognition regardless of its position in the image.)

- The generality of features learned in the early layers allows the same CNN architecture to be applied to multiple tasks. For instance, in this assignment, the network is utilized for both image colorization and super-resolution (e.g., enhancing an image's resolution).
