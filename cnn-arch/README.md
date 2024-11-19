# Convolutional Neural Network Architecture for Image Colorization

This is a simple CNN architecture for image colorization based on assignment 2 of the Neural Networks and Deep Learning course of the University of Toronto. See details [here](http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/), section "Assignments".

The goal is to colorize grayscale images (e.g, given a black and white image, predict the colors in the image). This is not a straightforward problem because the mapping from grayscale to color is not one-to-one. For example, a grayscale image of a red apple could be colorized in many different ways, depending on the color of the apple.

It is insightful to check the model behavior by visualizing the predictions.

| ![Model Prediction at Epoch 0](examples/unet-0-ex1.png) | ![Model Prediction at Epoch 24](examples/unet-24-ex1.png) |
|:---------------------------------------------------------------:|:---------------------------------------------------------------:|

The image on the left (epoch 0) has a much lower color contrast than the image on the right (epoch 24). The model is learning to represent colors as the training progresses. Another example:

| ![Model Prediction at Epoch 0](examples/unet-0-ex2.png) | ![Model Prediction at Epoch 24](examples/unet-24-ex2.png) |
|:---------------------------------------------------------------:|:---------------------------------------------------------------:|

The image on the left is much lighter than the original image. At epoch 24, the image is more colorful and closer to the original image.
