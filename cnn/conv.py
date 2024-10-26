import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create a sample black and white image
image =  np.array([[0, 1, 1, 1, 1, 0],
    [1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1]], dtype=np.uint8)

image = 255 * np.array([
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0],
], dtype=np.uint8)

# Define the vertical Sobel kernel
vertical_sobel_kernel = np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]])

left_kernel = np.array([[-1, 0, 1],
                                  [-1, 0, 1],
                                  [-1, 0, 1]])


# Define the horizontal Sobel kernel
horizontal_sobel_kernel = np.array([[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]])

right_kernel = np.array([[1, 0, -1],
                                  [1, 0, -1],
                                  [1, 0, -1]])

# Apply the vertical Sobel kernel
vertical_edges = cv2.filter2D(image, -1, left_kernel) # activates when it turns from white to black
print(vertical_edges)
# Apply the horizontal Sobel kernel
horizontal_edges = cv2.filter2D(image, -1, right_kernel) # activates when it turns from black to white
print(horizontal_edges)

# Display the results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Vertical Edges')
plt.imshow(vertical_edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Horizontal Edges')
plt.imshow(horizontal_edges, cmap='gray')
plt.axis('off')

plt.show()
