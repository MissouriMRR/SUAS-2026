import numpy as np
import cv2

# Coordinate system:
#  __ +y axis
# |
# +x axis

def calculate_padding(img1, img2, r12):
    # Pre-defined arbitrary offset r12: img1 top left corner - img2 top left corner
    """
    Calculates padding for adding img1 to img2
    r12 = (offset_x, offset_y) = img1_topleft - img2_topleft
    """
    
    # Only need to pad img2
    
    # img2 top left padding:
    # Only pad top left if some axis of r12 is negative
    img2_top_padding = max(0, -r12[0])
    img2_left_padding = max(0, -r12[1])
    
    # Only pad bottom right if overlayed img1 will go beyond the bottom/right of img2
    img2_bottom_padding = max(0, img1.shape[0] - img2.shape[0] + r12[0])
    img2_right_padding = max(0, img1.shape[1] - img2.shape[1] + r12[1])
    
    return img2_top_padding, img2_bottom_padding, img2_left_padding, img2_right_padding


def pad_image(img, padding):
    value = np.array([0, 0, 0, np.inf])  # Zero for color channels, inf for depth channel
    padded = cv2.copyMakeBorder(img, int(padding[0]), int(padding[1]), int(padding[2]), int(padding[3]), cv2.BORDER_CONSTANT, value=value)
    
    return padded


def alpha_over(img1, img2, alpha_channel):
    """
    Overlays img1 on top of img2 using alpha_channel
    """
   
    
    overlayed = img1 * alpha_channel[:,:,None] + (1 - alpha_channel[:,:,None]) * img2
    
    return overlayed


def rgb2rgba(img):
    img_alpha = np.ones((img.shape[0], img.shape[1])) * 255
    
    img_w_alpha = np.dstack((img, img_alpha))
    
    return img_w_alpha


def distance_alpha(distance1, distance2, feather_width):
    # Sigmoid function
    
    return (np.tanh(4 * (distance1 - distance2) / feather_width) + 1) / 2


def offset_overlay(new_img, base_img, new_location, feather_width):
    padding = calculate_padding(new_img, base_img, new_location)
    
    padded = pad_image(base_img, padding)
    
    # New offset relative to the padded image
    new_offset = np.maximum(np.array([0, 0]), new_location)
    
    # calculate the bottom left corner of the region to overlay on
    bottom_left = new_offset + np.array(new_img.shape[:2])
    
    # Get the section of the padded image where img1 will be overlayed
    padded_section = padded[int(new_offset[0]):int(bottom_left[0]), int(new_offset[1]):int(bottom_left[1]), :]
    
    alpha_channel = distance_alpha(new_img[:, :, 3], padded_section[:, :, 3], feather_width)
    
    overlayed_section = alpha_over(new_img, padded_section, alpha_channel)
    
    # Write the overlayed section back to the image
    padded[int(new_offset[0]):int(bottom_left[0]), int(new_offset[1]):int(bottom_left[1]), :] = overlayed_section
    
    return padded


def main():
    img1 = cv2.imread("test.png", cv2.IMREAD_UNCHANGED)
    
    img2 = cv2.imread("goatthing.jpg")
    img2 = rgb2rgba(img2)
    
    offset = np.array([-200, 50])
    
    padded = offset_overlay(img1, img2, offset)
    
    cv2.imwrite("padded.png", padded)


if __name__ == "__main__":
    main()