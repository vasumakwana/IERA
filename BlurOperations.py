import cv2

def blur_fun(num, image):
    if num == 1:
        new_img1 = Gaussian_Blur(image)
        return new_img1
    elif num == 2:
        new_img2 = median_Blur(image)
        return new_img2
    elif num == 3:
        new_img1, new_img2 = Gaussian_Blur(image),median_Blur(image)
        return new_img1,new_img2

def Gaussian_Blur(img):

    #reading a image from computer and taking dimensions
    rows, cols = img.shape[:2]

    #gaussian Blur
    #sigma X and  Y: The standard deviation of the Gaussian function controls the amount of blurring.
    output_gaus = cv2.GaussianBlur(img, (5,5), 0)

    return output_gaus

def median_Blur(img):

    #reading a image from computer and taking dimensions
    rows, cols = img.shape[:2]

    #median Blur (reduction of noise)
    output_med = cv2.medianBlur(img, 5)

    return output_med