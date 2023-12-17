import cv2

img_gray = cv2.imread("D:/vscode/SuperGluePretrainedNetwork-master/assets/scannet_sample_images/S__258727957.jpg", 0)
img_sobel_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
img_sobel_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
imgs = cv2.hconcat([img_sobel_x, img_sobel_y])
cv2.imshow("sample", imgs)
cv2.waitKey(0)
