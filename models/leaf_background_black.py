import cv2

img = cv2.imread('./images/bacteria_leaf.JPG')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# find the green color
mask_green = cv2.inRange(hsv, (36,0,0), (86,255,255))
# find the brown color
mask_brown = cv2.inRange(hsv, (8, 60, 20), (30, 255, 200))
# find the yellow color in the leaf
mask_yellow = cv2.inRange(hsv, (21, 39, 64), (40, 255, 255))

# find any of the three colors(green or brown or yellow) in the image
mask = cv2.bitwise_or(mask_green, mask_brown)
mask = cv2.bitwise_or(mask, mask_yellow)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(img,img, mask= mask)

cv2.imshow("original", img)
cv2.imshow("final image", res)
cv2.imwrite('./images/rzlt_image.JPG',res)
cv2.waitKey(0)
cv2.destroyAllWindows()

