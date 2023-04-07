import cv2 as cv

cap = cv.VideoCapture(0)
ret, img = cap.read()
cv.imshow("frame",img)
cv.imwrite("imagen_0001.jpg",img)
while True:
	ret, img = cap.read()

	if ret == None: break

	cv.imshow("img",img)
	k = cv.waitKey(1)
	if k==27: break
cap.release()
cv.destroyAllWindows()


