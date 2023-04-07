# Para trabajar con un entorno virtual en un cuaderno de jupyter se instala en el kernel
# el entorno virtual de la sieguiente manera:
    # ipython kernel install --user --name=venv
# Después se abre un cuaderno cualquiera, y se clica en los siguientes pasos:
    # Kernel --> Change kernel --> "name_kernel"
# En la siguiente página viene detallado
    # https://es.acervolima.com/uso-de-jupyter-notebook-en-un-entorno-virtual/
##########################################################################################
# Se usa la librería dlib y opencv para mostrar los 68 puntos característicos del rostro
import cv2 as cv
import dlib
import numpy as np

# Se abre la cámara para leer la imagen
cap = cv.VideoCapture(0)

# Se crea el modelo para detectar los 68 puntos
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Se asignan los colores que de los puntos a dibujar
b,g,r = int(np.random.rand()*255),int(np.random.rand()*255),int(np.random.rand()*255)
bb,gb,rb = int(np.random.rand()*255),int(np.random.rand()*255),int(np.random.rand()*255)


# Se abre el bucle while para mostrar los resultados por pantalla
while True:
	ret, img = cap.read()
	if ret == False: break

	img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

	# Se usa el modelo aplicado a la imagen en escala de grises
	faces = detector(img_gray)
	# se abre un bucle for para identificar los puntos
	for face in faces:
		# Se predicen los puntos con la imagen obtenida y el modelo
		landmark = predictor(img_gray,face)
		# Se identifican los 68 puntos
		landmark_points = []
		for n in range(0,68):
			x = landmark.part(n).x
			y = landmark.part(n).y
			landmark_points.append((x,y))
			cv.circle(img,(x,y), 3, (b,g,r),-1)

		points = np.array(landmark_points,np.int32)
		# Se detectan los puntos externos para hacer un detector de cara también
		hull = cv.convexHull(points)
		rect = cv.boundingRect(hull)
		(xR,yR,wR,hR) = rect
		cv.rectangle(img,(xR,yR),(xR+wR,yR+hR),(bb,gb,rb),2)
	cv.imshow('frame',img)
	key = cv.waitKey(1)
	if key == 27: break

cap.release()
cv.destroyAllWindows()