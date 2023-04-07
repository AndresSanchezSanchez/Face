# Se usará la librería de opencv y mediapipe para contar los parpadeos de una 
# persona
# Se usarán como apollo la librería collection y la de matplotlib
import cv2 as cv
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Con la siguiente función se puede aplicar cierto color a la zona de los ojos
# para poder representar mejor el resultado en el parpadeo

def drawing_output(frame, coodinates_left_eye, 
	coodinates_right_eye,blink_counter):
	
	aux_image = np.zeros_like(frame)
	countours1 = np.array([coodinates_left_eye])
	countours2 = np.array([coodinates_right_eye])
	# La función cv2.fillPoly rellena el área delimitada por uno o más 
	# polígonos.
	# cv2.fillPoly(img, pts, color, lineType = LINE_8, shift = 0, 
		# offset = Point())

		# img = Imagen.
		# pts = Matriz de polígonos donde cada polígono se representa 
			# como una matriz de puntos.
		# color = color del polígono.
		# lineType = Tipo de los límites del polígono. Ver tipos de línea
		# shift = Número de bits fraccionarios en las coordenadas del vértice.
		# offset = 	Desplazamiento opcional de todos los puntos de los contornos.
	cv.fillPoly(aux_image, pts=[countours1],color=(0,0,255))
	cv.fillPoly(aux_image, pts=[countours2],color=(0,0,255))
	# La función cv2.addWeighted sirve para sumar dos imágenes con pesos 
	# diferentes o  iguales
	# cv2.addWeighted(src1 ,alpha, src2, beta, gamma, dst, dtype = -1)
		# src1 = primera matriz de entrada.
		# alpha = peso de los primeros elementos de la matriz.
		# src2 = segunda matriz de entrada del mismo tamaño y número de 
			# canal que src1.
		# beta = peso de los elementos de la segunda matriz.
		# gamma = escalar agregado a cada suma.
		# dst = matriz de salida que tiene el mismo tamaño y número 
			# de canales que las matrices de entrada.
		# dtype = profundidad opcional de la matriz de salida; 
			# cuando ambas matrices de entrada tienen la misma profundidad, 
			# dtype se puede establecer en -1, que será 
			# equivalente a src1. depth() .
	output = cv.addWeighted(frame,1, aux_image,0.7,1)

	# Después se crean dos rectángulos donde se pone un texto y el número de
	# parpadeos
	cv.rectangle(output,(0,0),(200,50),(255,0,40),-1)
	cv.rectangle(output,(202,0),(265,50),(255,0,40),2)
	cv.putText(output,"Num. Parpadeos", (10,30),cv.FONT_HERSHEY_SIMPLEX,
		0.7,(0,255,113),2)
	cv.putText(output,"{}".format(blink_counter), (220,30),
		cv.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,113),2)

	return output

# Con la función eye_aspect_ratio sirve para calcular la relación de aspecto 
# de los ojos que según esas variacios se podrá ver si ha habido un parpadeo
def eye_aspect_ratio(coodinates):
	d_A = np.linalg.norm(np.array(coodinates[1])-np.array(coodinates[5]))
	d_B = np.linalg.norm(np.array(coodinates[2])-np.array(coodinates[4]))
	d_C = np.linalg.norm(np.array(coodinates[0])-np.array(coodinates[3]))

	return (d_A+d_B)/(2*d_C)

# Con la siguiente función se grafica en tiempo real la relación de aspecto
# de los ojos, variando el eje de abcisas
# def plotting_ear(pts_ear,line1):
# 	global figure
# 	pts = np.linspace(0,1,64)
# 	if line1 == []:
# 		plt.style.use("ggplot")
# 		plt.ion()

# 		figure, ax = plt.subplots()
# 		line1 = ax.plot(pts,pts_ear)
# 		plt.xlim(0.1,0.4)
# 		plt.ylim(0,1)
# 		plt.ylabel("EAR",fonsize=9)
# 	else:
# 		line1.set_ydata(pts_ear)
# 		figure.canvas.draw()
# 		figure.canvas.fush_events()

# 	return line1

# Después de las funciones se codifica el programa

cap = cv.VideoCapture(0)

# Se usa el modelo preentrenado para detectar los puntos de la cara
mp_face_mesh = mp.solutions.face_mesh
# Se idenifican los puntos indices de los ojos de la persona que está
# en fente
index_left_eye = [33,160,158,133,153,144]
index_right_eye = [362,385,387,263,373,380]
# Se crean una serei de variables auxiliares que hacen las veces de 
# humbral, nímero de frames en cada momento y los conteos, entre otros
EAR_THRESH = 0.24
NUM_FRAMES = 2
aux_counter = 0
blink_counter = 0
line1 = []
# Una vez que un deque de longitud limitada está lleno, cuando se 
# agregan nuevos elementos, se descarta una cantidad correspondiente 
# de elementos del extremo opuesto.
pts_ear = deque(maxlen = 64)
i = 0
# Con este modelo se puede determinar esos puntos con las siguientes variables
# de entrada
# mediapipe.solutions.face_mesh.FaceMesh(tatic_image_mode=True, 
	# max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
	
	# static_image_mode=True,
    # max_num_faces=1,
    # refine_landmarks=True,
    # min_detection_confidence=0.5
with mp_face_mesh.FaceMesh(
	static_image_mode = False,
	max_num_faces = 1) as face_mesh:

	
	while True:
		ret, frame = cap.read()
		if ret == False: break
		frame = cv.flip(frame,1)
		height, width,_ = frame.shape
		# Para procesar la imagen se necesita la imagen en escala RGB
		frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
		results = face_mesh.process(frame_rgb)


		# Se inicializan las coordenadas en un vector vacio
		coodinates_left_eye = []
		coodinates_right_eye = []
		# Se pone una condición para asegurar que haya una cara
		if results.multi_face_landmarks is not None:
			for face_landmarks in results.multi_face_landmarks:
				for index in index_left_eye:
					# Hay que añadir .landmark después para que propocione 
					# los datos como lista
					x = int(face_landmarks.landmark[index].x*width)
					y = int(face_landmarks.landmark[index].y*height)
					coodinates_left_eye.append([x,y])
					cv.circle(frame,(x,y),2,(0,255,255),1)
					cv.circle(frame,(x,y),1,(128,50,255),1)
				for index in index_right_eye:
					# Hay que añadir .landmark después para que propocione 
					# los datos como lista
					x = int(face_landmarks.landmark[index].x*width)
					y = int(face_landmarks.landmark[index].y*height)
					coodinates_right_eye.append([x,y])
					cv.circle(frame,(x,y),2,(128,50,255),1)
					cv.circle(frame,(x,y),1,(0,255,255),1)

				ear_left_eye = eye_aspect_ratio(coodinates_left_eye)
				ear_right_eye = eye_aspect_ratio(coodinates_right_eye)
				# Se calcula la realción de especto para determinar el humbral
				ear = (ear_left_eye+ear_right_eye)/2
				print(ear)

				# Condición para ojo cerrado
				if ear < EAR_THRESH:
					aux_counter += 1
				else:
					if aux_counter >= NUM_FRAMES:
						aux_counter = 0
						blink_counter += 1

				# Se tratan de obtener esas coordenadas para pintar el resultado
				frame = drawing_output(frame,coodinates_left_eye,
					coodinates_right_eye, blink_counter)
				pts_ear.append(ear)
				# Se representan menos de 70 puntos
				# if i>70:
				# 	line1 = plotting_ear(pts_ear,line1)
				# i += 1

			cv.imshow("Frame",frame)
			cv.imshow("frame_rgb",frame_rgb)
			k = cv.waitKey(1) & 0xFF
			if k == 27: break

	cap.release()
	cv.destroyAllWindows()