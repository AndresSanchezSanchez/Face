# Con la librería de opencv y dlib se extrae la cara de una imagen
# Entorno virtual en windows "face_swapping" en la dirección:
# C:\Users\andressanchez\.conda\envs (Windows)
# /home/grasshopper41/anaconda3/envs/env_dlib (Linux)
# El entorno virtual está en python 3.6
# Para windows crear un entorno virtual en anaconda
# También hay que instalar Cmake (pip install cmake)
	# 1. create new environment conda create -n env_dlib python=3.6
	# 2. activate enviroment conda activate env_dlib
	# 3. install dlib conda install -c conda-forge dlib
import cv2 as cv
import numpy as np
import dlib

# Se carga la imagen con el rostro y se redimensiona a 396x489
img = cv.pyrDown(cv.imread("image/Paul_Walker.jpg"))
print(img.shape)
# img = cv.resize(img,(396,489))
img = cv.pyrDown(img)
img_copy = img.copy()
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# Se crea una máscara que se usará más adelante
mask = np.zeros_like(img_gray)

# Se crea un modelo para detectar los puntos faciales y des pues el predictor del mismo
# Metthod :: get_frontal_face_detector() devuelve el detector de rostros predeterminado
# get_frontal_face_detector()
detector = dlib.get_frontal_face_detector()
# Se copia el archivo shape_predictor_68_face_landmarks.dat en el mismo directorio
# La función dlib.shape_predictor() ejecuta el predictor de forma en la imagen de entrada 
# y devuelve una única detección_de_objetos_completa.
# Este objeto es una herramienta que toma una región de imagen que contiene algún objeto 
# y genera un conjunto de ubicaciones de puntos que definen la pose del objeto. 
# El ejemplo clásico de esto es la predicción de la pose del rostro humano, donde 
# toma una imagen de un rostro humano como entrada y se espera que identifique las 
# ubicaciones de puntos de referencia faciales importantes, como las comisuras de la 
# boca y los ojos, la punta de la nariz, etc. adelante.
# class :: dlib.shape_predictor(image, box)
	# image = es un ndarray numpy que contiene una imagen RGB o en escala de grises de 8 bits. 
	# box = es el cuadro delimitador para comenzar la predicción de la forma en el interior.
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Se alamcenan los puntos predichos en la varaible faces con el modelo
faces = detector(img_gray)
# Para la rostro humano se tieen 68 puntos y con este bucle se pintan todos los puntos en la
# imagen del rostro usada
for face in faces:
	landmarks = predictor(img_gray, face)
	landmarks_points = []
	for n in range(0,68):
		x = landmarks.part(n).x
		y = landmarks.part(n).y
		landmarks_points.append((x,y))
		cv.circle(img, (x,y),3,(255,75,255),-1)

# Se convierten los puntos en un array
points = np.array(landmarks_points,np.int32)
# La función cv2.convexHull() encuentra el casco convexo de un conjunto de puntos, es decir
# encuentra la envolvente convexa de un conjunto de puntos 2D utilizando el 
# algoritmo de Sklansky
# cv2.convexHull(points, hull, clockwise = false, returnPoints = true)
	# points = 	Introduzca un conjunto de puntos 2D, almacenado en std::vector o Mat.
	# hull = 	Salida casco convexo. Es un vector entero de índices o un vector de puntos.
		# En el primer caso, los elementos del casco son índices basados 
		# ​​en 0 de los puntos del casco convexo en la matriz original 
		# (dado que el conjunto de puntos del casco convexo es un subconjunto del conjunto 
		# de puntos original). En el segundo caso, los elementos del casco son los propios 
		# puntos convexos del casco.
	# clockwise = Bandera de orientación. Si es cierto, el casco convexo de salida está 
		# orientado en el sentido de las agujas del reloj. De lo contrario, se orienta en 
		# sentido contrario a las agujas del reloj. El sistema de coordenadas asumido 
		# tiene su eje X apuntando hacia la derecha y su eje Y apuntando hacia arriba.
	# returnPoints = Bandera de operación. En el caso de una matriz, cuando la bandera 
		# es verdadera, la función devuelve puntos de casco convexos. De lo contrario, 
		# devuelve índices de los puntos del casco convexo. Cuando la matriz de salida 
		# es std::vector, el indicador se ignora y la salida depende del tipo de 
		# vector: std::vector<int> implica returnPoints=false, std::vector<Point> 
		# implica returnPoints=true.
hull = cv.convexHull(points)
# print(hull)

# Se pinta el resultado en la imagen, que coincide con el rostro de la foto
cv.polylines(img,[hull],True,(118,118,0),2)
cv.fillConvexPoly(mask,hull, 255)

# Con la función cv.bitwise_and() se muestra solo el rostro como resultado de 
# aplicar la máscara
face_image = cv.bitwise_and(img_copy,img_copy,mask=mask)

# Delaunay triangulation
# La funcion cv2.boundingRect() devuelve el rectángulo entero vertical mínimo 
# que contiene el rectángulo girado
rect = cv.boundingRect(hull)
# Se se usa este código se pinta el rectángulo que encierra el rostro
# (x,y,w,h) = rect
# cv.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
# El rectángulo es necesario crearlo para poder usar la función cv2.Subdiv2S
# la cual crea una subdivisión vacía de Delaunay donde se pueden agregar 
# puntos 2D usando la función insert() . Todos los puntos que se agregarán 
# deben estar dentro del rectángulo especificado; de lo contrario, 
# se generará un error de tiempo de ejecución.
subdiv = cv.Subdiv2D(rect)
# El  métod cv2::inser() inserta un solo punto en una triangulación de Delaunay.
# cv2insert(pt)
	# pt = punto a insertar
subdiv.insert(landmarks_points)

# El métoodo cv2::getTriangleList() devuelve una lista de triánguloes en forma de
# vector
triangles = subdiv.getTriangleList()
# Es necesario convertir las coordenadas en números enteros porque son las 
# coordenadas exactas de una imagen
triangles = np.array(triangles,dtype=np.int32)

# El vector de triángulos siempre devuelve tres triángulos que se separan
# y se pintan en el rostro
for t in triangles:
	pt1 = (t[0],t[1])
	pt2 = (t[2],t[3])
	pt3 = (t[4],t[5])
	print(pt1)

	cv.line(img,pt1,pt2,(0,255,255),2)
	cv.line(img,pt2,pt3,(255,255,0),2)
	cv.line(img,pt1,pt3,(0,255,0),2)

cv.imshow("Paul", img)
cv.imshow("mask",mask)
cv.imshow("face_image",face_image)
cv.waitKey()
cv.destroyAllWindows()
