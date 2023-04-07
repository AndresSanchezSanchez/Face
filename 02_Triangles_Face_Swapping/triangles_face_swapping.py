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

# Se crea usa la función donde se extraen el punto del rostro que se ha calculado que será
# usada más adelante
def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

# Se crea una función que permita dar la información de las caras como se ha hecho en 
# los códigos anteriores

# Se carga las imágenes con el rostro y se redimensiona a 396x489
img1 = cv.imread("image/Andres_Sanchez.jpg")
# img1 = cv.resize(img1,(400,500))
img1 = cv.resize(img1,(396,489))
# img = cv.pyrDown(img)
img1_copy = img1.copy()
img1_gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
# Se crea una máscara que se usará más adelante
mask1 = np.zeros_like(img1_gray)

img2 = cv.imread("image/Paul_Walker.jpg")
# img2 = cv.resize(img2,(400,500))
img2 = cv.resize(img2,(396,489))
# img = cv.pyrDown(img)
img2_copy = img2.copy()
img2_gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
# Se crea una máscara que se usará más adelante
mask2 = np.zeros_like(img2_gray)

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

# Face 1
# Se alamcenan los puntos predichos en la varaible faces con el modelo
faces1 = detector(img1_gray)
# Para la rostro humano se tieen 68 puntos y con este bucle se pintan todos los puntos en la
# imagen del rostro usada
for face in faces1:
	landmarks = predictor(img1_gray, face)
	landmarks_points1 = []
	for n in range(0,68):
		x = landmarks.part(n).x
		y = landmarks.part(n).y
		landmarks_points1.append((x,y))
		# cv.circle(img1, (x,y),3,(255,75,255),-1)
# Se convierten los puntos en un array
points1 = np.array(landmarks_points1,np.int32)
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
hull = cv.convexHull(points1)

# Se pinta el resultado en la imagen, que coincide con el rostro de la foto
# cv.polylines(img,[hull],True,(118,118,0),2)
cv.fillConvexPoly(mask1,hull, 255)

# Con la función cv.bitwise_and() se muestra solo el rostro como resultado de 
# aplicar la máscara
face_image1 = cv.bitwise_and(img1,img1,mask=mask1)

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
subdiv.insert(landmarks_points1)
	
# El métoodo cv2::getTriangleList() devuelve una lista de triánguloes en forma de
# vector
triangles = subdiv.getTriangleList()
# Es necesario convertir las coordenadas en números enteros porque son las 
# coordenadas exactas de una imagen
triangles = np.array(triangles,dtype=np.int32)

# Se crea un vector donde se almacenan los índices de los triangulos equivalentes
# en el rostro
indexes_triangles = []
# El vector de triángulos siempre devuelve tres triángulos que se separan
# y se pintan en el rostro
# Se necesitan obtener obtener los puntos necesarios del rostro de cada 
# una de las dos caras
for t in triangles:
	pt1 = (t[0],t[1])
	pt2 = (t[2],t[3])
	pt3 = (t[4],t[5])


	# Se analiza el primer punto de la variable puntos que equivale al índice del rostro
	# de los triángulos que se van a generar
	index_pt1 = np.where((points1==pt1).all(axis=1))
	# Se crea usa la función donde se extraen el punto del rostro que se ha calculado
	index_pt1 = extract_index_nparray(index_pt1)

	index_pt2 = np.where((points1==pt2).all(axis=1))
	# Se crea usa la función donde se extraen el punto del rostro que se ha calculado
	index_pt2 = extract_index_nparray(index_pt2)

	index_pt3 = np.where((points1==pt3).all(axis=1))
	# Se crea usa la función donde se extraen el punto del rostro que se ha calculado
	index_pt3 = extract_index_nparray(index_pt3)

	# Se cre3a una condicción para introducir la información obtenida en el vector
	# indexes_triangles
	if index_pt1 is not None  and index_pt2 is not None  and index_pt3 is not None:
		# Se crea el triángulo y se añade al vector
		triangle = [index_pt1,index_pt2,index_pt3]
		indexes_triangles.append(triangle)

# Ahora se va a trabajar con los triángulos. Se pasan los índices de los 
# triángulos y se detecta el triángulo de ambas caras. Se recorta el área de 
# cada triángulo para trabajar con él específicamente.
# Se pone una condición para saber si es la primera cara o la segunda
# Face 2
# Se alamcenan los puntos predichos en la varaible faces con el modelo
faces2 = detector(img2_gray)
# Para la rostro humano se tieen 68 puntos y con este bucle se pintan todos los puntos en la
# imagen del rostro usada
for face in faces2:
	landmarks = predictor(img2_gray, face)
	landmarks_points2 = []
	for n in range(0,68):
		x = landmarks.part(n).x
		y = landmarks.part(n).y
		landmarks_points2.append((x,y))
		# cv.circle(img2, (x,y),3,(255,75,255),-1)

# Triangulación de las dos caras
for triangle_index in indexes_triangles:
	# Triangulación de la primera cara
	tr1_pt1 = landmarks_points1[triangle_index[0]]
	tr1_pt2 = landmarks_points1[triangle_index[1]]
	tr1_pt3 = landmarks_points1[triangle_index[2]]
	triangle1 = np.array([tr1_pt1,tr1_pt2,tr1_pt3], np.int32)

	# Se busca la frontera de los puntos para construir un rectángulo
	# La funcion cv2.boundingRect() devuelve el rectángulo entero vertical mínimo 
	# que contiene el rectángulo girado
	rect1 = cv.boundingRect(triangle1)
	(x,y,w,h) = rect1
	cropped_triangle1=img1[y:y+h,x:x+w]
	cropped_tr1_mask = np.zeros((h,w),np.uint8)

	points1 = np.array([[tr1_pt1[0]-x,tr1_pt1[1]-y],
		[tr1_pt2[0]-x,tr1_pt2[1]-y],
		[tr1_pt3[0]-x,tr1_pt3[1]-y]], np.int32)

	cv.fillConvexPoly(cropped_tr1_mask,points1,255)
	cropped_triangle1 = cv.bitwise_and(cropped_triangle1,cropped_triangle1,
		mask=cropped_tr1_mask)

	cv.line(img1,tr1_pt1,tr1_pt2,(0,0,255),2)
	cv.line(img1,tr1_pt2,tr1_pt3,(0,0,255),2)
	cv.line(img1,tr1_pt1,tr1_pt3,(0,0,255),2)

	# Face 2
	# Triangulación de la segunda cara
	tr2_pt1 = landmarks_points2[triangle_index[0]]
	tr2_pt2 = landmarks_points2[triangle_index[1]]
	tr2_pt3 = landmarks_points2[triangle_index[2]]
	triangle2 = np.array([tr2_pt1,tr2_pt2,tr2_pt3], np.int32)

	# Se busca la frontera de los puntos para construir un rectángulo
	# La funcion cv2.boundingRect() devuelve el rectángulo entero vertical mínimo 
	# que contiene el rectángulo girado
	rect2 = cv.boundingRect(triangle2)
	(x,y,w,h) = rect2
	cropped_triangle2=img2[y:y+h,x:x+w]
	cropped_tr2_mask = np.zeros((h,w),np.uint8)

	points2 = np.array([[tr2_pt1[0]-x,tr2_pt1[1]-y],
		[tr2_pt2[0]-x,tr2_pt2[1]-y],
		[tr2_pt3[0]-x,tr2_pt3[1]-y]], np.int32)

	cv.fillConvexPoly(cropped_tr2_mask,points2,255)
	cropped_triangle2 = cv.bitwise_and(cropped_triangle2,cropped_triangle2,
		mask=cropped_tr2_mask)

	cv.line(img2,tr2_pt1,tr2_pt2,(0,0,255),2)
	cv.line(img2,tr2_pt2,tr2_pt3,(0,0,255),2)
	cv.line(img2,tr2_pt1,tr2_pt3,(0,0,255),2)

	# Deformación de los triángulos para adaptarlos
	points1 = np.float32(points1)
	points2 = np.float32(points2)

	# Con la función cv2.getAffineTransform() se calcula una transformada 
	# afín a partir de tres pares de los puntos correspondientes.
	# cv2.getAffineTransform(src[], dst[])
		# src = Coordenadas de los vértices del triángulo en la imagen de origen.
		# dst = Coordenadas de los vértices del triángulo correspondiente en la 
			# imagen de destino.
	M = cv.getAffineTransform(points1,points2)

	# Con la función cv2.warpAffine() se deforman los triangulos para ajustarlos
	# aplicando una transformación afín a una imagen.
	# cv2.warpAffine(src, dst, M, dsize, flags = INTER_LINEAR, borderMode = BORDER_CONSTANT,
		# borderValue = Scalar())
	######################################################################################
		# src = imagen de entrada
		# dst = imagen de salida que tiene el tamaño dsize y el mismo tipo que src.
		# M = 2×3 transformation matrix.
		# dsize = tamaño de la imagen de salida.
		# flags = 	combinación de métodos de interpolación (ver InterpolationFlags ) 
			# y el indicador opcional WARP_INVERSE_MAP que significa que M es la 
			# transformación inversa (dst→src).
		# borderMode = 	método de extrapolación de píxeles (ver BorderTypes ); 
			# cuando borderMode=BORDER_TRANSPARENT , significa que la función 
			# no modifica los píxeles de la imagen de destino correspondientes 
			# a los "valores atípicos" de la imagen de origen.
		# borderValue = valor utilizado en caso de un borde constante; 
			# por defecto, es 0.
	warped_triangle = cv.warpAffine(cropped_triangle1, M, (w, h))

	break



cv.imshow("Image 1", img1)
cv.imshow("image2", img2)
cv.imshow("cropped triangle 1", cropped_triangle1)
cv.imshow("cropped triangle 2", cropped_triangle2)
cv.imshow("Warped triangle", warped_triangle)
cv.waitKey()
cv.destroyAllWindows()
