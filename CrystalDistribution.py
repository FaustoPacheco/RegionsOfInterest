import cv2
import numpy as np
import sys
import os
from scipy.optimize import curve_fit as fit
from scipy import exp 
from matplotlib import pyplot as plt
import os.path as path
import pdb


entrada = input("¿Cúantas capas desea analizar?: \n")
entrada = int(entrada)
n = 0 #amount of cuts
h = 0 #altura de la imagen de entrada
w = 0 #ancho de la imagen de entrada
area_img = 0 #area de imagen de entrada
area_capa = 0 

rojo1_total = 0.0
rojo2_total = 0.0
amarillo_total = 0.0
azul_total = 0.0
verde_total = 0.0

cont_lin = 0
fin_lin = 0
#Estos arrays guardan la nueva información
coord_x = []
coord_y = []
diamt_prov = []
suma_diamt = []
K=0

factor_conv = 0

f = open("data_image.txt",'a')
f.write("LAYER DATA \n \n")
f.close()

def Color(capa, numero_c): ## capa se refiere al nombre de la imagen de la capa correspondiente, numero_c es la cantidad de capas
	numero_c = str(numero_c)
	img = capa
	img = cv2.imread(img)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 

	azules_oscuros = np.array([100,10,1])#Esto arrays son los que daran el rango de deteccion de azul
	#MII_R69_p1 70,95,20
	#MII_R69_p4 70, 5, 55
	#sirve para amarillo 20,95,20
	#MI_45 100,10,1
	azules_claros = np.array([255,160,170])
	#MII_R69_p1 145,200,255
	#MII_R69_p1 200,255,255  
	# #MI_45 255,1160,170

	rojos1_oscuros = np.array([1, 150, 125])
	#MII_R69_p1 0, 50, 100
	#MII_R69_p4 0, 50, 100
	#MI_54P 0,50,100
	#MI_45 1, 150, 125
	rojos1_claros = np.array([25, 200, 200])
	#MII_R69_p1 5, 255, 255
	#MII_R69_p4 5, 255, 255
	#MII_54P 5, 255, 255
	#MI_45 100,10,1 25, 200, 200

	rojos2_oscuros = np.array([0, 50, 100])
	#MII_R69_p1 7, 112, 70
	#MII_R69_p4 7, 112, 70
	#MII_54P 7, 100, 100
	#MI_45 0,50,100
	rojos2_claros = np.array([0, 50, 100])
	#MII_R69_p1 15, 255, 255
	#MII_R69_p4 15, 255, 255
	#MII_54P 30, 255, 255
	#MI_45 0,50,100

	amarillos_oscuros = np.array([10, 95, 20])
	#MII_R69_p1 10, 114, 155
	#MII_R69_p4 20, 95, 20
	#MI_54 20, 95, 20
	#MI_45 15,150,150
	amarillos_claros = np.array([80, 255, 255])
	#MII_R69_p1 60, 255, 255
	#MII_R69_p4 60, 255, 255
	#MII_54 60, 255, 255
	#MI_45 255,230,230

	verdes_oscuros = np.array([50,8,8])
	#MII_R69_p1 50, 8, 8
	#MII_R69_p4 50, 8, 8
	#MI_45 50,8,8
	verdes_claros = np.array([100,254,254])
	#MII_R69_p1 100, 254, 254
	#MII_R69_p4 100, 254, 254
	#MI_45 100,254,254


	#Creacion de mascaras
	mask1 = cv2.inRange(hsv, azules_oscuros, azules_claros)
	mask2 = cv2.inRange(hsv, rojos1_oscuros, rojos1_claros)
	mask3 = cv2.inRange(hsv, rojos2_oscuros, rojos2_claros)
	mask4 = cv2.inRange(hsv, amarillos_oscuros, amarillos_claros)
	mask5 = cv2.inRange(hsv, verdes_oscuros, verdes_claros)

	imask = mask1>0 
	azul = np.zeros_like(img, np.uint8) 
	azul[imask] = img[imask] 
	cv2.imwrite("azul_" + numero_c + ".png", azul) #24415524

	imask = mask2>0 
	rojo1 = np.zeros_like(img, np.uint8) 
	rojo1[imask] = img[imask] 
	## save 
	cv2.imwrite("rojo1_" + numero_c + ".png", rojo1)

	imask = mask3>0 
	rojo2 = np.zeros_like(img, np.uint8) 
	rojo2[imask] = img[imask] 
	## save 
	cv2.imwrite("rojo2_" + numero_c + ".png", rojo2)

	imask = mask4>0 
	amarillo = np.zeros_like(img, np.uint8) 
	amarillo[imask] = img[imask] 
	## save 
	cv2.imwrite("amarillo_" + numero_c + ".png", amarillo)

	imask = mask5>0 
	verde = np.zeros_like(img, np.uint8) 
	verde[imask] = img[imask] 
	## save 
	cv2.imwrite("verde_" + numero_c + ".png", verde)


def Centros(h, v, s, H, V, S, color, num, color_total):
    font = cv2.FONT_HERSHEY_SIMPLEX

    num = str(num)
    img = color + "_" + num + ".png"
    imagen = cv2.imread(img)
    if(imagen is None):
        print("Error: no se ha podido encontrar la imagen")
        quit()
    
    #Convertimos la imagen a HSV
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    
    color_bajos = np.array([h,v,s])
    color_altos = np.array([H, V, S])
    fondo = cv2.inRange(hsv, color_bajos, color_altos)
   
    kernel = np.ones((3,3),np.uint8)
    fondo = cv2.morphologyEx(fondo,cv2.MORPH_OPEN,kernel)
    fondo = cv2.morphologyEx(fondo,cv2.MORPH_CLOSE,kernel)
    
    
    #Buscamos los contornos de las bolas y los dibujamos en verde
    _, contours, hierarchy = cv2.findContours(fondo, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imagen, contours, -1, (0,255,0), 2)
    
    #Buscamos el centro de las bolas y lo pintamos en rojo
    for i in contours:
        #Calcular el centro a partir de los momentos
        momentos = cv2.moments(i)
        cx = int(momentos['m10']/momentos['m00'])
        cy = int(momentos['m01']/momentos['m00'])

        area = cv2.contourArea(i)
        area_micras = area * factor_conv
        #print("Área: ", area_micras)

        #color_total = color_total + area_micras
        #print("Área total de" + color + ": ", color_total)
        #print("n: ", num)

        diametro = ((area_micras/3.141592)**(1/2))*2
        coordenadas = str(diametro) + "\n"

        text_centy = "center_cy" + color + num + ".txt"
        text_centx = "center_cx" + color + num + ".txt"
        text = "diametro_" + color + num + ".txt"
        f = open(text,'a')
        f.write(coordenadas)
        f.close()

        cx = int(cx)
        cy = int(cy)
        f = open(text_centx,'a')
        f.write(str(cx) + "\n")
        f.close()
        f = open(text_centy,'a')
        f.write(str(cy) + "\n")
        f.close()

        cv2.circle(imagen,(cx, cy), 3, (0,0,255), -1)
        cv2.circle(imagen,(cx, cy), 3, (0,0,255), -1)
        cv2.imshow('Cristales detectados', imagen)
        



# def Unir_rojos(n):
# 	rojo1 = "diametro_rojo1" + str(n) + ".txt"
# 	rojo2 = "diametro_rojo2" + str(n) + ".txt"
# 	rojo = "diametro_rojo" + str(n) + ".txt"

# 	if path.exists(rojo1):
# 		if path.exists(rojo2):
# 			filenames = [rojo1, rojo2] 
# 			with open(rojo, 'w') as outfile: 
# 				for fname in filenames: 
# 					with open(fname) as infile:
# 						for line in infile: 
# 							outfile.write(line)
# 		else:
# 			os.rename(rojo1, rojo)
# 	else:
# 		os.rename(rojo2, rojo)

def file_array(text):
    dato = []
    try:
        archivo=open(text,'r')
        leer_fila= archivo.readlines()
        archivo.close()
        for lista in leer_fila:
            if lista[-1]=="\n":
                x = float(lista[:-1].split(", ")[0])
                dato.append(x)
            else:
                x = float(lista.split(", ")[0])
                dato.append(x)
    except(FileNotFoundError):
        advert = "File not found"
        print(advert)
    return dato
    
def change_file(text, array):
	try: 
		f = open(text,'w')
		num = str(array[0])
		f.write(num + '\n')
		f.close()
		for i in range(len(array)-1):
			f = open(text,'a')
			num = str(array[i+1])
			f.write(num + '\n')
			f.close()
	except(IndexError):
		print("End of the list")

def leerLineaFichero(filename, xtext, ytext, cont_lin, suma_diamt, coord_x, coord_y, diamt_prov,K):
    lines = file_array(filename)
    #print(len(lines))
    diamt_big = 0.0
    diamt = 0.0
    for i in range(len(lines)):
        if i+1 <= len(lines)-1:
            var1 = lines[i]
            var2 = lines[i+1]
            var1 = float(var1)
            var2 = float(var2)
            if var1 >= var2:
                diamt = float(lines[i])
                if diamt >= diamt_big:
                    cont_lin = i
                    diamt_big = diamt
            else:
                diamt = float(lines[i+1])
                if diamt >= diamt_big:
                    cont_lin = i+1
                    diamt_big = diamt

    #print(diamt_big, cont_lin)

    linescx=file_array(xtext)
    cx_big = linescx[cont_lin]
    #print(cx_big)

    linescy=file_array(ytext)
    cy_big = linescy[cont_lin]
    #print(cy_big)

    diamt_i = diamt_big
    for i in range(len(linescx)):
        if abs(cx_big-linescx[i])<=95 and abs(cy_big-linescy[i])<=95:
            if diamt_i!=lines[i]:
                diamt_big += lines[i]
        else:
            if diamt_i!=lines[i]:
                coord_x.append(linescx[i])
                coord_y.append(linescy[i])
                diamt_prov.append(lines[i])
            #print(lines[i], linescx[i], linescy[i])
    suma_diamt.append(diamt_big)
    # print(suma_diamt)
    # print(diamt_big)
    if len(diamt_prov) <= 1:
        K = 1
    # print(coord_x)
    # print(coord_y)
    # print(diamt_prov)
    change_file(xtext, coord_x)
    change_file(ytext, coord_y)
    change_file(filename, diamt_prov)
    
    coord_x.clear()
    coord_y.clear()
    diamt_prov.clear()


def final_doc(filename, xtext, ytext, cont_lin, suma_diamt, coord_x, coord_y, diamt_prov,K):
	try: 
		while K == 0:
			#print(len(diamt_prov))
			leerLineaFichero(filename, xtext, ytext, cont_lin, suma_diamt, coord_x, coord_y, diamt_prov,K)

	except(IndexError):
		print("End of the list")

	change_file(filename, suma_diamt)
	coord_x.clear()
	coord_y.clear()
	diamt_prov.clear()
	suma_diamt.clear()

def Area_cut(image):
	cut_image = cv2.imread(image)#Lee la imagen a color
	hcut, wcut, _ = cut_image.shape
	return hcut*wcut

def Porcentaje_total(n, color_total, color, area_img, text, image):
	diamt = []
	diamt = file_array(text)
	cant = len(diamt)
	area_cut = Area_cut(image)*factor_conv

	for i in range (0, cant):
		area = (np.pi/4)*diamt[i]**(2)
		color_total = color_total + area
	 
	percnt_total = (color_total/area_img)*100
	percnt_cut = (color_total/area_cut)*100

	text1 = "Percentage of the total image: " + str(percnt_total) + "%"
	text2 = "Percentage of the layer: " + str(percnt_cut) + "%"
	f = open("data_image.txt",'a')
	f.write("LAYER " + str(n) + " " + color + ": \n" )
	f.write(text1 + "\n")
	f.write(text2 + "\n")

	try:
		diamt_prom = sum(diamt)/cant
		text3 = "Average diameter: " + str(diamt_prom)
		f.write(text3 + "\n" + "\n")

	except(ZeroDivisionError):
		text4 = "Average diameter: There is no pigments" 
		f.write(text4 + "\n" + "\n")


	f.close()

def Crear_hist(text, color, n):
    dato = []
    try:
        archivo=open(text,'r')
        histograma = "histograma" + "_" + color + str(n) + ".png"
        leer_fila= archivo.readlines()
        archivo.close()
        for lista in leer_fila:
            if lista[-1]=="\n":
                x = float(lista[:-1].split(", ")[0])
                dato.append(x)
            else:
                x = float(lista.split(", ")[0])
                dato.append(x)
        plt.hist(dato, bins='auto')
        # plt.hist(dato[3], **kwargs)
        plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
        title = "Amount of crystals color " + color + " in the layer " + str(n)
        plt.title(title)
        plt.xlabel("Diameter (micrometer)")
        plt.ylabel("Frecuency")
        #plt.show(histograma)
        plt.savefig(histograma)
        plt.close()
    except(FileNotFoundError):
        advert = "No se detectan cristales de color " + color + " en la capa " + str(n)
        print(advert)

for i in range(0, entrada):
	
	drawing = False # true if mouse is pressed
	ix = -1 #Creamos un punto inicial x,y
	iy = -1,
	dotslist = [] #Creamos una lista donde almacenaremos los puntos del contorno

	# mouse callback function
	def draw_dots(event,x,y,flags,param): #Crea los puntos de contorno
		global ix,iy,drawing, dotslist#Hacemos globales la variabbles dentro de la funcion

		if event == cv2.EVENT_LBUTTONDOWN:#creamos la accion que se realizara si damos click
			drawing = True #Drawinf se vuelve True
			ix = x #Tomamos el punto donde se dio click
			iy = y
			dot = [x,y]
			dotslist.append(dot)#Lo agregamos al dotslist

		elif event == cv2.EVENT_MOUSEMOVE:#Creamos la accion si el mouse se mueve
			if drawing == True: #drawing se vuelve true
				cv2.line(img, (x,y), (x,y), (255,255,255), 2)#Dibujamos una linea de un solo pixel
				x = x
				y = y
				dot = [x,y]
				dotslist.append(dot)#Agregamos el punto a dotslist

		elif event == cv2.EVENT_LBUTTONUP:#Creamos el evento si el boton se levanta
			drawing = False
			cv2.line(img, (x,y), (x,y), (255,255,255), 2)#Dibujamos la ultima linea en el ultimo punto, vamos (x,y) a (x,y), es una linea de un punto.
			
		return dotslist#Retornamos el dotlist
	
	def Croped(dotslist, img):#hacemos un corte de la imagen en linea recta de tal forma que tenga las 
								#dimenciones maximas del poligono que creamos
		rect = cv2.boundingRect(dotslist)#Encontramos los limites maximos del
		(x,y,w,h) = rect#Tomamos las dimenciones maximas del dotlist y las guardamos para dimencionar la mascara
		croped = img[y:y+h, x:x+w].copy()#cortamos una seccion rectangular de la imagen
		dotslist2 = dotslist- dotslist.min(axis=0)#reajustamos el dotslist con el minimo 

		mask = np.zeros(croped.shape[:2], dtype = np.uint8)# creamos una mascara de ceros para poder hacer el corte irregular
		cv2.drawContours(mask, [dotslist2], -1, (255,255, 255), -1, cv2.LINE_AA)#dibujamos el contorno
		dts = cv2.bitwise_and(croped,croped, mask=mask)#hacemos ceros todos los pixeles externos al contorno
		

		return [dts, mask, croped]

	def histogram(img, mask):
		hist = cv2.calcHist([img], [0], mask, [256], [0,256])
		return hist

	def Listing(y):
		y1 = []#Creamos una lista vacia
		for i in range(len(y)):#llenamos la lista vacia con los datos de y, esto porque y es de la forma y = [[],[],[]], y necesitamos y = []
			y1.append(y[i][0])  
		
		return y1

	file = str(sys.argv[1])

	img = cv2.imread(file)#Lee la imagen a color
	h, w, _ = img.shape
	#print("alto: ", h)
	#print("ancho: ", w)
	factor_conv = ((128.8/w + 91.2/h)/2)*0.1066
	area_img = h*w*factor_conv
	img2 = cv2.imread(file,cv2.IMREAD_GRAYSCALE)#Lee la imagen pero en intensidad (B and W)
	cv2.namedWindow(file)#Cremaos la ventana para mistras a img
	cv2.setMouseCallback(file,draw_dots) #llamamos al MouseCall para dibujar el contorno


	while(1):
		n = str(n)
		cv2.imshow(file,img) #Mostramos a img en la ventana para dibujar el contono

		k = cv2.waitKey(1) & 0xFF

		if k == 32: #space
			dotslist = np.asarray(dotslist)#Convertimos el contorno en un array de numpy

			img_croped_BB = Croped(dotslist, img)[0]#Rcuperamos solo la region de interes (imagen cortada con bordes negros)
			mask = Croped(dotslist, img)[1] #Recuperamos la mascara creada
			img_croped = Croped(dotslist, img)[2]#recuperamos la imagen cortada en rectangulo para analizar con la mascara     
			hist = histogram(img_croped, mask)#Calculamos el histograma usando la mascara #len(hist) = 256
			hist = Listing(hist)
			hist.pop(0)

			plt.show()
			cv2.imshow('croped', img_croped_BB)#Mostramos img2 con el contorno 
			capa = "capa_" + n +'.png'
			cv2.imwrite(capa, img_croped_BB)
			Color(capa, n)
            
		
		if k == 27:#esc
			Centros(100, 10, 1, 255, 160, 170, "azul", n, azul_total) 
			Centros(20, 95, 20, 100, 255, 255, "amarillo", n, amarillo_total)
			Centros(1, 150, 125, 25, 200, 200, "rojo1", n, rojo1_total) #7,150,50
			Centros(0, 50, 100, 5, 200, 200, "rojo2", n, rojo2_total)
			Centros(50, 8, 8, 100, 254, 254, "verde", n, verde_total)

			#if n == entrada:

			cut = "capa_" + str(n) + ".png"
			
			amarillo = "diametro_amarillo" + str(n) + ".txt"
			azul = "diametro_azul" + str(n) + ".txt"
			rojo = "diametro_rojo" + str(n) + ".txt"
			rojo1 = "diametro_rojo1" + str(n) + ".txt"
			rojo2 = "diametro_rojo2" + str(n) + ".txt"
			verde = "diametro_verde" + str(n) + ".txt"

			cx_azul = "center_cxazul" + str(n) + ".txt" 
			cx_rojo1 = "center_cxrojo1" + str(n) + ".txt" 
			cx_rojo2 = "center_cxrojo2" + str(n) + ".txt" 
			cx_verde = "center_cxverde" + str(n) + ".txt" 
			cx_amarillo = "center_cxamarillo" + str(n) + ".txt" 

			cy_azul = "center_cyazul" + str(n) + ".txt" 
			cy_rojo1 = "center_cyrojo1" + str(n) + ".txt" 
			cy_rojo2 = "center_cyrojo2" + str(n) + ".txt" 
			cy_verde = "center_cyverde" + str(n) + ".txt" 
			cy_amarillo = "center_cyamarillo" + str(n) + ".txt" 

			
			final_doc(azul, cx_azul, cy_azul, cont_lin, suma_diamt, coord_x, coord_y, diamt_prov,K)
			final_doc(verde, cx_verde, cy_verde, cont_lin, suma_diamt, coord_x, coord_y, diamt_prov,K)
			final_doc(amarillo, cx_amarillo, cy_amarillo, cont_lin, suma_diamt, coord_x, coord_y, diamt_prov,K)
			final_doc(rojo1, cx_rojo1, cy_rojo1, cont_lin, suma_diamt, coord_x, coord_y, diamt_prov,K)
			final_doc(rojo2, cx_rojo2, cy_rojo2, cont_lin, suma_diamt, coord_x, coord_y, diamt_prov,K)


			Crear_hist(rojo1, "vermilion red", n)
			Crear_hist(rojo2, "lead red", n)
			Crear_hist(azul, "ultramarine blue", n)
			Crear_hist(amarillo, "chrome yellow", n)
			Crear_hist(verde, "viridian", n)

			Porcentaje_total(n, verde_total, "Viridian", area_img, verde, cut)
			Porcentaje_total(n, rojo1_total, "Vermilion Red", area_img, rojo1, cut)
			Porcentaje_total(n, rojo2_total, "Lead Red", area_img, rojo2, cut)
			Porcentaje_total(n, amarillo_total, "Chrome Yellow", area_img, amarillo, cut)
			Porcentaje_total(n, azul_total, "Ultramarine Blue", area_img, azul, cut)

			try: 
				# os.remove(azul)
				# os.remove(amarillo)
				# os.remove(rojo1)
				# os.remove(rojo2)
				# os.remove(verde)

				os.remove(cx_azul)
				os.remove(cx_amarillo)
				os.remove(cx_rojo1)
				os.remove(cx_rojo2)
				os.remove(cx_verde)

				os.remove(cy_azul)
				os.remove(cy_amarillo)
				os.remove(cy_rojo1)
				os.remove(cy_rojo2)
				os.remove(cy_verde)

			except(FileNotFoundError):
				print(".")

			n = int(n)
			n = n + 1
			break

cv2.destroyAllWindows()#Destruimos todas las ventanas