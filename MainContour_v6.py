# -*- coding: utf-8 -*-
#!/usr/bin/python3

import cv2
import numpy as np
#import sys
from matplotlib import pyplot as plt
import argparse
from matplotlib import * 
#matplotlib.use('Gtk3Agg')
#import pdb


drawing = False # true if mouse is pressed
ix = -1 #Creamos un punto inicial x,y
iy = -1,
dotslist = [] #Creamos una lista donde almacenaremos los puntos del contorno

global thick_contour
thick_contour = 15

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
            #cv2.circle(img,(x,y),1,(0,0,255),2)
            cv2.line(img, (x,y), (x,y), (0,255,0), thick_contour)#Dibujamos una linea de un solo pixel
            x = x
            y = y
            dot = [x,y]
            dotslist.append(dot)#Agregamos el punto a dotslist
            #print(dotslist) #Imprimimos el dotslist

    elif event == cv2.EVENT_LBUTTONUP:#Cremaos el evento si el boton se levanta
        drawing = False
        #cv2.circle(img,(x,y),1,(0,0,255),1)
        cv2.line(img, (x,y), (x,y), (0,255,0), thick_contour)#Dibujamos la ultima lina en el ultimo punto
      
    return dotslist#Retornamos el dotlist


def Croped(dotslist, img):#hacemos un corte de la imagen en linea recta de tal forma que tenga las 
                          #dimenciones maximas del poligono que creamos
    rect = cv2.boundingRect(dotslist)#Encontramos los limites maximos del
    (x,y,w,h) = rect#Tomamos las dimenciones maximas del dotlist y las guardamos para dimencionar la mascara
    croped = img[y:y+h, x:x+w].copy()#cortamos una seccion rectangular de la imagen
    dotslist2 = dotslist- dotslist.min(axis=0)#reajustamos el dotslist con el minimo 

    mask = np.zeros(croped.shape[:2], dtype = np.uint8)# creamos una mascara de ceros para poder hacer el corte irregular
    cv2.drawContours(mask, [dotslist2], -1, (255,255, 255), -1, cv2.LINE_AA)#dibujamos el contorno
    dts = cv2.bitwise_and(croped,croped, mask=mask)#hacemos ceros todos los pixeles externos al contorno, aplicamos la mascara a la imagen
    
    return [dts, mask, croped]

def save_img_with_contour(dotslist, img):
    mask = np.zeros(img.shape[:2], dtype = np.uint8)# creamos una mascara de ceros para poder hacer el corte irregular
    dotslist2 = dotslist- dotslist.min(axis=0)#reajustamos el dotslist con el minim
    cv2.drawContours(mask, [dotslist2], -1, (255,255, 255), -1, cv2.LINE_AA)#dibujamos el contorno
    
    return mask

def histogram(img, mask):
    
    hist = cv2.calcHist([img], [0], mask, [256], [0,256])
    return hist

def Listing(y):
    
    y1 = []#Creamos una lista vacia
    for i in range(len(y)):#llenamos la lista vacia con los datos de y, esto porque y es de la forma y = [[],[],[]], y necesitamos y = []
        y1.append(y[i][0])  
    
    return y1

def max_values(hist):
    
    hist = np.asarray(hist)
    max_count = max(hist)
    max_intensity = hist.argmax()
    
    return [max_count, max_intensity] 

def img_in_memory(file):
    
    img = cv2.imread(file)#Lee la imagen a color
    return img

def File_Writer(file_name, data1):
    
    file = open(str(file_name) + '_Counts_vs_intensity.csv','a')
    
    file.write('# ' + str(file_name) + '_Counts_vs_intensity'+  '\n')
    file.write('# counts max value: ' + str(data1[1]) +  '\n')
    file.write('# intensity max value: ' + str(data1[2]) +  '\n')
    file.write('# intensity     counts \n')
    
    for h in range(len(data1[0])):
        file.write(str(h) + '	'+ str(data1[0][h]) +'\n')
    file.close() 
    

    

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help = 'path to the image')
args = vars(ap.parse_args())
#llamamos asi #python Color_Detector_1.py --image 1.jpg

file = args['image']
#file = str(sys.argv[1])

#Rfactor= 0.20

img = cv2.imread(file)#Lee la imagen a color
img2 = cv2.imread(file,cv2.IMREAD_GRAYSCALE)#Lee la imagen pero en intensidad (B and W)
img3 = cv2.imread(file)
cv2.namedWindow(file, cv2.WINDOW_NORMAL)#Cremaos la ventana para mistras a img
cv2.setMouseCallback(file,draw_dots) #llamamos al MouseCall para dibujar el contorno


name1 = file[0:-4]#definimos un mobre sin la extrension del archivo

#Espacio_de_graficacion_1_de_histograma
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
plt.ion()

#ax1.set_xlim(0,256)
ax1.set_xticks(np.linspace(0, 256, 10))#ajusta las etiquetas en x
ax1.set_xlabel('Intensity Values', fontsize= 12)
ax1.set_ylabel('Counts', fontsize= 12)
ax1.set_title('Histogram of: ' + name1)

#Espacio_de_graficacion_2_de_histograma e imagen
fig2 = plt.figure()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
ax2 = fig2.add_subplot(121)

plt.ion()
ax2.set_xticks(np.linspace(0, 256, 10))#ajusta las etiquetas en x
ax2.set_xlabel('Intensity Values', fontsize= 12)
ax2.set_ylabel('Counts', fontsize= 12)
ax2.set_title('Image: ' + name1)

ax3 = fig2.add_subplot(122)

ax3.set_xlabel('x pixel position', fontsize= 12)
ax3.set_ylabel('y pixel position', fontsize= 12)
ax3.set_title('Histogram of: ' + name1)

plt.ion()

plt.show(block=False)

i = 1
while(1):
    cv2.imshow(file,img) #Mostramos a img en la ventana para dibujar el contono

    k = cv2.waitKey(10) & 0xFF
    #l = cv2.waitKey(1) & 0xFF
    
    
    if k == ord('a'): #space 
        print('Corte_aplicado')  
        dotslist = np.asarray(dotslist)#Convertimos el contorno en un array de numpy
        #Aplicamos el contorno a la image a partir de dtolist
        img_croped_BB = Croped(dotslist, img2)[0]#Rcuperamos solo la region de interes (imagen cortada con bordes negros=Black Borders)
        mask = Croped(dotslist, img2)[1] #Recuperamos la mascara creada
        img_croped = Croped(dotslist, img2)[2]#recuperamos la imagen cortada en rectangulo para analizar con la mascara     
        
        dotslist = dotslist.tolist()
        #Sacamos el histograma
        hist = histogram(img_croped, mask)#Calculamos el histograma usando la mascara #len(hist) = 256
        hist = Listing(hist)
        [max_count, max_intensity] = max_values(hist)
        print('max count: ' + str(max_count) + '\n' + 'max intensity: ' + str(max_intensity))

        
    if k == ord('s'):
        print('Datos guardados')
        #Analisis
        #x = np.linspace(0.0, len(hist), len(hist)) #creamos los x para hace el fit, (inicio, final, numero total)
        #pdb.set_trace()

        #Histograma
        #ax1.set_xticks(np.arange(0, len(x), step=20))
        line1, = ax1.plot(np.linspace(0, max(hist), 256, endpoint=False))
        line1.set_ydata(hist)
        ax1.set_xlabel('Intensity Values', fontsize= 12)
        ax1.set_ylabel('Counts', fontsize= 12)
        fig1.canvas.draw()#updatea el grafico con la nueva curva
        ax1.draw_artist(ax1.patch)#selecciona al arreglo de ax, para darle rango al eje x
        ax1.draw_artist(line1) #Redibuja line solo si es necesario
        ax1.text(120, 120, 'max count: ' + str(max_count) + '\n' + 'max intensity: ' + str(max_intensity), fontsize=12)
        ax1.legend(loc='best')
        
        fig1.canvas.flush_events()#Hace un sleep para que se pueda crear la grafica
        fig1.savefig('Histogram of ' + name1 +'_' +str(i))       
        
        #Imagen e histograma
        ax3.set_xlabel('x pixel position', fontsize= 12)
        ax3.set_ylabel('y pixel position', fontsize= 12)
        ax3.imshow(img_croped_BB, cmap= 'Greys')
        
        
        line2, = ax2.plot(np.linspace(0, max(hist), 256, endpoint=False))
        line2.set_ydata(hist)
        ax2.set_xlabel('Intensity Values', fontsize= 12)
        ax2.set_ylabel('Counts', fontsize= 12)
        fig2.canvas.draw()#updatea el grafico con la nueva curva
        ax2.draw_artist(ax2.patch)#selecciona al arreglo de ax, para darle rango al eje x
        ax2.draw_artist(line2) #Redibuja line solo si es necesario

        fig2.canvas.flush_events()#Hace un sleep para que se pueda crear la grafica
        fig2.savefig('Image_and_histogram_of_' + name1 + '_' +str(i))
        
        #plt.show()
        cv2.imshow('croped', img_croped_BB)#Mostramos img2 con el contorno 
        cv2.imwrite("Corte_" + name1 + '_' + str(i) +'_.jpg', img_croped_BB) 
        
        File_Writer('Data_Analisis_Image:_'+ name1 + '_' + str(i), [hist, max_count, max_intensity])
        
        selected_area = cv2.drawContours(img, [np.asarray(dotslist)], -1, (0,255,0), thick_contour)#Dibujamos el contorno
        cv2.imwrite("Area_seleccionada_numero_" + str(i) + "de:" + name1 +'_.jpg', selected_area)
        
        i += 1#Para poder guardar en la siguiente ejecucion sin sobreescribir la anterior

    if k == ord('d'):
        print('Contorno y datos borrados')
        hist.clear()
        dotslist.clear()
        ax1.clear()
        ax2.clear()
        #cv2.destroyAllWindows()#Destruimos todas las ventanas
        cv2.namedWindow(file, cv2.WINDOW_NORMAL)
        img = img_in_memory(file)

        
    if k == ord('q'):#esc
        #pdb.set_trace()
        break# hace,el break para para el programa si se estripa esc

cv2.destroyAllWindows()#Destruimos todas las ventanas
