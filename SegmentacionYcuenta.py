import cv2 # OpenCV para computer vision
import numpy as np # Para cálculo de matrices
import matplotlib.pyplot as plt #Para graficar
import os #Habilita el manejo de directorios
#Leyendo imagen de entrada
Ruta = r'C:\Users\user\Desktop\Procesamiento de Imagenes\Final\Foto5.png' #Ubicación de la imagen desde el google drive
Imagen = cv2.imread(Ruta)#Lee
plt.imshow(Imagen[:,:,[2,1,0]].astype('uint8'),vmin=0, vmax=255)
plt.show()

[Fl, Cl, CH]=Imagen.shape
#Extracción de canales en HSV
Imagen=Imagen[:,:,[2,1,0]]
Imagen_hsv = cv2.cvtColor(Imagen,cv2.COLOR_BGR2HSV)
Hue=Imagen_hsv[:,:,0]
Saturation=Imagen_hsv[:,:,1]
Value=Imagen_hsv[:,:,2]
#Extracción de canales en RGB
#Imagen=Imagen[:,:,[2,1,0]]#Organiza
Rojo=Imagen[:,:,2]
Verde=Imagen[:,:,1]
Azul=Imagen[:,:,0]
Gris=cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)

#Mostrando en pantalla
#print('Mostrando gráficos en RGB & HSV')
fig, axs =plt.subplots(2, 4,figsize=(10,10))
axs[0, 0].imshow(Imagen[:,:,[2,1,0]].astype('uint8'),vmin=0, vmax=255)
axs[0, 0].set_title('Original')
axs[0, 1].imshow(Rojo.astype('uint8'),vmin=0, vmax=255,cmap='gray')
axs[0, 1].set_title('Rojo')
axs[0, 2].imshow(Verde.astype('uint8'),vmin=0, vmax=255,cmap='gray')
axs[0, 2].set_title('Verde')
axs[0, 3].imshow(Azul.astype('uint8'),vmin=0, vmax=255,cmap='gray')
axs[0, 3].set_title('Azul')
axs[1, 0].imshow(Gris.astype('uint8'),vmin=0, vmax=255,cmap='gray')
axs[1, 0].set_title('Gris')
axs[1, 1].imshow(Hue.astype('uint8'),vmin=0, vmax=179,cmap='gray')
axs[1, 1].set_title('Tono')
axs[1, 2].imshow(Saturation.astype('uint8'),vmin=0, vmax=255,cmap='gray')
axs[1, 2].set_title('Saturación')
axs[1, 3].imshow(Value.astype('uint8'),vmin=0, vmax=255,cmap='gray')
axs[1, 3].set_title('Intensidad')
#plt.show()
#print('')

#Segmentando Berenjenas
Bin_Berenjena=np.zeros((Fl,Cl))
Bin_Berenjena_1=np.zeros((Fl,Cl))
Bin_Berenjena_2=np.zeros((Fl,Cl))
Bin_Berenjena = (Hue > 157)
print('Segmentada')
plt.imshow(Bin_Berenjena.astype('uint8'),cmap='gray',vmin=0, vmax=1,)
plt.show()
plt.title='0'

#Mejorando regiones
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35))
Bin_Berenjena_1 = cv2.morphologyEx(Bin_Berenjena.astype('uint8'), cv2.MORPH_CLOSE, kernel)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(40,40))
Bin_Berenjena_2= cv2.morphologyEx(Bin_Berenjena_1.astype('uint8'), cv2.MORPH_OPEN, kernel2)
print('Mejorada')
plt.imshow(Bin_Berenjena_2.astype('uint8'),cmap='gray',vmin=0, vmax=1,)
plt.show()
plt.title='0'

#Contando objetos
(Bordes_Berenjena,_) =cv2.findContours(Bin_Berenjena_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print('Usted tiene ', len(Bordes_Berenjena), 'Berenjenas' )
print('El valor que debe pagar es: ', 1800*len(Bordes_Berenjena), 'pesos')

#Buscando Zanahorias
Bin_Zanahoria=np.zeros((Fl,Cl))
Bin_Zanahoria_1=np.zeros((Fl,Cl))
Bin_Zanahoria_2=np.zeros((Fl,Cl))
Bin_Zanahoria = (Verde > 140)*(Verde< 170)*(Value > 245)*(Saturation > 140)
#print('Segmentada')
#plt.imshow(Bin_Zanahoria.astype('uint8'),cmap='gray',vmin=0, vmax=1,)
#plt.show()
#plt.title='0'

#Mejorando resultado con apertura y cierre
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
Bin_Zanahoria_1 = cv2.morphologyEx(Bin_Zanahoria.astype('uint8'), cv2.MORPH_CLOSE, kernel)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
Bin_Zanahoria_2= cv2.morphologyEx(Bin_Zanahoria_1.astype('uint8'), cv2.MORPH_OPEN, kernel2)
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))
Bin_Zanahoria_3 = cv2.dilate(Bin_Zanahoria_2.astype('uint8'),kernel3)
#print('Mejorada')
#plt.imshow(Bin_Zanahoria_2.astype('uint8'),cmap='gray',vmin=0, vmax=1,)
#plt.show()
#plt.title='0'


(Bordes_Zanahoria,_) =cv2.findContours(Bin_Zanahoria_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print('Usted tiene', len(Bordes_Zanahoria), 'Zanahorias' )
#print('El valor que debe pagar es: ', 300*len(Bordes_Zanahoria), 'pesos')

#Buscando Manzanas
Bin_Manzana=np.zeros((Fl,Cl))
Bin_Manzana_1=np.zeros((Fl,Cl))
Bin_Manzana_2=np.zeros((Fl,Cl))
Bin_Manzana = (Azul>200)*(Azul<220)*(Hue>80)*(Hue<140)*(Verde>30)*(Verde<58)
print('Segmentada')
plt.imshow(Bin_Manzana.astype('uint8'),cmap='gray',vmin=0, vmax=1,)
plt.show()
plt.title='0'

#Mejorando resultado con apertura y cierre
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
Bin_Manzana_1 = cv2.morphologyEx(Bin_Manzana.astype('uint8'), cv2.MORPH_CLOSE, kernel)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
Bin_Manzana_2= cv2.morphologyEx(Bin_Manzana_1.astype('uint8'), cv2.MORPH_OPEN, kernel2)
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))
Bin_Manzana_3 = cv2.dilate(Bin_Manzana_2.astype('uint8'),kernel3)
print('Mejorada')
plt.imshow(Bin_Manzana_2.astype('uint8'),cmap='gray',vmin=0, vmax=1,)
plt.show()
plt.title='0'


(Bordes_Manzana,_) =cv2.findContours(Bin_Manzana_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print('Usted tiene', len(Bordes_Manzana), 'Manzanas' )
#print('El valor que debe pagar es: ', 300*len(Bordes_Manzana), 'pesos')

#Buscando Bananos
Bin_Banano=np.zeros((Fl,Cl))
Bin_Banano_1=np.zeros((Fl,Cl))
Bin_Banano_2=np.zeros((Fl,Cl))
Bin_Banano = (Gris>160)*(Gris<180)*(Saturation>170)*(Saturation<240)
#print('Segmentada')
#plt.imshow(Bin_Banano.astype('uint8'),cmap='gray',vmin=0, vmax=1,)
#plt.show()
#plt.title='0'

#Mejorando resultado con apertura y cierre
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
Bin_Banano_1= cv2.morphologyEx(Bin_Banano.astype('uint8'), cv2.MORPH_OPEN, kernel2)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35))
Bin_Banano_2 = cv2.morphologyEx(Bin_Banano_1.astype('uint8'), cv2.MORPH_CLOSE, kernel)
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))
Bin_Banano_3 = cv2.dilate(Bin_Banano_2.astype('uint8'),kernel3)
#print('Mejorada')
#plt.imshow(Bin_Banano_2.astype('uint8'),cmap='gray',vmin=0, vmax=1,)
#plt.show()
#plt.title='0'


(Bordes_Banano,_) =cv2.findContours(Bin_Banano_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print('Usted tiene', len(Bordes_Banano), 'Bananos' )
#print('El valor que debe pagar es: ', 300*len(Bordes_Banano), 'pesos')


#Contando productos y calculando cuenta
if len(Bordes_Berenjena)>0:
  print('Su pedido contiene '+ str(len(Bordes_Berenjena))+' Berenjena(s), el subtotal de este producto es: ' + str(len(Bordes_Berenjena)*1900)+' Pesos.')
if len(Bordes_Zanahoria)>0:
  print('Su pedido contiene '+ str(len(Bordes_Zanahoria))+' Zanahoria(s), el subtotal de este producto es: ' + str(len(Bordes_Zanahoria)*600)+' Pesos.')
if len(Bordes_Manzana)>0:
  print('Su pedido contiene '+ str(len(Bordes_Manzana))+' Manzana(s), el subtotal de este producto es: ' + str(len(Bordes_Manzana)*500)+' Pesos.')
if len(Bordes_Banano)>0:
  print('Su pedido contiene '+ str(len(Bordes_Banano))+' Banano(s), el subtotal de este producto es: ' + str(len(Bordes_Banano)*300)+' Pesos.')
print('En total, su pedido suma: ' + str(len(Bordes_Berenjena)*1900 + len(Bordes_Zanahoria)*600 + len(Bordes_Manzana)*500 + len(Bordes_Banano)*300) + ' Pesos.')





