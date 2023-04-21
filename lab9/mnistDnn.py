#Autor: Francisco García Lagos

#%matplotlib inline
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras import Sequential
from keras.layers import Convolution2D
from keras.layers import Activation
from keras.layers import MaxPooling2D, Dropout 
from keras.layers import Flatten, Dense
from keras.models import load_model, save_model

 
def cargar_mnist():
    """Cargar los datos de la BD MNIST.
       Los ficheros de la BD deben estar descomprimidos, y ubicados 
       en el directorio que señala `path`.
       Hay dos tipos de patrones, de entrenamiento, y de prueba. Para
       cargar los primeros, kind debe ser kind='train', y para cargar
       los patrones de test, kind='f10k'"""
    datos = np.load('./datos/mnist.npz')
    # Datos de entrenamiento
    # Entradas
    x_train = datos['x_train']
    y_train = datos['y_train']
    x_test = datos['x_test']
    y_test = datos['y_test']
    return x_train, y_train, x_test, y_test
    
def preprocesa_mnist(x_train, y_train, x_test, y_test):    
    # pre-procesamiento
    # En este caso sólo hace falta convertir los niveles de gris de cada pixel a un número
    #  en punto flotante, y normalizarlo
    #X_train /= 255    
    print('Dimensión del conjunto de patrones original')
    print('Dims x_train {}'.format(x_train.shape))
    print('Dims y_train {}'.format(y_train.shape))
    print('Dims x_test {}'.format(x_test.shape))
    print('Dims y_test {}'.format(y_test.shape))
    for i in range(20):    
        plt.subplot(2, 10, i + 1)
        plt.imshow(x_train[i, :, :], cmap='gray') 
        plt.axis('off')
    #plt.show()

    # cambiamos x_train y x_test a dimensiones 60000x784
    x_train_pre = x_train.reshape((x_train.shape[0], -1))
    x_test_pre = x_test.reshape((x_test.shape[0], -1))
    # normalizamos ambas matrices en [0,1]
    x_train_pre = x_train_pre.astype(np.float32) / 255.0
    x_test_pre = x_test_pre.astype(np.float32) / 255.0

    y_train_pre = np.zeros(shape=(y_train.shape[0],10),dtype=np.float32)
    for i in range(0,y_train_pre.shape[0]):
        y_train_pre[i,y_train[i]] = 1
    y_test_pre = np.zeros(shape=(y_test.shape[0],10),dtype=np.float32)
    for i in range(0,y_test_pre.shape[0]):
        y_test_pre[i,y_test[i]] = 1
    return x_train_pre, y_train_pre, x_test_pre, y_test_pre


# Para hacer los resulatdos reproducibles
#np.random.seed(1337)

# cargamos los datos a memoria
x_train, y_train, x_test, y_test = cargar_mnist()
# los preprocesamos
x_train, y_train, x_test, y_test = preprocesa_mnist(x_train, y_train, x_test, y_test)
print('Dimensión del conjunto de patrones preprocesado')
print('Dims x_train {}'.format(x_train.shape))
print('Dims y_train {}'.format(y_train.shape))
print('Dims x_test {}'.format(x_test.shape))
print('Dims y_test {}'.format(y_test.shape))

# Para trabajar con DNN es conveniente representar las imágenes como matrices de
# W x H x C, donde W es el ancho, H el alto y C el número de colores. En nuestro caso
# tenemos que convertir los patrones a matrices de 28x28x1. Cada fila será un patrón
# (una imagen)
img_rows, img_cols = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1) 
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) 
input_shape = (img_rows, img_cols, 1)

# creamos una DNN con Keras
model = Sequential()

# craemos una capa convolucional con un kernel de 3x3 píxeles 
# de dos dimensiones 
n_filters = 32
kernel_size = (3, 3)
model.add(Convolution2D(n_filters, kernel_size, 
                        padding='valid', input_shape=input_shape))
model.add(Activation('relu'))
# Añadimos una capa más de convolución
model.add(Convolution2D(n_filters, kernel_size))
model.add(Activation('relu'))
# añadimos a continuación una capa para pool, un especificados el dropout
pool_size = (2, 2)
model.add(MaxPooling2D(pool_size=pool_size)) 
model.add(Dropout(0.25))
# hacemos un flatten para el modelo y creamos una capa softmax para
# las salidas
model.add(Flatten()) 
model.add(Dense(128))
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
num_clases = 10
model.add(Dense(num_clases))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
                optimizer='adadelta', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=15, 
                verbose=1, validation_data=(x_test, y_test))
ret = model.evaluate(x_test, y_test, verbose=0)
print(ret)

save_model(model, 'prueba15.hdf5', overwrite=True, include_optimizer=True)

