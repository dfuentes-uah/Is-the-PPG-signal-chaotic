from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Reshape,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
from scipy.ndimage import rotate
import tensorflow.keras
import tensorflow.keras.layers as layers
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import GaussianNoise,Bidirectional,LSTM,Add,MaxPooling1D,Conv1D,Conv2D, MaxPooling2D,Dropout,Lambda,Cropping2D,concatenate,Reshape,ZeroPadding2D,UpSampling2D,Activation,Input,BatchNormalization,Concatenate, concatenate, GaussianNoise,Conv2DTranspose,Cropping2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, load_model,Model
from skimage import data, img_as_float
from skimage import exposure
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import tensorflow as tf
from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import filters
from scipy import ndimage
from tensorflow.keras.models import Sequential, load_model,Model
import scipy.io
from scipy import *
import tensorflow.keras.backend as K
import cv2
#import py3d
import hdf5storage
from tensorflow.keras.layers import LeakyReLU, PReLU
import pandas as pd


def adjust_mean(senal,media_objetivo):
    mediactual=senal.mean()
    dif=media_objetivo-mediactual
    senal=senal+dif

    return senal

timestep=5000
#1000#500
valsize=0#20000

# aleatoria=pd.read_csv('RSa.dat').values
# aperiodica=pd.read_csv('RSap.dat').values
# periodica=pd.read_csv('RSp.dat').values
# cuasiperiodica=pd.read_csv('RSq.dat').values
# caotica=pd.read_csv('RSc.dat').values
#
media_objetivo=0.0
# periodica=adjust_mean(periodica,media_objetivo)
# aperiodica=adjust_mean(aperiodica,media_objetivo)
# cuasiperiodica=adjust_mean(cuasiperiodica,media_objetivo)
# caotica=adjust_mean(caotica,media_objetivo)
# aleatoria=adjust_mean(aleatoria,media_objetivo)
#
# periodica=periodica/1.5
# aperiodica=aperiodica/2.5
# cuasiperiodica=cuasiperiodica/2.
# aleatoria=aleatoria/1.5
# caotica=caotica/1.5
# media_objetivo=0.5
# periodica=adjust_mean(periodica,media_objetivo)
# aperiodica=adjust_mean(aperiodica,media_objetivo)
# cuasiperiodica=adjust_mean(cuasiperiodica,media_objetivo)
# caotica=adjust_mean(caotica,media_objetivo)
# aleatoria=adjust_mean(aleatoria,media_objetivo)
#
# print('la periodica es=')
# print(periodica.mean())
# print(np.max(periodica))
# print('la cuasiperiodica es=')
# print(cuasiperiodica.mean())
# print(np.max(cuasiperiodica))
# print('la aperiodica es=')
# print(aperiodica.mean())
# print(np.max(aperiodica))
# print('la aleatoria es=')
# print(aleatoria.mean())
# print(np.max(aleatoria))
# print('la caotica es=')
# print(caotica.mean())
# print(np.max(caotica))


#
# señales=[periodica[:len(periodica)-valsize],cuasiperiodica[:len(periodica)-valsize],aperiodica[:len(periodica)-valsize],aleatoria[:len(periodica)-valsize],caotica[:len(periodica)-valsize]]
# valseñales=[periodica[len(periodica)-valsize:len(periodica)],cuasiperiodica[len(periodica)-valsize:len(periodica)],aperiodica[len(periodica)-valsize:len(periodica)],aleatoria[len(periodica)-valsize:len(periodica)],caotica[len(periodica)-valsize:len(periodica)]]
# #


def create_dataset(señales):
    contador=0
    entradas=[]
    salidas=[]
    while(contador<40000):#contador<120000
        i=np.int(np.round(rand()*4.0))
        señal=señales[i]
        k=np.int(np.round(rand()*len(señal)))
        while(k+timestep>len(señal)-1):
            k=k-1
        n = np.int(np.round(rand()))
        if(n==0):
            muestra=señal[k:k+timestep]
        if (n == 1):
            muestra = np.flip(señal[k:k + timestep],axis=0)

        muestra=muestra*(1.1-rand()*0.2)+(0.1-rand()*0.2)
        entradas.append(muestra)
        salida=0
        if(i==0):
            salida=0
        if(i==1):
            salida=1
        if(i==2):
            salida=2
        if(i==3):
            salida=3
        if(i==4):
            salida=4
        salidas.append(salida)
        contador=contador+1
    return entradas,salidas

# señales=np.array(señales)
# [entradas,salidas]=create_dataset(señales)
# salidas=np.array(salidas)
# entradas=np.array(entradas)
# pesos=[len(salidas[salidas==0])/len(salidas),len(salidas[salidas==1])/len(salidas),len(salidas[salidas==2])/len(salidas),len(salidas[salidas==3])/len(salidas),len(salidas[salidas==4])/len(salidas)]
# from sklearn.utils import class_weight
# class_weightss = class_weight.compute_class_weight('balanced',np.unique(salidas),salidas)
# salidas=keras.utils.to_categorical(salidas,5)
#
# [valentradas,valsalidas]=create_dataset(valseñales)
# valsalidas=np.array(valsalidas)
# valentradas=np.array(valentradas)
# valsalidas=keras.utils.to_categorical(valsalidas,5)



#
# df=pd.read_csv('P1B.dat').values
# dg=pd.read_csv('P2B.dat').values
# dh=pd.read_csv('P3B.dat').values
# di=pd.read_csv('P4B.dat').values
# dj=pd.read_csv('P5B.dat').values



# plt.figure(1)
# plt.plot(periodica[:1000])
# plt.figure(2)
# plt.plot(cuasiperiodica[:1000])
# plt.figure(3)
# plt.plot(aperiodica[:3000])
# plt.figure(4)
# plt.plot(aleatoria[:1000])
# plt.figure(5)
# plt.plot(caotica[:1000])
# plt.show()


# plt.figure(1)
# plt.plot(df[:4000])
# plt.figure(2)
# plt.plot(dg[:4000])
# plt.figure(3)
# plt.plot(dh[:4000])
# plt.figure(4)
# plt.plot(di[:4000])
# plt.figure(5)
# plt.plot(dj[:4000])
# plt.show()

NET=0
if(NET==0):
    def residualblock(entrada,filtros,kernel):
        x = Conv1D(filters=filtros, kernel_size=kernel, strides=1,activation='relu')(entrada)
        x = Conv1D(filters=filtros, kernel_size=kernel, strides=1, padding='same',activation='relu')(x)
        x = Conv1D(filters=filtros, kernel_size=kernel, strides=1, padding='same')(x)

        x2 = Conv1D(filters=filtros, kernel_size=kernel, strides=1)(entrada)
        fusion = Add()([x, x2])
        fusion = Activation("relu")(fusion)
        return fusion

    entrada = Input(shape=(timestep, 1))
    #x=GaussianNoise(0.000002)(entrada)
    x=residualblock(entrada=entrada,filtros=32,kernel=14)
    x = MaxPooling1D(pool_size=5, strides=2)(x)
    x=residualblock(entrada=x,filtros=64,kernel=7)
    x = MaxPooling1D(pool_size=5, strides=2)(x)
    x = Flatten()(x)
    x = Dense(12,activation='relu')(x)
    #x=Dropout(0.3)(x)
    x = Dense(5,activation='softmax')(x)

    model = Model(inputs=entrada, outputs=x)
    model.summary()
    adam = keras.optimizers.Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999,decay=5e-6)


    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])
    if (timestep == 30000):
    # STEPSIZE 1000
        check=keras.callbacks.ModelCheckpoint('pesos_step_50000_2.h5',monitor='loss',mode='auto',save_best_only=True,save_weights_only=True,verbose=1)
        #model.load_weights('pesos_media_reajustada2.h5')
        #history = model.fit(entradas, salidas, epochs=3, batch_size=100, verbose=1,shuffle=True,class_weight=class_weightss,callbacks=[check])
    if (timestep == 5000):
    # STEPSIZE 1000
        check=keras.callbacks.ModelCheckpoint('pesos_step_5000_3.h5',monitor='loss',mode='auto',save_best_only=True,save_weights_only=True,verbose=1)
        #model.load_weights('pesos_media_reajustada2.h5')
        #history = model.fit(entradas, salidas, epochs=5, batch_size=500, verbose=1,shuffle=True,class_weight=class_weightss,callbacks=[check])
    if (timestep == 1000):
    # STEPSIZE 1000
        check=keras.callbacks.ModelCheckpoint('pesos_step_1000.h5',monitor='val_loss',mode='auto',save_best_only=True,save_weights_only=True,verbose=1)
        model.load_weights('pesos_media_reajustada2.h5')
        history = model.fit(entradas, salidas, epochs=20, batch_size=50, verbose=2, validation_data=[valentradas,valsalidas],shuffle=True,class_weight=class_weightss,callbacks=[check])

    #model.load_weights('pesos_step_1000.h5')
    if (timestep == 500):
    #STEPSIZE 500
        check=keras.callbacks.ModelCheckpoint('pesos_media_reajustada2.h5',monitor='val_loss',mode='auto',save_best_only=True,save_weights_only=True,verbose=1)
        model.load_weights('pesos_media_reajustada2.h5')
        history = model.fit(entradas, salidas, epochs=5, batch_size=50, verbose=2, validation_data=[valentradas,valsalidas],shuffle=True,class_weight=class_weightss,callbacks=[check])

    #model.load_weights('pesos_media_reajustada2.h5')
if(timestep==30000):
    # # STEPSIZE 1000
    model.load_weights('pesos_step_50000_2.h5')
if(timestep==5000):
    # # STEPSIZE 1000
    model.load_weights('pesos_step_5000.h5')
if(timestep==1000):
    # # STEPSIZE 1000
    model.load_weights('pesos_step_1000.h5')
if(timestep==500):
    #STEPSIZE 500
    model.load_weights('pesos_media_reajustada2.h5')
if (NET == 1):
    def residualblock(entrada,filtros,kernel):
        x = Conv1D(filters=filtros, kernel_size=kernel, strides=1,activation='relu')(entrada)
        x = Conv1D(filters=filtros, kernel_size=kernel, strides=1, padding='same',activation='relu')(x)
        x = Conv1D(filters=filtros, kernel_size=kernel, strides=1, padding='same')(x)

        x2 = Conv1D(filters=filtros, kernel_size=kernel, strides=1)(entrada)
        fusion = Add()([x, x2])
        fusion = Activation("relu")(fusion)
        return fusion

    entrada = Input(shape=(timestep, 1))
    x=residualblock(entrada=entrada,filtros=32,kernel=14)
    x = MaxPooling1D(pool_size=5, strides=2)(x)
    x=residualblock(entrada=x,filtros=64,kernel=7)
    x = MaxPooling1D(pool_size=5, strides=2)(x)
    x = LSTM(12,return_sequences=False)(x)
    #x = Flatten()(x)
    x = Dense(5,activation='softmax')(x)

    model = Model(inputs=entrada, outputs=x)
    model.summary()
    adam = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])
    check = keras.callbacks.ModelCheckpoint('pesos2.h5', monitor='val_loss', mode='auto', save_best_only=True,
                                            save_weights_only=True, verbose=1)
    history = model.fit(entradas, salidas, epochs=3, batch_size=50, verbose=1,
                        validation_data=[valentradas, valsalidas], shuffle=True, class_weight=class_weightss,
                        callbacks=[check])

    model.load_weights('pesos2.h5')

def create_test(señal):
    contador=0
    entradas=[]
    salidas=[]
    while(contador<90001-1-timestep):#len(señal) y no 60000
        muestra=señal[contador:contador+timestep]
        entradas.append(muestra)
        contador=contador+1
    return entradas

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def calculate(di,ma):
    entrada=di
    entrada=np.array(create_test(entrada))
    prediccion=model.predict(entrada)

    a=np.expand_dims(moving_average(prediccion[:,0],ma),axis=1)
    b = np.expand_dims(moving_average(prediccion[:, 1], ma),axis=1)
    c = np.expand_dims(moving_average(prediccion[:, 2], ma),axis=1)
    d = np.expand_dims(moving_average(prediccion[:, 3], ma),axis=1)
    e = np.expand_dims(moving_average(prediccion[:, 4], ma),axis=1)
    prediccion=np.concatenate([a,b,c,d,e],axis=1)
    #prediccion=np.argmax(prediccion,axis=1)
    return prediccion

# cad='PRUEBAa'
# dk=pd.read_csv(cad+'.dat').values
# dk=adjust_mean(dk,0.0)
# print(dk.mean())
# # print(np.max(dk))
# if(np.min(dk)<0.0):
#     dk=dk+np.abs(np.min(dk))
# dk=dk/np.max(dk)
# prediccion=calculate(dk,2)
# plt.figure(6)
# plt.plot(dk[2:timestep*2,0])
# plt.plot(prediccion[:,0],'-b',label='periodic')
# plt.plot(prediccion[:,1],'-r',label='cuasiperiodic')
# plt.plot(prediccion[:,3],'-g',label='aperiodic')
# plt.plot(prediccion[:,2],'-y',label='random')
# plt.plot(prediccion[:,4],'-m',label='chaotic')
# plt.legend(loc='upper left')
# plt.savefig('figuras/'+cad+'.eps', format='eps')
# f = open('figuras/' + cad+'.txt', 'w')
# f.write('media periodica:')
# f.write(str(prediccion[:, 0].mean() * 100) + '\n')
# f.write('media cuasiperiodica:')
# f.write(str(prediccion[:, 1].mean() * 100) + '\n')
# f.write('media aperiodica:')
# f.write(str(prediccion[:, 3].mean() * 100) + '\n')
# f.write('media aleatoria:')
# f.write(str(prediccion[:, 2].mean() * 100) + '\n')
# f.write('media caotica:')
# f.write(str(prediccion[:, 4].mean() * 100) + '\n')
# f.close()
# plt.show()
MAKEVIDEO=1
if (MAKEVIDEO==1):
    visualize = 0
    ma = 2
    save = 0
    filtrado = 0
    cortas = 0
    dk = pd.read_csv('P1B.dat').values
    # dk=adjust_mean(dk,media_objetivo)
    maximo = np.max(np.abs(dk))
    dk = dk / maximo
    dk = adjust_mean(dk, media_objetivo)
    print(dk.mean())
    prediccion = calculate(dk[::1], ma)
    for j in range(1, timestep):

        plt.figure(1)
        plt.plot(dk[timestep + ma:(timestep+j) * 2, 0])
        plt.plot(prediccion[timestep + ma:(timestep+j) * 2, 0], '-b', label='periodic')
        plt.plot(prediccion[timestep + ma:(timestep+j) * 2, 1], '-r', label='cuasiperiodic')
        plt.plot(prediccion[timestep + ma:(timestep+j) * 2, 2], '-g', label='aperiodic')
        plt.plot(prediccion[timestep + ma:(timestep+j) * 2, 3], '-y', label='random')
        plt.plot(prediccion[timestep + ma:(timestep+j) * 2, 4], '-m', label='chaotic')
        plt.legend(loc='upper left')
        #plt.show()
        plt.savefig('video/%d.png'%(j), format='png')
        #
        plt.close(1)
#
visualize=0
ma=2
save=0
filtrado=0
cortas=0
if(cortas==1):
    cadena2=['datosa','datosap','datosc','datosp','datosq','P1B','P2B','P3B','P4B','P5B','Pruebaa','Pruebac']

    cadena=['datosa','datosap','datosc','datosp','datosq','P1B','P2B','P3B','P4B','P5B','Pruebaa','Pruebac']
    for j in range(5,len(cadena)):
        dk=pd.read_csv(cadena[j]+'.dat').values
        #dk=adjust_mean(dk,media_objetivo)
        maximo = np.max(np.abs(dk))
        dk = dk / maximo
        dk = adjust_mean(dk, media_objetivo)
        print(dk.mean())
        prediccion=calculate(dk[::1],ma)
        if(visualize==1):
            plt.figure(1)
            plt.plot(dk[timestep+ma:timestep*2,0])
            plt.plot(prediccion[:,0],'-b',label='periodic')
            plt.plot(prediccion[:,1],'-r',label='cuasiperiodic')
            plt.plot(prediccion[:,2],'-g',label='aperiodic')
            plt.plot(prediccion[:,3],'-y',label='random')
            plt.plot(prediccion[:,4],'-m',label='chaotic')
            plt.legend(loc='upper left')
            plt.show()
        if(save==1):
            plt.savefig('figuras/'+str(timestep)+'/'+cadena2[j]+'.eps', format='eps')
        #
        print('media periodica')
        print(prediccion[:,0].mean()*100)
        print('media cuasiperiodica')
        print(prediccion[:,1].mean()*100)
        print('media aperiodica')
        print(prediccion[:,2].mean()*100)
        print('media aleatoria')
        print(prediccion[:,3].mean()*100)
        print('media caotica')
        print(prediccion[:,4].mean()*100)
        f = open('figuras/' + str(timestep) +'/'+cadena2[j]+ '.txt', 'w')
        f.write('media periodica:')
        f.write(str(prediccion[:, 0].mean() * 100) + '\n')
        f.write('media cuasiperiodica:')
        f.write(str(prediccion[:, 1].mean() * 100) + '\n')
        f.write('media aperiodica:')
        f.write(str(prediccion[:, 2].mean() * 100) + '\n')
        f.write('media aleatoria:')
        f.write(str(prediccion[:, 3].mean() * 100) + '\n')
        f.write('media caotica:')
        f.write(str(prediccion[:, 4].mean() * 100) + '\n')
        f.close()
        plt.close(1)







    df=adjust_mean(df,media_objetivo)
    print(df.mean())
    # print(np.max(df))
    # df=df/np.max(df)
    prediccion=calculate(df,ma)
    if(visualize==1):
        plt.figure(1)
        plt.plot(df)
        plt.plot(prediccion[:,0],'-b',label='periodic')
        plt.plot(prediccion[:,1],'-r',label='cuasiperiodic')
        plt.plot(prediccion[:,2],'-g',label='aperiodic')
        plt.plot(prediccion[:,3],'-y',label='random')
        plt.plot(prediccion[:,4],'-m',label='chaotic')
        plt.legend(loc='upper left')

        #plt.show()
    if(save==1):
        plt.savefig('figuras/P1B.eps', format='eps')
    print('PB1')
    print('media periodica')
    print(prediccion[:,0].mean()*100)
    print('media cuasiperiodica')
    print(prediccion[:,1].mean()*100)
    print('media aperiodica')
    print(prediccion[:,2].mean()*100)
    print('media aleatoria')
    print(prediccion[:,3].mean()*100)
    print('media caotica')
    print(prediccion[:,4].mean()*100)

    dg=adjust_mean(dg,media_objetivo)
    print(dg.mean())
    # print(np.max(dg))
    # dg=dg/np.max(dg)
    prediccion=calculate(dg,ma)
    if(visualize==1):
        plt.figure(2)
        plt.plot(dg)
        plt.plot(prediccion[:,0],'-b',label='periodic')
        plt.plot(prediccion[:,1],'-r',label='cuasiperiodic')
        plt.plot(prediccion[:,2],'-g',label='aperiodic')
        plt.plot(prediccion[:,3],'-y',label='random')
        plt.plot(prediccion[:,4],'-m',label='chaotic')
        plt.legend(loc='upper left')
    if(save==1):
        plt.savefig('figuras/P2B.eps', format='eps')
    print('PB2')
    print('media periodica')
    print(prediccion[:,0].mean()*100)
    print('media cuasiperiodica')
    print(prediccion[:,1].mean()*100)
    print('media aperiodica')
    print(prediccion[:,2].mean()*100)
    print('media aleatoria')
    print(prediccion[:,3].mean()*100)
    print('media caotica')
    print(prediccion[:,4].mean()*100)

    dh=adjust_mean(dh,media_objetivo)
    print(dh.mean())
    # print(np.max(dh))
    # dh=dh/np.max(dh)
    prediccion=calculate(dh,ma)
    if(visualize==1):
        plt.figure(3)
        plt.plot(dh)
        plt.plot(prediccion[:,0],'-b',label='periodic')
        plt.plot(prediccion[:,1],'-r',label='cuasiperiodic')
        plt.plot(prediccion[:,2],'-g',label='aperiodic')
        plt.plot(prediccion[:,3],'-y',label='random')
        plt.plot(prediccion[:,4],'-m',label='chaotic')
        plt.legend(loc='upper left')

    if(save==1):
        plt.savefig('figuras/P3B.eps', format='eps')
    print('PB3')
    print('media periodica')
    print(prediccion[:,0].mean()*100)
    print('media cuasiperiodica')
    print(prediccion[:,1].mean()*100)
    print('media aperiodica')
    print(prediccion[:,2].mean()*100)
    print('media aleatoria')
    print(prediccion[:,3].mean()*100)
    print('media caotica')
    print(prediccion[:,4].mean()*100)

    di=adjust_mean(di,media_objetivo)
    print(di.mean())
    # print(np.max(di))
    # di=di/np.max(di)
    prediccion=calculate(di,ma)
    if(visualize==1):
        plt.figure(4)
        plt.plot(di)
        plt.plot(prediccion[:,0],'-b',label='periodic')
        plt.plot(prediccion[:,1],'-r',label='cuasiperiodic')
        plt.plot(prediccion[:,2],'-g',label='aperiodic')
        plt.plot(prediccion[:,3],'-y',label='random')
        plt.plot(prediccion[:,4],'-m',label='chaotic')
        plt.legend(loc='upper left')

    if(save==1):
        plt.savefig('figuras/P4B.eps', format='eps')
    print('PB4')
    print('media periodica')
    print(prediccion[:,0].mean()*100)
    print('media cuasiperiodica')
    print(prediccion[:,1].mean()*100)
    print('media aperiodica')
    print(prediccion[:,2].mean()*100)
    print('media aleatoria')
    print(prediccion[:,3].mean()*100)
    print('media caotica')
    print(prediccion[:,4].mean()*100)


    dj=adjust_mean(dj,media_objetivo)
    print(dj.mean())
    # print(np.max(dj))
    # dj=dj/np.max(dj)
    prediccion=calculate(dj,ma)
    if(visualize==1):
        plt.figure(5)
        plt.plot(dj)
        plt.plot(prediccion[:,0],'-b',label='periodic')
        plt.plot(prediccion[:,1],'-r',label='cuasiperiodic')
        plt.plot(prediccion[:,2],'-g',label='aperiodic')
        plt.plot(prediccion[:,3],'-y',label='random')
        plt.plot(prediccion[:,4],'-m',label='chaotic')
        plt.legend(loc='upper left')

    if(save==1):
        plt.savefig('figuras/P5B.eps', format='eps')
    print('PB5')
    print('media periodica')
    print(prediccion[:,0].mean()*100)
    print('media cuasiperiodica')
    print(prediccion[:,1].mean()*100)
    print('media aperiodica')
    print(prediccion[:,2].mean()*100)
    print('media aleatoria')
    print(prediccion[:,3].mean()*100)
    print('media caotica')
    print(prediccion[:,4].mean()*100)


dk=pd.read_csv('PPG24Bc.dat').values
dk=adjust_mean(dk,media_objetivo)
print(dk.mean())
# print(np.max(dk))
dk=dk/np.max(dk)
prediccion=calculate(dk+0.5,ma)
if(visualize==1):
    plt.figure(6)
    plt.plot(dk[timestep+ma:timestep*2,0]+0.5)
    plt.plot(prediccion[:,0],'-b',label='periodic')
    plt.plot(prediccion[:,1],'-r',label='cuasiperiodic')
    plt.plot(prediccion[:,2],'-g',label='aperiodic')
    plt.plot(prediccion[:,3],'-y',label='random')
    plt.plot(prediccion[:,4],'-m',label='chaotic')
    plt.legend(loc='upper left')
if(save==1):
    plt.savefig('figuras/PPG24Bc.eps', format='eps')
print('PPG24Bc')
print('media periodica')
print(prediccion[:,0].mean()*100)
print('media cuasiperiodica')
print(prediccion[:,1].mean()*100)
print('media aperiodica')
print(prediccion[:,2].mean()*100)
print('media aleatoria')
print(prediccion[:,3].mean()*100)
print('media caotica')
print(prediccion[:,4].mean()*100)
f = open('figuras/' + 'PPG24Bc.txt', 'w')
f.write('media periodica:')
f.write(str(prediccion[:, 0].mean() * 100) + '\n')
f.write('media cuasiperiodica:')
f.write(str(prediccion[:, 1].mean() * 100) + '\n')
f.write('media aperiodica:')
f.write(str(prediccion[:, 2].mean() * 100) + '\n')
f.write('media aleatoria:')
f.write(str(prediccion[:, 3].mean() * 100) + '\n')
f.write('media caotica:')
f.write(str(prediccion[:, 4].mean() * 100) + '\n')
f.close()

dk=pd.read_csv('PPG24Bf.dat').values
dk=adjust_mean(dk,media_objetivo)
print(dk.mean())
# print(np.max(dk))
# dk=dk/np.max(dk)
prediccion=calculate(dk,ma)
if(visualize==1):
    plt.figure(7)
    plt.plot(dk[timestep+ma:timestep*2,0])
    plt.plot(prediccion[:,0],'-b',label='periodic')
    plt.plot(prediccion[:,1],'-r',label='cuasiperiodic')
    plt.plot(prediccion[:,2],'-g',label='aperiodic')
    plt.plot(prediccion[:,3],'-y',label='random')
    plt.plot(prediccion[:,4],'-m',label='chaotic')
    plt.legend(loc='upper left')
if(save==1):
    plt.savefig('figuras/PPG24Bf.eps', format='eps')
print('PPG24Bf')
print('media periodica')
print(prediccion[:,0].mean()*100)
print('media cuasiperiodica')
print(prediccion[:,1].mean()*100)
print('media aperiodica')
print(prediccion[:,2].mean()*100)
print('media aleatoria')
print(prediccion[:,3].mean()*100)
print('media caotica')
print(prediccion[:,4].mean()*100)
f = open('figuras/' + 'PPG24Bf.txt', 'w')
f.write('media periodica:')
f.write(str(prediccion[:, 0].mean() * 100) + '\n')
f.write('media cuasiperiodica:')
f.write(str(prediccion[:, 1].mean() * 100) + '\n')
f.write('media aperiodica:')
f.write(str(prediccion[:, 2].mean() * 100) + '\n')
f.write('media aleatoria:')
f.write(str(prediccion[:, 3].mean() * 100) + '\n')
f.write('media caotica:')
f.write(str(prediccion[:, 4].mean() * 100) + '\n')
f.close()


dk=pd.read_csv('PPG24Ec.dat').values
dk=adjust_mean(dk,media_objetivo)
print(dk.mean())
# print(np.max(dk))
# dk=dk/np.max(dk)
prediccion=calculate(dk,ma)
if(visualize==1):
    plt.figure(8)
    plt.plot(dk)
    plt.plot(prediccion[:,0],'-b',label='periodic')
    plt.plot(prediccion[:,1],'-r',label='cuasiperiodic')
    plt.plot(prediccion[:,2],'-g',label='aperiodic')
    plt.plot(prediccion[:,3],'-y',label='random')
    plt.plot(prediccion[:,4],'-m',label='chaotic')
    plt.legend(loc='upper left')
if(save==1):
    plt.savefig('figuras/PPG24Ec.eps', format='eps')
print('PPG24Ec')
print('media periodica')
print(prediccion[:,0].mean()*100)
print('media cuasiperiodica')
print(prediccion[:,1].mean()*100)
print('media aperiodica')
print(prediccion[:,2].mean()*100)
print('media aleatoria')
print(prediccion[:,3].mean()*100)
print('media caotica')
print(prediccion[:,4].mean()*100)


dk=pd.read_csv('PPG24Ef.dat').values
dk=adjust_mean(dk,media_objetivo)
print(dk.mean())
# # print(np.max(dk))
# # dk=dk/np.max(dk)
prediccion=calculate(dk,ma)
#
if(visualize==1):
    plt.figure(9)
    plt.plot(dk)
    plt.plot(prediccion[:,0],'-b',label='periodic')
    plt.plot(prediccion[:,1],'-r',label='cuasiperiodic')
    plt.plot(prediccion[:,2],'-g',label='aperiodic')
    plt.plot(prediccion[:,3],'-y',label='random')
    plt.plot(prediccion[:,4],'-m',label='chaotic')
    plt.legend(loc='upper left')
if(save==1):
    plt.savefig('figuras/PPG24Ef.eps', format='eps')
print('PPG24Ef')
print('media periodica')
print(prediccion[:,0].mean()*100)
print('media cuasiperiodica')
print(prediccion[:,1].mean()*100)
print('media aperiodica')
print(prediccion[:,2].mean()*100)
print('media aleatoria')
print(prediccion[:,3].mean()*100)
print('media caotica')
print(prediccion[:,4].mean()*100)

# dk=pd.read_csv('Pruebaa.dat').values
# dk=adjust_mean(dk,media_objetivo)
# print(dk.mean())

# prediccion=calculate(dk[::1],ma)
# if(visualize==1):
#     plt.figure(10)
#     plt.plot(dk[::1])
#     plt.plot(prediccion[:,0],'-b',label='periodic')
#     plt.plot(prediccion[:,1],'-r',label='cuasiperiodic')
#     plt.plot(prediccion[:,2],'-g',label='aperiodic')
#     plt.plot(prediccion[:,3],'-y',label='random')
#     plt.plot(prediccion[:,4],'-m',label='chaotic')
#     plt.legend(loc='upper left')
# if(save==1):
#     plt.savefig('figuras/Pruebaa.eps', format='eps')
# print('Pruebaa')
# print('media periodica')
# print(prediccion[:,0].mean()*100)
# print('media cuasiperiodica')
# print(prediccion[:,1].mean()*100)
# print('media aperiodica')
# print(prediccion[:,2].mean()*100)
# print('media aleatoria')
# print(prediccion[:,3].mean()*100)
# print('media caotica')
# print(prediccion[:,4].mean()*100)
#
#
# dk=pd.read_csv('Pruebac.dat').values
# dk=adjust_mean(dk,media_objetivo)
# print(dk.mean())
#
# prediccion=calculate(dk[::1],ma)
# if(visualize==1):
#     plt.figure(11)
#     plt.plot(dk[::1])
#     plt.plot(prediccion[:,0],'-b',label='periodic')
#     plt.plot(prediccion[:,1],'-r',label='cuasiperiodic')
#     plt.plot(prediccion[:,2],'-g',label='aperiodic')
#     plt.plot(prediccion[:,3],'-y',label='random')
#     plt.plot(prediccion[:,4],'-m',label='chaotic')
#     plt.legend(loc='upper left')
# if(save==1):
#     plt.savefig('figuras/Pruebac.eps', format='eps')
# print('Pruebac')
# print('media periodica')
# print(prediccion[:,0].mean()*100)
# print('media cuasiperiodica')
# print(prediccion[:,1].mean()*100)
# print('media aperiodica')
# print(prediccion[:,2].mean()*100)
# print('media aleatoria')
# print(prediccion[:,3].mean()*100)
# print('media caotica')
# print(prediccion[:,4].mean()*100)
#
#
# dk=pd.read_csv('PPG24Bf01.dat').values
# dk=adjust_mean(dk,media_objetivo)
# print(dk.mean())
#
# prediccion=calculate(dk[::1],ma)
# if(visualize==1):
#     plt.figure(12)
#     plt.plot(dk[::1])
#     plt.plot(prediccion[:,0],'-b',label='periodic')
#     plt.plot(prediccion[:,1],'-r',label='cuasiperiodic')
#     plt.plot(prediccion[:,2],'-g',label='aperiodic')
#     plt.plot(prediccion[:,3],'-y',label='random')
#     plt.plot(prediccion[:,4],'-m',label='chaotic')
#     plt.legend(loc='upper left')
# if(save==1):
#     plt.savefig('figuras/PPG24Bf01.eps', format='eps')
# print('PPG24Bf01')
# print('media periodica')
# print(prediccion[:,0].mean()*100)
# print('media cuasiperiodica')
# print(prediccion[:,1].mean()*100)
# print('media aperiodica')
# print(prediccion[:,2].mean()*100)
# print('media aleatoria')
# print(prediccion[:,3].mean()*100)
# print('media caotica')
# print(prediccion[:,4].mean()*100)

# dk=pd.read_csv('PPG24Ef01.dat').values
# dk=adjust_mean(dk,media_objetivo)
# print(dk.mean())
#
# prediccion=calculate(dk[::1],ma)
# if(visualize==1):
#     plt.figure(13)
#     plt.plot(dk[::1])
#     plt.plot(prediccion[:,0],'-b',label='periodic')
#     plt.plot(prediccion[:,1],'-r',label='cuasiperiodic')
#     plt.plot(prediccion[:,2],'-g',label='aperiodic')
#     plt.plot(prediccion[:,3],'-y',label='random')
#     plt.plot(prediccion[:,4],'-m',label='chaotic')
#     plt.legend(loc='upper left')
# if(save==1):
#     plt.savefig('figuras/PPG24Ef01.eps', format='eps')
# print('PPG24Ef01')
# print('media periodica')
# print(prediccion[:,0].mean()*100)
# print('media cuasiperiodica')
# print(prediccion[:,1].mean()*100)
# print('media aperiodica')
# print(prediccion[:,2].mean()*100)
# print('media aleatoria')
# print(prediccion[:,3].mean()*100)
# print('media caotica')
# print(prediccion[:,4].mean()*100)

# noise = np.expand_dims(np.random.normal(0,0.07,len(dk)),axis=1)
# dk=noise+0.5
# prediccion=calculate(dk,ma)
# if(visualize==1):
#     plt.figure(14)
#     plt.plot(dk[::1])
#     plt.plot(prediccion[:,0],'-b',label='periodic')
#     plt.plot(prediccion[:,1],'-r',label='cuasiperiodic')
#     plt.plot(prediccion[:,2],'-g',label='aperiodic')
#     plt.plot(prediccion[:,3],'-y',label='random')
#     plt.plot(prediccion[:,4],'-m',label='chaotic')
#     plt.legend(loc='upper left')
# if(save==1):
#     plt.savefig('figuras/noise.eps', format='eps')
# print('noise')
# print('media periodica')
# print(prediccion[:,0].mean()*100)
# print('media cuasiperiodica')
# print(prediccion[:,1].mean()*100)
# print('media aperiodica')
# print(prediccion[:,2].mean()*100)
# print('media aleatoria')
# print(prediccion[:,3].mean()*100)
# print('media caotica')
# print(prediccion[:,4].mean()*100)

dk=pd.read_csv('PPG24Bf.dat').values
dk=adjust_mean(dk,media_objetivo)
print(dk.mean())
# print(np.max(dk))
noise = np.expand_dims(np.random.normal(0,0.01,len(dk)),axis=1)
rmss=np.sqrt(np.mean(dk*dk))
rmsn=np.sqrt(np.mean(noise*noise))
SNR=20*np.log(rmss/rmsn)
dk=dk+noise
prediccion=calculate(dk,ma)
plt.figure(15)
plt.plot(dk[timestep+ma:timestep*2,0])
plt.plot(prediccion[:,0],'-b',label='periodic')
plt.plot(prediccion[:,1],'-r',label='cuasiperiodic')
plt.plot(prediccion[:,2],'-g',label='aperiodic')
plt.plot(prediccion[:,3],'-y',label='random')
plt.plot(prediccion[:,4],'-m',label='chaotic')
plt.legend(loc='upper left')
if(save==1):
    plt.savefig('figuras/PPG24Bf_noise_001.eps', format='eps')
print('figuras/PPG24Bf_noise_001.eps')
print('media periodica')
print(prediccion[:,0].mean()*100)
print('media cuasiperiodica')
print(prediccion[:,1].mean()*100)
print('media aperiodica')
print(prediccion[:,2].mean()*100)
print('media aleatoria')
print(prediccion[:,3].mean()*100)
print('media caotica')
print(prediccion[:,4].mean()*100)
f = open('figuras/' + 'PPG24Bf_noise_001.txt', 'w')
f.write('media periodica:')
f.write(str(prediccion[:, 0].mean() * 100) + '\n')
f.write('media cuasiperiodica:')
f.write(str(prediccion[:, 1].mean() * 100) + '\n')
f.write('media aperiodica:')
f.write(str(prediccion[:, 2].mean() * 100) + '\n')
f.write('media aleatoria:')
f.write(str(prediccion[:, 3].mean() * 100) + '\n')
f.write('media caotica:')
f.write(str(prediccion[:, 4].mean() * 100) + '\n')
f.close()


dk=pd.read_csv('PPG24Bf.dat').values
dk=adjust_mean(dk,media_objetivo)
print(dk.mean())
# print(np.max(dk))
noise = np.expand_dims(np.random.normal(0,0.06,len(dk)),axis=1)
dk=dk+noise
prediccion=calculate(dk,ma)
plt.figure(16)
plt.plot(dk[timestep+ma:timestep*2,0])
plt.plot(prediccion[:,0],'-b',label='periodic')
plt.plot(prediccion[:,1],'-r',label='cuasiperiodic')
plt.plot(prediccion[:,2],'-g',label='aperiodic')
plt.plot(prediccion[:,3],'-y',label='random')
plt.plot(prediccion[:,4],'-m',label='chaotic')
plt.legend(loc='upper left')
if(save==1):
    plt.savefig('figuras/PPG24Bf_noise_005.eps', format='eps')
print('figuras/PPG24Bf_noise_005.eps')
print('media periodica')
print(prediccion[:,0].mean()*100)
print('media cuasiperiodica')
print(prediccion[:,1].mean()*100)
print('media aperiodica')
print(prediccion[:,2].mean()*100)
print('media aleatoria')
print(prediccion[:,3].mean()*100)
print('media caotica')
print(prediccion[:,4].mean()*100)
f = open('figuras/' + 'PPG24Bf_noise_005.txt', 'w')
f.write('media periodica:')
f.write(str(prediccion[:, 0].mean() * 100) + '\n')
f.write('media cuasiperiodica:')
f.write(str(prediccion[:, 1].mean() * 100) + '\n')
f.write('media aperiodica:')
f.write(str(prediccion[:, 2].mean() * 100) + '\n')
f.write('media aleatoria:')
f.write(str(prediccion[:, 3].mean() * 100) + '\n')
f.write('media caotica:')
f.write(str(prediccion[:, 4].mean() * 100) + '\n')
f.close()


dk=pd.read_csv('PPG24Bf.dat').values
dk=adjust_mean(dk,media_objetivo)
print(dk.mean())
# print(np.max(dk))
noise = np.expand_dims(np.random.normal(0,0.1,len(dk)),axis=1)
dk=dk+noise
prediccion=calculate(dk,ma)
plt.figure(17)
plt.plot(dk[timestep+ma:timestep*2,0])
plt.plot(prediccion[:,0],'-b',label='periodic')
plt.plot(prediccion[:,1],'-r',label='cuasiperiodic')
plt.plot(prediccion[:,2],'-g',label='aperiodic')
plt.plot(prediccion[:,3],'-y',label='random')
plt.plot(prediccion[:,4],'-m',label='chaotic')
plt.legend(loc='upper left')
if(save==1):
    plt.savefig('figuras/PPG24Bf_noise_01.eps', format='eps')
print('figuras/PPG24Bf_noise_01.eps')
print('media periodica')
print(prediccion[:,0].mean()*100)
print('media cuasiperiodica')
print(prediccion[:,1].mean()*100)
print('media aperiodica')
print(prediccion[:,2].mean()*100)
print('media aleatoria')
print(prediccion[:,3].mean()*100)
print('media caotica')
print(prediccion[:,4].mean()*100)
f = open('figuras/' + 'PPG24Bf_noise_01.txt', 'w')
f.write('media periodica:')
f.write(str(prediccion[:, 0].mean() * 100) + '\n')
f.write('media cuasiperiodica:')
f.write(str(prediccion[:, 1].mean() * 100) + '\n')
f.write('media aperiodica:')
f.write(str(prediccion[:, 2].mean() * 100) + '\n')
f.write('media aleatoria:')
f.write(str(prediccion[:, 3].mean() * 100) + '\n')
f.write('media caotica:')
f.write(str(prediccion[:, 4].mean() * 100) + '\n')
f.close()


if(visualize==1):
    plt.show()

# from fpdf import FPDF
# pdf = FPDF()
# # imagelist is the list with all image filenames
# for image in imagelist:
#     pdf.add_page()
#     pdf.image(image,x,y,w,h)
# pdf.output("yourfile.pdf", "F")

# cadena2=['SNA_ap','SNA_p','SNA_q','SNA_c']
#
# cadena=['datos_fisiologicos/datos_fisiológicos/sna_type/SNA_ap','datos_fisiologicos/datos_fisiológicos/sna_type/SNA_p','datos_fisiologicos/datos_fisiológicos/sna_type/SNA_q','datos_fisiologicos/datos_fisiológicos/sna_type/SNA_c']
# for j in range(0,len(cadena)):
#     dk=pd.read_csv(cadena[j]+'.dat').values
#     #dk=adjust_mean(dk,media_objetivo)
#     maximo = np.max(np.abs(dk))
#     dk=dk+0.5
#     print(dk.mean())
#     prediccion=calculate(dk[::1],ma)
#     if(visualize==1):
#         plt.figure(1)
#         plt.plot(dk[::1])
#         plt.plot(prediccion[:,0],'-b',label='periodic')
#         plt.plot(prediccion[:,1],'-r',label='cuasiperiodic')
#         plt.plot(prediccion[:,2],'-g',label='aperiodic')
#         plt.plot(prediccion[:,3],'-y',label='random')
#         plt.plot(prediccion[:,4],'-m',label='chaotic')
#         plt.legend(loc='upper left')
#     if(save==1):
#         plt.savefig('figuras/'+str(timestep)+'/'+cadena2[j]+'.eps', format='eps')
#
#     print('media periodica')
#     print(prediccion[:,0].mean()*100)
#     print('media cuasiperiodica')
#     print(prediccion[:,1].mean()*100)
#     print('media aperiodica')
#     print(prediccion[:,2].mean()*100)
#     print('media aleatoria')
#     print(prediccion[:,3].mean()*100)
#     print('media caotica')
#     print(prediccion[:,4].mean()*100)
#     f = open('figuras/' + str(timestep) +'/'+cadena2[j]+ '.txt', 'w')
#     f.write('media periodica:')
#     f.write(str(prediccion[:, 0].mean() * 100) + '\n')
#     f.write('media cuasiperiodica:')
#     f.write(str(prediccion[:, 1].mean() * 100) + '\n')
#     f.write('media aperiodica:')
#     f.write(str(prediccion[:, 2].mean() * 100) + '\n')
#     f.write('media aleatoria:')
#     f.write(str(prediccion[:, 3].mean() * 100) + '\n')
#     f.write('media caotica:')
#     f.write(str(prediccion[:, 4].mean() * 100) + '\n')
#     f.close()
#     plt.close(1)