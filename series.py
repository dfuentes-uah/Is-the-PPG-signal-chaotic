
import tensorflow as tf
import numpy as np
import keras
import math
from pandas import read_csv
from keras.datasets import cifar10
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,Input,Conv3D,BatchNormalization,ConvLSTM2D,LSTM,GRU
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import SGD,Adam
from keras.preprocessing import image
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
acumulador=[]
offset=5
for offset in range(0,100):
    # Conversion Array-Matriz
    def create_base_datos(datos, mirar_atras=1000):
        datosX, datosY = [], []
        for i in range(len(datos)-mirar_atras-offset):
            a = datos[i:(i+mirar_atras), 0]
            datosX.append(a)
            datosY.append(datos[i + mirar_atras+offset, 0])
        return np.array(datosX), np.array(datosY)

    # aleatoria=pd.read_csv('datosa.dat').values
    # aperiodica=pd.read_csv('datosap.dat').values
    # periodica=pd.read_csv('datosp.dat').values
    # cuasiperiodica=pd.read_csv('datosq.dat').values
    # caotica=pd.read_csv('datosc.dat').values
    #Cargar Datos
    modo=0
    if(modo==0):
        cadena='datosa'
    if(modo==1):
        cadena='datosap'
    if(modo==2):
        cadena='datosp'
    if(modo==3):
        cadena='datosq'
    if(modo==4):
        cadena='datosc'
    dataframe = read_csv(cadena+'.dat')
    datos = dataframe.values
    datos = datos.astype('float32')

    #Normalizacion de Datos
    escalado = MinMaxScaler(feature_range=(0, 1))
    datos = escalado.fit_transform(datos)

    #Division en Train/Test
    tsize = int(len(datos) * 0.67)
    testsize = len(datos) - tsize
    entrenamiento, test = datos[0:tsize,:], datos[tsize:len(datos),:]

    # Cambio de tamaños de train y test
    mirar_atras= 1200
    tX, tY = create_base_datos(entrenamiento, mirar_atras)
    testX, testY = create_base_datos(test, mirar_atras)
    tX = np.reshape(tX, (tX.shape[0], 1, tX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Creacion de Red con LSTM
    input=Input(shape=(1,mirar_atras))
    x=LSTM(100,return_sequences=True,activation='tanh')(input)
    x=LSTM(20,activation='tanh')(input)
    #x=GRU(10)(input)
    x=Dense(1,activation='linear')(x)
    model=Model(inputs=input,outputs=x)
    model.compile(loss='mae', optimizer='adam')
    check=keras.callbacks.ModelCheckpoint(str(modo)+'.h5',monitor='val_loss',mode='auto',save_best_only=True,save_weights_only=True,verbose=1)

    history=model.fit(tX, tY, epochs=100, batch_size=50, verbose=2,validation_data=[testX,testY],callbacks=[check])
    model.load_weights(str(modo)+'.h5')
    # plt.figure(1)
    # plt.plot(history.history['loss'])
    # plt.title('Perdidas del Modelo')
    # plt.ylabel('Perdidas')
    # plt.xlabel('Epocas')
    # plt.legend(['Entrenamiento'], loc='upper left')


    # Realizacion de predicciones
    prediciones_entrenamiento = model.predict(tX)
    prediciones_test = model.predict(testX)

    # Inversion de las predicciones para calcular su error
    prediciones_entrenamiento = escalado.inverse_transform(prediciones_entrenamiento)
    tY = escalado.inverse_transform([tY])
    prediciones_test = escalado.inverse_transform(prediciones_test)
    testY = escalado.inverse_transform([testY])

    # Calculo de la raiz del error cuadratico medio o RMSE
    Puntuacion_Train = math.sqrt(mean_squared_error(tY[0,:], prediciones_entrenamiento[:,0]))
    Puntuacion_Test = math.sqrt(mean_squared_error(testY[0], prediciones_test[:,0]))
    print('Puntuacion Train: %.2f RMSE  y Puntuacion Test: %.2f RMSE' % (Puntuacion_Train,Puntuacion_Test))
    acumulador.append(Puntuacion_Test)
acumulador=np.array(acumulador)
np.save(cadena+'.npy',acumulador)
plt.figure(4)
plt.plot(acumulador)
plt.show()

# # Desplazamiento de predicciones de entrenamiento
# plot_prediccion_entrenamiento = np.empty_like(datos)
# plot_prediccion_entrenamiento[:, :] = np.nan
# plot_prediccion_entrenamiento[mirar_atras:len(prediciones_entrenamiento)+mirar_atras+offset, :] = prediciones_entrenamiento
#
# # Desplazamiento de predicciones de test
# plot_prediccion_test = np.empty_like(datos)
# plot_prediccion_test[:, :] = np.nan
# plot_prediccion_test[len(prediciones_entrenamiento)+(mirar_atras*2)+1:len(datos), :] = prediciones_test
#
# # Mostrar predicciones y datos
# plt.figure(2)
# plt.plot(escalado.inverse_transform(datos))
# #plt.plot(plot_prediccion_entrenamiento)
# plt.plot(plot_prediccion_test)
# plt.show()

