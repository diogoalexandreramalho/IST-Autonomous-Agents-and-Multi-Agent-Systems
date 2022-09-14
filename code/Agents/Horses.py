import sys
sys.path.append("../")
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from keras.optimizers import SGD
import chess
from board import SimpleBoard
import chess.engine
from players import stockfish_player, random_player
from StateRepresentations.pieceList import numberToSquare, boardToList, listIndex, correctOrder, squareToNumber
import pickle
#Tuner
import kerastuner
from kerastuner import Hyperband
import tensorflow as tf
from tensorflow import keras

import IPython

from Agent import Agent
import random


class Horses(Agent):
    '''
    OUTPUT : clockwise -> [ UUR1 URR1 DRR1 DDR1 DDL1 DLL1 LLU1 UUL1 UUR2 URR2 RRD2 DDR2 DDL2 DLL2 LLU2 UUL2 ]
    - U->UP, R->RIGHT, L->LEFT, D->DOWN
    1 and 2 are the horses

            [7] [0]                       [15] [8]
        [6]        [1]                [14]        [9]
              N                              N
        [5]        [2]                [13]        [10]
           [4] [3]                       [12] [11]
    horseNum is 0 or 1
    '''

    def uciToY(self,uci,state):
        if(state[1]!=0 and chess.SQUARE_NAMES[state[1]-1] == uci[0:2]):
            horseNum = 0
        elif(state[6]!=0 and chess.SQUARE_NAMES[state[6]-1]==uci[0:2]):
            horseNum = 1
        elif(state[1]==0 and state[6]==0):
            pass
        else:
            print("No horse in movePlay?")

        coord = (ord(uci[2])-ord(uci[0]),int(uci[3])-int(uci[1]))
        outputIndex = [[4,3],[5,2],[6,1],[7,0]]
        if coord[1] > 0:
            if coord[0] > 0:
                return outputIndex[coord[1]+1][1] + horseNum*8
            else:
                return outputIndex[coord[1] + 1][0] + horseNum*8
        else:
            if coord[0] > 0:
                return outputIndex[coord[1]+2][1] + horseNum*8
            else:
                return outputIndex[coord[1]+2][0] + horseNum*8

    def yToUci(self,y,state):
        #increments on the board
        letter = [1, 2,  2,  1, -1, -2, -2, -1] #increments on the letters dimension
        number = [2, 1, -1, -2, -2, -1,  1,  2] #incremeents on the number dimension

        if y<8: #first horse
            if state[1]==0:
                return ""
            horseToplay = chess.SQUARE_NAMES[state[1]-1]


        else:#second horse
            if state[6]==0:#if horse dead
                return ""
            horseToplay = chess.SQUARE_NAMES[state[6]-1]
            y = y - 8

        toSquareLetter = chr(ord(horseToplay[0])+letter[y])
        #if toSquareLetter not in ['a','b','c','d','e','f','g','h']:
        #    return ""
        toSquareNumber =   chr(int(horseToplay[1])+number[y]+48)
        #if toSquareNumber not in ['1','2','3','4','5','6','7','8']:
        #    return ""

        toSquare = toSquareLetter + toSquareNumber
        uci = horseToplay[0:2] + toSquare
        return uci


    def __init__(self,loadWeights):
        self.train_x = list()
        self.train_y = list()
        self.valid_x = list()
        self.valid_y = list()
        self.network= self.build_compile_model()
        if(loadWeights==True):
            self.loadWeights()

    def build_compile_model(self):  # The structure of the network
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(32,)))
        model.add(Dense(352, activation='relu'))
        model.add(Dense(16,activation='softmax'));
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer= 'Adam',metrics=['accuracy'])
        return model

    def store(self, boardList,y): #putExperienceOnQueue
        self.train_x.append(boardList)
        self.train_y.append(y)

    def serializeStorage(self): #SerializeSamples
        with open('./Samples/HorsesX', 'wb') as handle:
            pickle.dump(self.train_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('./Samples/HorsesY', 'wb') as handle:
            pickle.dump(self.train_y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def loadStorage(self):
        HorsesX = open('./Samples/HorsesX', 'rb')
        self.train_x.extend(pickle.load(HorsesX))
        HorsesY = open('./Samples/HorsesY', 'rb')
        self.train_y.extend(pickle.load(HorsesY))
        HorsesX.close()
        HorsesY.close()


    def serializeWeights(self, path="./checkpoints/Horses/my_checkpoint"): #serialize Weights so we can load it after
        self.network.save_weights(path)

    def loadWeights(self, path="./checkpoints/Horses/my_checkpoint"): #load serialized Weights
        self.network.load_weights(path)

    def predict(self,board):#example x=np.array([self.train_x[0]])
        return self.network.predict(np.array([board]))

    def train(self):
        '''
        for i in range(20):
            print("i="+str(i)+" " + str(self.train_x[i] ))
            self.network.fit(np.array(self.train_x[i+0:i+1000]), np.array(self.train_y[i+0:i+1000]).T)
        '''
        print(len(self.train_x))
        print(len(self.train_y))
        Tx = np.array(self.train_x)
        print(len(Tx))
        self.network.fit(Tx, np.array(self.train_y).T, batch_size=100, epochs=30, validation_split=0.3)
        print("did it")

    def model_builder(self,hp):
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(32,)))

        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 32-512
        hp_units1 = hp.Int('units1', min_value=32, max_value=512, step=32)
        hp_units2 = hp.Int('units2', min_value=32, max_value=512, step=32)
        model.add(keras.layers.Dense(units=hp_units1, activation='relu'))
        model.add(keras.layers.Dense(units=hp_units2, activation='relu'))
        model.add(keras.layers.Dense(16, activation="softmax"))

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        adam = keras.optimizers.Adam(learning_rate=hp_learning_rate)

        sgd = keras.optimizers.SGD(learning_rate=hp_learning_rate, momentum=0.0, nesterov=False, name="SGD")
        model.compile(optimizer=sgd,
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

    def run_tuner(self):
        class ClearTrainingOutput(tf.keras.callbacks.Callback):
            print("hey")
       #     def on_train_end(*args, **kwargs):
       #         IPython.display.clear_output(wait=True)


        tuner = Hyperband(self.model_builder,
                          objective='val_accuracy',
                          max_epochs=5,
                          factor=3,
                          directory='./tunner',
                          project_name='Horses')

        tuner.search(np.array(self.train_x),np.array(self.train_y).T, epochs=10, validation_data = (np.array(self.valid_x),np.array(self.valid_y).T),
                     callbacks=[ClearTrainingOutput()])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(best_hps)
        self.network = tuner.hypermodel.build(best_hps)
        print(self.network.summary())
        self.network.fit(np.array(self.train_x), np.array(self.train_y).T, epochs = 15, validation_data = (np.array(self.test_x),np.array(self.test_y).T))

    def load_network_tuner(self):
        tuner = Hyperband(self.model_builder,
                          objective='val_accuracy',
                          max_epochs=5,
                          factor=3,
                          directory='./tunner',
                          project_name='Horses')

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.network = tuner.hypermodel.build(best_hps)
