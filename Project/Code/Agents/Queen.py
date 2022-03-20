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

class Queen(Agent):
    '''

    '''

    def uciToY(self,uci,state):
        if(chess.SQUARE_NAMES[state[3]-1] != uci[0:2]):
            print("No queen in movePlay?")
        else:
            pass

        val = -1
        if(uci[0] == uci[2] and uci[1] < uci[3]): #N
            val = 0
        elif(uci[0] == uci[2] and uci[1] > uci[3]): #S
            val = 1
        elif(uci[0] < uci[2] and uci[1] == uci[3]): #E
            val = 2
        elif(uci[0] > uci[2] and uci[1] == uci[3]): #W
            val = 3
        if(uci[0] < uci[2] and uci[1] < uci[3]): #NE
            val = 4
        elif(uci[0] > uci[2] and uci[1] < uci[3]): #NW
            val = 5
        elif(uci[0] < uci[2] and uci[1] > uci[3]): #SE
            val = 6
        elif(uci[0] > uci[2] and uci[1] > uci[3]): #SW
            val = 7
        else:
            pass

        from_square, to_square = chess.SQUARE_NAMES.index(uci[0:2]), chess.SQUARE_NAMES.index(uci[2:4])
        dist = chess.square_distance(from_square, to_square)
        return 7*val + dist - 1 # 0 a 55


    def yToUci(self,y,state):
        print("here?")
        from_square = chess.SQUARE_NAMES[state[3]-1]
        dist = (y+1)%7
        val = (y+1-dist) // 7
        if(val == 0): #N
            to_square = from_square[0] + str(int(from_square[1]) + dist)
        elif(val == 1): #S
            to_square = from_square[0] + str(int(from_square[1]) - dist)
        elif(val == 2): #E
            to_square = chr(ord(from_square[0]) + dist) + from_square[1]
        elif(val == 3): #W
            to_square = chr(ord(from_square[0]) - dist) + from_square[1]
        elif(val == 4): #NE
            to_square = chr(ord(from_square[0]) + dist) + str(int(from_square[1]) + dist)
        elif(val == 5): #NW
            to_square = chr(ord(from_square[0]) - dist) + str(int(from_square[1]) + dist)
        elif(val == 6): #SE
            to_square = chr(ord(from_square[0]) + dist) + str(int(from_square[1]) - dist)
        elif(val == 7): #SW
            to_square = chr(ord(from_square[0]) - dist) + str(int(from_square[1]) - dist)
        else:
            pass

        return from_square + to_square


    def __init__(self,loadWeights):
        #self.train_x = deque()
        #self.train_y = deque()
        self.train_x = list()
        self.train_y = list()
        self.valid_x = list()
        self.valid_y = list()
        self.network= self.build_compile_model()
        if(loadWeights==True):
            self.loadWeights()

    def build_compile_model(self):  # The structure of the network
        model = Sequential()
        model.add(Dense(160, activation='relu', input_shape=(32,)))
        model.add(Dense(480, activation='relu'))
        model.add(Dense(56, activation="softmax"))
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='sgd',metrics=['accuracy'])
        return model

    def store(self, boardList,y): #putExperienceOnQueue
        self.train_x.append(boardList)
        self.train_y.append(y)

    def createValidationSet(self,percentage):
        lensamples = int(len(self.train_x)*(1-percentage))
        temptrain_x = self.train_x[0:lensamples]
        temptrain_y = self.train_y[0:lensamples]
        self.valid_x =  self.train_x[lensamples:]
        self.valid_y = self.train_y[lensamples:]
        self.train_x = temptrain_x
        self.train_y = temptrain_y


    def serializeStorage(self): #SerializeSamples
        with open('./Samples/QueenX', 'wb') as handle:
            pickle.dump(self.train_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('./Samples/QueenY', 'wb') as handle:
            pickle.dump(self.train_y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def loadStorage(self):
        QueenX = open('./Samples/QueenX', 'rb')
        self.train_x.extend(pickle.load(QueenX))
        QueenY = open('./Samples/QueenY', 'rb')
        self.train_y.extend(pickle.load(QueenY))
        QueenX.close()
        QueenY.close()



    def serializeWeights(self, path="./checkpoints/Queen/my_checkpoint"): #serialize Weights so we can load it after
        self.network.save_weights(path)

    def loadWeights(self, path="./checkpoints/Queen/my_checkpoint"): #load serialized Weights
        self.network.load_weights(path)

    def predict(self,board):#example x=np.array([self.train_x[0]])
        return self.network.predict(np.array([board]))

    def train(self):
        '''
        for i in range(len(self.train_x)):
            self.network.fit(np.array([self.train_x[i]]), np.array([self.train_y[i]]))

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
        hp_units3 = hp.Int('units3', min_value=32, max_value=512, step=32)
        hp_units4 = hp.Int('units4', min_value=32, max_value=512, step=32)
        model.add(keras.layers.Dense(units=hp_units1, activation='relu'))
        model.add(keras.layers.Dense(units=hp_units2, activation='relu'))
        #model.add(keras.layers.Dense(units=hp_units3, activation='relu'))
        #model.add(keras.layers.Dense(units=hp_units4, activation='relu'))
        model.add(keras.layers.Dense(56, activation="softmax"))

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
            print("hey:",len(self.train_x))
       #     def on_train_end(*args, **kwargs):
       #         IPython.display.clear_output(wait=True)


        tuner = Hyperband(self.model_builder,
                          objective='val_accuracy',
                          max_epochs=5,
                          factor=3,
                          directory='./tunner',
                          project_name='Queen')

        tuner.search(np.array(self.train_x),np.array(self.train_y).T, epochs=10, validation_data = (np.array(self.valid_x),np.array(self.valid_y).T),
                     callbacks=[ClearTrainingOutput()])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        #print("best_hps:",best_hps)
        self.network = tuner.hypermodel.build(best_hps)
        #print("here's the summary:",self.network.summary())
        self.network.fit(np.array(self.train_x), np.array(self.train_y).T, epochs = 30, validation_data = (np.array(self.test_x),np.array(self.test_y).T))

    def load_network_tuner(self):
            tuner = Hyperband(self.model_builder,
                          objective='val_accuracy',
                          max_epochs=5,
                          factor=3,
                          directory='./tunner',
                          project_name='Queen')

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            self.network = tuner.hypermodel.build(best_hps)



'''testing
uci = "d4b2"
val = -1
if(uci[0] == uci[2] and uci[1] < uci[3]): #N
    val = 0
elif(uci[0] == uci[2] and uci[1] > uci[3]): #S
    val = 1
elif(uci[0] < uci[2] and uci[1] == uci[3]): #E
    val = 2
elif(uci[0] > uci[2] and uci[1] == uci[3]): #W
    val = 3
if(uci[0] < uci[2] and uci[1] < uci[3]): #NE
    val = 4
elif(uci[0] > uci[2] and uci[1] < uci[3]): #NW
    val = 5
elif(uci[0] < uci[2] and uci[1] > uci[3]): #SE
    val = 6
elif(uci[0] > uci[2] and uci[1] > uci[3]): #SW
    val = 7
else:
    pass

print("val",val)
from_square, to_square = chess.SQUARE_NAMES.index(uci[0:2]), chess.SQUARE_NAMES.index(uci[2:4])
print("from sq uci",  from_square)
print("to sq uci", to_square)
dist = chess.square_distance(from_square, to_square)
print("dist uci", dist)
print("return uci", 7*val + dist - 1)
y = 7*val + dist - 1

from_square = "d4"
print("from sq y", from_square)
dist = (y+1)%7
print("dist y", dist)
val = (y + 1 - dist) // 7
print("val y", val)
if(val == 0): #N
    to_square = from_square[0] + str(int(from_square[1]) + dist)
elif(val == 1): #S
    to_square = from_square[0] + str(int(from_square[1]) - dist)
elif(val == 2): #E
    to_square = chr(ord(from_square[0]) + dist) + from_square[1]
elif(val == 3): #W
    to_square = chr(ord(from_square[0]) - dist) + from_square[1]
elif(val == 4): #NE
    to_square = chr(ord(from_square[0]) + dist) + str(int(from_square[1]) + dist)
elif(val == 5): #NW
    to_square = chr(ord(from_square[0]) - dist) + str(int(from_square[1]) + dist)
elif(val == 6): #SE
    to_square = chr(ord(from_square[0]) + dist) + str(int(from_square[1]) - dist)
elif(val == 7): #SW
    to_square = chr(ord(from_square[0]) - dist) + str(int(from_square[1]) - dist)
else:
    pass

print("to sq y", to_square)
print("ret", from_square + to_square)
'''


