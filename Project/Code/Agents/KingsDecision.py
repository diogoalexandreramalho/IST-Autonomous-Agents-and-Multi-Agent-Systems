import sys
sys.path.append("../")
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import SGD
import chess
import random
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

class KingsDecision:

    '''
        [Rook, Horse, Bishop, Queen, King , Pawn]

    '''

    def __init__(self,rook,horse,bishop,queen,king,pawn):
        self.train_x = list()
        self.train_y = list()
        self.test_x = list()
        self.test_y = list()
        self.agents = [rook,horse,bishop,queen,king,pawn]
        self.network = self.build_compile_model()


    def yToAgent(self,y):
        try:
            return self.agents[y]
        except:
            return random.choice(self.agents)



    def loadStorage(self):
        i = 0
        RooksX = open('./Samples/RooksX', 'rb')
        self.train_x.extend(pickle.load(RooksX))
        RooksX.close()

        for i in range(i,len(self.train_x)):
            self.train_y.append(0)
        print("i="+str(i)+" Rook how much?" + str(len(self.train_x)))


        HorsesX = open('./Samples/HorsesX', 'rb')
        self.train_x.extend(pickle.load(HorsesX))
        HorsesX.close()

    
        for i in range(i+1,len(self.train_x)):
            self.train_y.append(1)
        print("i="+str(i)+" Horses how much?" + str(len(self.train_x)))

        BishopX = open('./Samples/BishopsX', 'rb')
        self.train_x.extend(pickle.load(BishopX))
        BishopX.close()

        for i in range(i+1,len(self.train_x)):
            self.train_y.append(2)
        print("i="+str(i)+" Bishop how much?" + str(len(self.train_x)))

        QueenX = open('./Samples/QueenX', 'rb')
        self.train_x.extend(pickle.load(QueenX))
        QueenX.close()

        for i in range(i+1,len(self.train_x)):
            self.train_y.append(3)
        print("i="+str(i)+" Queen how much?" + str(len(self.train_x)))

        KingX = open('./Samples/KingX', 'rb')
        self.train_x.extend(pickle.load(KingX))
        KingX.close()

        for i in range(i+1,len(self.train_x)):
            self.train_y.append(4)
        print("i="+str(i)+" King how much?" + str(len(self.train_x)))

        PawnsX = open('./Samples/PawnsX', 'rb')
        self.train_x.extend(pickle.load(PawnsX))
        PawnsX.close()

        for i in range(i+1,len(self.train_x)):
            self.train_y.append(5)
        print("i="+str(i)+" Pawns how much?" + str(len(self.train_x)))

        c = list(zip(self.train_x,self.train_y))
        random.shuffle(c)
        tup = list(c)
        self.train_x = list()
        self.train_y = list()
        for i in range(len(tup)):
            self.train_x.append(tup[i][0])
            self.train_y.append(tup[i][1])

    def createValandTestSets(self,vPercentage,tPercentage):
        trainLimit= int(len(self.train_x)*(1-vPercentage-tPercentage))
        validLimit = int(len(self.train_x)*(1-tPercentage))
        testLimit = int(len(self.train_x))

        temptrain_x = self.train_x[0:trainLimit]
        temptrain_y = self.train_y[0:trainLimit]
        self.valid_x =  self.train_x[trainLimit:validLimit]
        self.valid_y = self.train_y[trainLimit:validLimit]
        self.test_x = self.train_x[validLimit:testLimit]
        self.test_y = self.train_y[validLimit:testLimit]
        self.train_x = temptrain_x
        self.train_y = temptrain_y


    def serializeWeights(self, path="./checkpoints/KingsDecision/my_checkpoint"): #serialize Weights so we can load it after
        self.network.save_weights(path)

    def loadWeights(self, path="./checkpoints/KingsDecision/my_checkpoint"): #load serialized Weights
        self.network.load_weights(path)

    def predict(self,board):#example x=np.array([self.train_x[0]])
        return self.network.predict(np.array([board]))

    def model_builder(self,hp):
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(32,)))

        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 32-512
        hp_units1 = hp.Int('units1', min_value=128, max_value=512, step=128)
        hp_units2 = hp.Int('units2', min_value=128, max_value=512, step=128)
        hp_units3 = hp.Int('units3', min_value=128, max_value=512, step=128)
        hp_units4 = hp.Int('units4', min_value=128, max_value=512, step=128)
        model.add(keras.layers.Dense(units=hp_units1, activation='relu'))
        model.add(keras.layers.Dense(units=hp_units2, activation='relu'))
        model.add(keras.layers.Dense(units=hp_units3, activation='relu'))
        model.add(keras.layers.Dense(units=hp_units4, activation='relu'))
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
                          directory='./tunner/KingsDecision',
                          project_name='KingsDecision')

        tuner.search(np.array(self.train_x),np.array(self.train_y).T, epochs=10, validation_data = (np.array(self.valid_x),np.array(self.valid_y).T),
                     callbacks=[ClearTrainingOutput()])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(best_hps)
        self.network = tuner.hypermodel.build(best_hps)
        print(self.network.summary())
        self.network.fit(np.array(self.train_x), np.array(self.train_y).T, epochs = 30, validation_data = (np.array(self.test_x),np.array(self.test_y).T))

    def load_network_tuner(self):
        tuner = Hyperband(self.model_builder,
                          objective='val_accuracy',
                          max_epochs=5,
                          factor=3,
                          directory='./tunner/KingsDecision',
                          project_name='KingsDecision')

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.network = tuner.hypermodel.build(best_hps)

        
    def build_compile_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(32,)))

        hp_units1 = 480
        hp_units2 = 64
        hp_units3 = 128
        hp_units4 = 128
        model.add(keras.layers.Dense(units=hp_units1, activation='relu'))
        model.add(keras.layers.Dense(units=hp_units2, activation='relu'))
        model.add(keras.layers.Dense(units=hp_units3, activation='relu'))
        model.add(keras.layers.Dense(units=hp_units4, activation='relu'))
        model.add(keras.layers.Dense(16, activation="softmax"))

        model.compile(optimizer='sgd',
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model