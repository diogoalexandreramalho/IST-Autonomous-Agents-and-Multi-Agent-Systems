import sys
sys.path.append("../")
import chess
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from Agent import Agent
from StateRepresentations import pieceList
import pickle
from kerastuner import Hyperband


class Rooks(Agent):
    '''
    The output is a kind of 14x2 sliding shifts,
    0-6 for the horizontal
    7-13 for the vertical
    + 14 for the other rook
    How:

     a b c d e f g h   R going from b to a: output 0
    | |R| | | | | | |                 to h: output 6


    ---------                  from 8 to 7: output 13
    8  R                                 1: output 7
    ---------
    7
    ---------
    6
    ---------
    5
    ---------
    4
    ---------
    3
    ---------
    2
    ---------
    1
    ---------
    '''
    def uciToY(self, uci,state):
        rookNum = -1
        r0, r1 = state[pieceList.listIndex['R'][0]], state[pieceList.listIndex['R'][1]]
        if r0 and chess.SQUARE_NAMES[r0-1] == uci[:2]:
            rookNum = 0
        elif r1 and chess.SQUARE_NAMES[r1-1] == uci[:2]:
            rookNum = 1
        else: #not for us
            pass

        # Now, calculate direction (assuming that uci is not trash)
        isVertical = uci[0] == uci[2]

        # Finally, shift
        from_square, to_square = chess.SQUARE_NAMES.index(uci[0:2]), chess.SQUARE_NAMES.index(uci[2:4])
        shift = chess.square_distance(from_square, to_square)
        if from_square > to_square:
            shift *= -1
        else:
            shift -= 1
        if isVertical:
            shift += from_square// 8
        else:
            shift += (from_square % 8)

        return 14*rookNum + 7*isVertical + shift


    def yToUci(self,y,state):
        rookNum = (y >= 14)
        refined_play = y % 14
        isVertical = (refined_play >= 7)
        shift = refined_play % 7


        from_square = state[pieceList.listIndex['R'][rookNum]] - 1
        if isVertical:
            to_square = shift
            to_square += (shift >= from_square//8)
            to_square = to_square*8 + from_square%8
        else:
            to_square = from_square - from_square%8 + shift
            to_square += (to_square >= from_square)


        return chess.SQUARE_NAMES[from_square] + chess.SQUARE_NAMES[to_square]


    def __init__(self,tuned=False,loadWeights=False):
        self.train_x = list()
        self.train_y = list()
        self.valid_x = list()
        self.valid_y = list()
        if tuned:
            self.load_network_tuner()
        else:
            self.build_compile_model()
        if loadWeights:
            self.loadWeights()

    def build_compile_model(self):  # The structure of the network
        model = Sequential()
        model.add(Input(32))
        model.add(Dense(2**6, activation="sigmoid"))
        model.add(Dense(28, activation="softmax"))
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam',metrics=['accuracy'])
        self.network = model
        return model

    def store(self, boardList,y): #putExperienceOnQueue
        self.train_x.append(boardList)
        self.train_y.append(y)

    def serializeStorage(self): #SerializeSamples
         with open('./Samples/RooksX', 'wb') as handleX, open('./Samples/RooksY', 'wb') as handleY:
            pickle.dump(self.train_x, handleX, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.train_y, handleY, protocol=pickle.HIGHEST_PROTOCOL)

    def loadStorage(self):
        with open('./Samples/RooksX', 'rb') as RooksX, open('./Samples/RooksY', 'rb') as RooksY:
            self.train_x.extend(pickle.load(RooksX))
            self.train_y.extend(pickle.load(RooksY))

    def serializeWeights(self, path="./checkpoints/Rooks/my_checkpoint"): #serialize Weights so we can load it after
        self.network.save_weights(path)

    def loadWeights(self, path="./checkpoints/Rooks/my_checkpoint"): #load serialized Weights
        self.network.load_weights(path)

    def predict(self,x):
        return self.network.predict(x.reshape(1,32))

    def model_builder(self, hp):
        model = Sequential()
        model.add(Input(32))
        neuron_choice = hp.Choice('activation', ["sigmoid", "relu"]) 
        for l in range(hp.Int('num_layers', 0, 4)):
            model.add(Dense(2**hp.Int('units_'+str(l), 4, 14), activation=neuron_choice))
        model.add(Dense(28, activation="softmax"))
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=hp.Choice('optimizer', ['adam', 'sgd']), metrics=['accuracy'])
        return model 

    def run_tuner(self):
        if not self.train_x:
            print("No samples, trying to read storage")
            self.loadStorage()
            print("Done")
        if not self.valid_x:
            print("Creating validation set")
            self.createValandTestSets()
            print("Done")
        t = Hyperband(self.model_builder,
                      objective='val_accuracy',
                      max_epochs=20,
                      directory='tunner',
                      project_name='Rooks')
        t.search(np.array(self.train_x), np.array(self.train_y).T, epochs=10, validation_data=(np.array(self.valid_x), np.array(self.valid_y).T))
        best_hps = t.get_best_hyperparameters(num_trials=1)[0]
        print(best_hps)
        self.network = t.hypermodel.build(best_hps)
        print(self.network.summary())
        self.train()

    def load_network_tuner(self):
        t = Hyperband(self.model_builder,
                      objective='val_accuracy',
                      max_epochs=5,
                      factor=3,
                      directory='tunner',
                      project_name='Rooks')

        best_hps = t.get_best_hyperparameters(num_trials=1)[0]
        print(best_hps.values)
        self.network = t.hypermodel.build(best_hps)

    def train(self):
        if not self.train_x:
            print("No samples, trying to read storage")
            self.loadStorage()
            print("Done")
        if not self.valid_x:
            print("Creating validation set")
            self.createValandTestSets()
            print("Done")
        earlystop = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1) 
        self.network.fit(np.array(self.train_x), np.array(self.train_y).T, epochs = 15, validation_data=(np.array(self.test_x), np.array(self.test_y).T), callbacks=[earlystop])

