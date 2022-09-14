import sys
sys.path.append("../")
import chess
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from Agent import Agent
import pickle
from kerastuner import Hyperband


class Pawns(Agent):
    '''
     The output is simple: 8*3 possibilities (0-7), where the number indicates which pawn moves forward 1 step
     (remember, we eliminated double move pawns)
     
     the question is how to enumerate pawns. simple: by the order they appear on state:
     from position 8 to 15
     
     
     
      1  0  2|4  3  5  ...
         P0  |   P1
    '''
    
    def uciToY(self, uci, state):
        try:
            pos = np.where(state == (chess.SQUARE_NAMES.index(uci[:2]) + 1))[0][0]
            if not 7 < pos < 16:
                raise ValueError("Not a pawn")
            if uci[0] == uci[2]:
                return (pos-8)*3
            else:
                return (pos-8)*3+1+(uci[2] > uci[0])
        except IndexError:
            pass
            
    def yToUci(self,y,state):
        from_square = state[y//3+8] - 1
        to_square = from_square + 8 # they always advance 8 at least
        if y%3 == 1:
            to_square -= 1
        elif y%3 == 2:
            to_square += 1
        
        return chess.SQUARE_NAMES[min(from_square, 55)] + chess.SQUARE_NAMES[min(to_square, 63)]

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
        model.add(Dense(2**10, activation="sigmoid"))
        model.add(Dense(2**6, activation="sigmoid"))
        model.add(Dense(24, activation="softmax", name="output"))
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam',metrics=['accuracy'])
        self.network = model
        return model

    def store(self, boardList,y): #putExperienceOnQueue
        self.train_x.append(boardList)
        self.train_y.append(y)

    def serializeStorage(self): #SerializeSamples
        with open('./Samples/PawnsX', 'wb') as handleX, open('./Samples/PawnsY', 'wb') as handleY:
            pickle.dump(self.train_x, handleX, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.train_y, handleY, protocol=pickle.HIGHEST_PROTOCOL)

    def loadStorage(self):
        with open('./Samples/PawnsX', 'rb') as PawnsX, open('./Samples/PawnsY', 'rb') as PawnsY:
            self.train_x.extend(pickle.load(PawnsX))
            self.train_y.extend(pickle.load(PawnsY))

    def serializeWeights(self, path="./checkpoints/Pawns/my_checkpoint"): #serialize Weights so we can load it after
        self.network.save_weights(path)

    def loadWeights(self, path="./checkpoints/Pawns/my_checkpoint"): #load serialized Weights
        self.network.load_weights(path)

    def predict(self,x):
        return self.network.predict(x.reshape(1,32))

    def model_builder(self, hp):
        model = Sequential()
        model.add(Input(32))
        neuron_choice = hp.Choice('activation', ["sigmoid", "relu"]) 
        for l in range(hp.Int('num_layers', 0, 3)):
            model.add(Dense(2**hp.Int('units_'+str(l), 4, hp.values['units_'+str(l-1)] if l else 14, step=2),
                            activation=neuron_choice))
        model.add(Dense(24, activation="softmax"))
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
                      project_name='Pawns')
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
                      directory='./tunner',
                      project_name='Pawns')

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
