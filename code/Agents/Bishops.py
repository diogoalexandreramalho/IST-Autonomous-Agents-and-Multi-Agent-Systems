import sys
sys.path.append("../")
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Dense, Embedding, Reshape, Flatten
from keras.optimizers import SGD
import chess
from board import SimpleBoard
import chess.engine
from players import stockfish_player, random_player
from StateRepresentations import pieceList 
import pickle
#Tuner
import kerastuner
from kerastuner import Hyperband
import tensorflow as tf
from tensorflow import keras

import IPython

from Agent import Agent
import random





class Bishops(Agent):
	'''
	Direction - NE          Direction - NW   

	  a b c d e f g h       a   b  c  d e f g h  
	8| | | | | | | |6|    8|13|  |  |  | | | | |    SE - 14 to 20
	7| | | | | | |5| |    7|  |12|  |  | | | | |    SW - 21 to 27
	6| | | | | |4| | |    6|  |  |11|  | | | | |    And then, the nest 28 actions are for Bishop 2, same logic applies
	5| | | | |3| | | |    5|  |  |  |10| | | | |
	4| | | |2| | | | |    4|  |  |  |  |9| | | |
	3| | |1| | | | | |    3|  |  |  |  | |8| | |
	2| |0| | | | | | |    2|  |  |  |  | | |7| |
	1|B| | | | | | | |    1|  |  |  |  | | | |B|
	'''

	def uciToY(self,uci,state):
		# select the bishop
		bishopNum = -1
		b0, b1 = state[pieceList.listIndex['B'][0]], state[pieceList.listIndex['B'][1]]
		if b0 and chess.SQUARE_NAMES[b0-1] == uci[:2]:
			bishopNum = 0
		elif b1 and chess.SQUARE_NAMES[b1-1] == uci[:2]:
			bishopNum = 1
		else: 
			print("No bishop in movePlay?")

		# choose the direction
		val = -1
		if(uci[0] < uci[2] and uci[1] < uci[3]): #NE
			val = 0
		elif(uci[0] > uci[2] and uci[1] < uci[3]): #NW
			val = 1
		elif(uci[0] < uci[2] and uci[1] > uci[3]): #SE
			val = 2
		elif(uci[0] > uci[2] and uci[1] > uci[3]): #SW
			val = 3

		from_square, to_square = chess.SQUARE_NAMES.index(uci[0:2]), chess.SQUARE_NAMES.index(uci[2:4])
		dist = chess.square_distance(from_square, to_square)
		return 28*bishopNum + 7*val + dist - 1 # 0 a 55
		

	def yToUci(self,y,state):
		bishopNum = (y >= 28)
		from_square = chess.SQUARE_NAMES[state[pieceList.listIndex['B'][bishopNum]] - 1]

		dist = (y+1)%7
		val = (y // 7) - 4*bishopNum

		if(val == 0): #NE
			to_square = chr(ord(from_square[0]) + dist) + str(int(from_square[1]) + dist)
		elif(val == 1): #NW
			to_square = chr(ord(from_square[0]) - dist) + str(int(from_square[1]) + dist)
		elif(val == 2): #SE
			to_square = chr(ord(from_square[0]) + dist) + str(int(from_square[1]) - dist)
		elif(val == 3): #SW
			to_square = chr(ord(from_square[0]) - dist) + str(int(from_square[1]) - dist)

		return from_square + to_square


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
		model.add(Dense(1056, activation='relu', input_shape=(32,)))
		model.add(Dense(1056, activation='relu'))
		model.add(Dense(56, activation="softmax"))
		model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='sgd',metrics=['accuracy'])
		return model

	def store(self, boardList,y): #putExperienceOnQueue
		self.train_x.append(boardList)
		self.train_y.append(y)

	def serializeStorage(self): #SerializeSamples
		with open('./Samples/BishopsX', 'wb') as handle:
			pickle.dump(self.train_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
		with open('./Samples/BishopsY', 'wb') as handle:
			pickle.dump(self.train_y, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def loadStorage(self):
		BishopsX = open('./Samples/BishopsX', 'rb')
		self.train_x.extend(pickle.load(BishopsX))
		BishopsY = open('./Samples/BishopsY', 'rb')
		self.train_y.extend(pickle.load(BishopsY))
		BishopsX.close()
		BishopsY.close()

	def serializeWeights(self, path="./checkpoints/Bishops/my_checkpoint"): #serialize Weights so we can load it after
		self.network.save_weights(path)

	def loadWeights(self, path="./checkpoints/Bishops/my_checkpoint"): #load serialized Weights
		self.network.load_weights(path)

	def predict(self,board):#example x=np.array([self.train_x[0]])
		return self.network.predict(np.array([board]))

	def train(self):
		self.network.fit(np.array(self.train_x), np.array(self.train_y).T, batch_size=100, epochs=30, validation_split=0.2)

	def model_builder(self,hp):
		model = keras.Sequential()

		model.add(Flatten(input_shape=(32,))) #input

		# adds hidden layers
		for i in range(hp.Int('layers', 1, 3)):
			model.add(Dense(
				units=hp.Int('units_' + str(i), 32, 2592, step=128),
				activation='relu'))
		
		model.add(Dense(56, activation='softmax')) #output

		# Tune the learning rate for the optimizer
		# Choose an optimal value from 0.01, 0.001, or 0.0001
		hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
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
						  project_name='Bishops')

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
						  project_name='Bishops')

		best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
		self.network = tuner.hypermodel.build(best_hps)
		
