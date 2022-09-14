import sys
sys.path.append("../")
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, Flatten
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

class King(Agent):
	'''
		  [5] [0] [4]
		  [3]  K  [2]
		  [7] [1] [6]
	'''

	def uciToY(self,uci,state):
		if(chess.SQUARE_NAMES[state[4]-1] != uci[0:2]):
			print("No King in movePlay?")
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
		return val # 0 a 7


	def yToUci(self,y,state):
		from_square = chess.SQUARE_NAMES[state[4]-1]
		dist = 1
		val = y
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
		self.train_x = list()
		self.train_y = list()
		self.valid_x = list()
		self.valid_y = list()
		self.network= self.build_compile_model()
		if(loadWeights==True):
			self.loadWeights()

	def build_compile_model(self):  # The structure of the network
		model = Sequential()
		model.add(Dense(110, activation='relu', input_shape=(32,)))
		model.add(Dense(8, activation="softmax"))
		model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='sgd',metrics=['accuracy'])
		return model

	def store(self, boardList,y): #putExperienceOnQueue
		#print("King " + str(len(self.train_x)) +" "+ str(y))
		self.train_x.append(boardList)
		self.train_y.append(y)

	def serializeStorage(self): #SerializeSamples
		with open('./Samples/KingX', 'wb') as handle:
			pickle.dump(self.train_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
		with open('./Samples/KingY', 'wb') as handle:
			pickle.dump(self.train_y, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def loadStorage(self):
		KingX = open('./Samples/KingX', 'rb')
		self.train_x.extend(pickle.load(KingX))
		KingY = open('./Samples/KingY', 'rb')
		self.train_y.extend(pickle.load(KingY))
		KingX.close()
		KingY.close()

	def serializeWeights(self, path="./checkpoints/King/my_checkpoint"): #serialize Weights so we can load it after
		self.network.save_weights(path)

	def loadWeights(self, path="./checkpoints/King/my_checkpoint"): #load serialized Weights
		self.network.load_weights(path)

	def predict(self,board):#example x=np.array([self.train_x[0]])
		return self.network.predict(np.array([board]))

	def train(self):
		self.network.fit(np.array(self.train_x), np.array(self.train_y).T, batch_size=100, epochs=30, validation_split=0.3)

	def model_builder(self,hp):
		model = keras.Sequential()

		model.add(Flatten(input_shape=(32,))) #input

		# adds hidden layers
		for i in range(hp.Int('layers', 0, 4)):
			model.add(Dense(
				units=hp.Int('units_' + str(i), 10, 400, step=20),
				activation='relu'))
		
		model.add(Dense(8, activation='softmax')) #output

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
						  project_name='King')

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
						  project_name='King')

		best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
		self.network = tuner.hypermodel.build(best_hps)
		

