
from abc import ABC, abstractmethod


class Agent:

    @abstractmethod
    def uciToY(uci,state):
        '''
        INPUT:
        uci is a uci description of a play (check the python-chess docs).
        state is a piecelist, implemented by us in the folder StateRepresentations

        OUTPUT:
        number corresponding to the neural network classification (a number)
        '''
        pass

    @abstractmethod
    def yToUci(self,y,state):
       '''
       The Oposite of playToOutput function.
       INPUT:
       y is the classification of the neural network
       state is the piecelist which represent the board

       OUTPUT:
       The uci play correspondent.
       '''
       pass


    @abstractmethod
    def build_compile_model(self):  # The structure of the network
        '''
        Build neural network
        '''
        pass

    @abstractmethod
    def store(self, state,y): #putExperienceOnQueue
        '''
        store states and it's correct outputs.
        '''
        pass

    @abstractmethod
    def serializeStorage(self): #SerializeSamples
        '''
        serialize storage of states and outputs.
        We don't want to keep generating states.
        '''
        pass

    @abstractmethod
    def loadStorage(sel):
        '''
        loads serialized storage
        '''
        pass

    def createValandTestSets(self,vPercentage=0.1,tPercentage=0.1):
        '''
        Separates the self.train_x into a new train_x, validation_x, test_x
        Don't use train_split methods - Maintains order, since we're simulating "games"
        '''
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

    @abstractmethod
    def serializeWeights(self, path): #serialize Weights so we can load it after
        '''
        serialize neural network weights
        '''
        pass

    @abstractmethod
    def loadWeights(self, path): #load serialized Weights
        '''
        Load serialized neural networks weights
        '''
        pass

    @abstractmethod
    def predict(self,x):#example x=np.array([self.train_x[0]])
        '''
        Input x in the NN and receives label y (Don't transform this into uci, y is the raw output of NN)
        '''
        pass

    @abstractmethod
    def train(self):
        '''
        All the storage of states and y's is used here to train the NN
        '''
        pass

    @abstractmethod
    def model_builder(self,hp):
        '''
        constructs the model to vary in hyperparametrization
        '''
        pass

    @abstractmethod
    def run_tuner(self):
        '''
        runs the hyperparametrization
        Only run this if your data is separated into train, validity and test sets.
        '''
        pass

    @abstractmethod
    def load_network_tuner(self):
        '''
        loads the hyperparametrized network.
        THIS DOESN'T LOAD THE WEIGHTS. it only puts self.network to a network with the same
        parameters as the one that was decided as the best.
        So, after you run run_tuner, you might want to call serializeWeights and, if you want to
        restart where you left, you then call load_netowork_tuner and then load_weights.
        '''
        pass


    
