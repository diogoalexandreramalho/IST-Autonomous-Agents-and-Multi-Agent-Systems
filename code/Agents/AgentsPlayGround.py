import sys
sys.path.append("../")
import chess
from board import SimpleBoard
from players import stockfish_player, random_player
from StateRepresentations.pieceList import numberToSquare, boardToList, listIndex, correctOrder, squareToNumber
from IPython.display import display, HTML, clear_output
import numpy as np
import random


#AGENTS
from Agent import Agent
from Horses import Horses
from Rooks import Rooks
from Queen import Queen
from King import King
from Pawns import Pawns
from Bishops import Bishops
from KingsDecision import KingsDecision


class AgentsPlayGround(Agent):

    def __init__(self):
        self.initializeAgents()

    '''
    Before every game which structures do we want to initialize?
    '''
    def initialize(self):
        self.board = SimpleBoard()
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        self.stockfish = stockfish_player(self.engine)
        self.randomPlayer = random_player()

    '''
    In the initialize, we instanciate all smart agents which control the pieces
    '''
    def initializeAgents(self):

        self.horses = Horses(False)
        self.rooks = Rooks(False)
        self.queen = Queen(False)
        self.king = King(False)
        self.pawns = Pawns(False)
        self.bishops = Bishops(False)
        self.Agents = [None,self.pawns,self.horses,self.bishops, self.rooks, self.queen, self.king]
            #This map ordering has to do with https://python-chess.readthedocs.io/en/latest/core.html#pieces
            #the None is because pawns number are 1... just to abstract and not have to subtract stuff...
        self.kingsDecision = KingsDecision(self.rooks,self.horses,self.bishops,self.queen,self.king,self.pawns)

    '''
    Returns the agent which controls the piece doing the move
    '''
    def mapFromMoveAgent(self,move): #Which piece is making the movement?
            square = chess.square(ord(move.uci()[0])-97, int(move.uci()[1])-1)
            piece = self.board.piece_at(square)
            return self.Agents[piece.piece_type]

    '''
    Returns the agent which controls the pieces that is going to die by the move
    '''
    def mapToMoveAgent(self, move): #Which agent has a piece that is going to die?
        square = chess.square(ord(move.uci()[2]) - 97, int(move.uci()[3]) - 1)
        piece = self.board.piece_at(square)
        return self.Agents[piece.piece_type]

    '''
    Generates a move from stockfish
    '''
    def generateStockFishMove(self):
        move = self.stockfish.move(self.board)
        pb = self.board
        aux=0
        while (move == None and aux<10):  # our pseudo chess bugs the stockfish once in a while
            print("stock fished blocked")
            self.board = SimpleBoard(self.board.fen())
            self.board.move_stack = pb.move_stack
            self.board._stack = pb._stack
            move = self.stockfish.move(self.board)
            aux = aux + 1


        return move

    '''
    Generates a random legal move
    '''
    def generateRandomMove(self):
        return self.randomPlayer.move(self.board)

    '''
    Plays the game, while storing data
    '''
    def play(self):
        self.initialize()
        while not self.board.is_game_over(claim_draw=True):

            if self.board.turn == chess.WHITE:  #StockFish
                move = self.generateStockFishMove()
                state = boardToList(self.board)
                agent = self.mapFromMoveAgent(move)
                if type(agent)!=type(""):#we don't have all agents,yet
                    y = agent.uciToY(move.uci(),state)
                    agent.store(state,y)

            else: #RandomPlayer
                move = self.generateRandomMove()

            self.board.push_uci(move.uci())
        self.engine.quit()

    '''
    Runs Play a numOfGames times and collects samples
    '''
    def collectSamples(self,numOfGames): #call play multiple times
        for i in range(numOfGames):
            if i%10==0: print("collecting("+str(i)+")")
            try: #Sometimes failures happen, it has to do with our simpleboard, but they're rare.
                self.play()
            except:
                print("game number " + str(i) + "failed to finish. No worries, we'll go to the next game!")

    '''
    Runs through every agent, performing their training
    This doesn't use hyperparametrization
    '''
    def train(self):
        self.kingsDecision.train()
        for agent in self.Agents:
            if(agent != None ):
                agent.train()

    '''
    Runs through every agent, performing storage of samples of each agent
    '''
    def serializeStorage(self):
        self.kingsDecision.serializeStorage()
        for agent in self.Agents:
            if(agent != None ):
                agent.serializeStorage()

    '''
    Runs through every agent, performing serialization of weights
    '''
    def serializeWeights(self):
        self.kingsDecision.serializeWeights()
        for agent in self.Agents:
            if(agent != None):
                agent.serializeWeights()

    '''
     Runs through every agent, performing load of weights
     '''
    def loadWeights(self):
        self.kingsDecision.loadWeights()
        for agent in self.Agents:
            if (agent != None):
                agent.loadWeights()

    '''
    Runs through every agent, performing load of tunner networks
    '''
    def load_network_tuner(self):
        self.kingsDecision.load_network_tuner()
        for agent in self.Agents:
            if(agent != None):
                agent.load_network_tuner()

    '''
    This is a Test Agent. Uses Stockfish to decide whether or not
    Agent plays or random plays. It's to test results.
    '''
    def oneAgentWStockfish(self,agent):
        state = boardToList(self.board)
        #STOCKFISH PART
        move = self.generateStockFishMove()
        if agent == self.mapFromMoveAgent(move):
            #print("YOUR AGENT IS DECIDING THE NEXT MOVE")
            playsY = agent.predict(state)[0]
            ys = playsY.argsort()[::-1]
            for y in ys:
                try:
                    uci = agent.yToUci(y, state)
                    self.board.push_uci(uci)

                    print("\t YOUR AGENT PLAYED " + uci + " and stockfish would have played " + move.uci())
                    if(move.uci() == uci):
                        self.agentCorrectDecisions = self.agentCorrectDecisions + 1.0
                    self.agentDecisions = self.agentDecisions + 1.0
                    return
                except:
                    self.agentIlegalPlays = self.agentIlegalPlays + 1.0
                    print("YOUR AGENT PLAYED AN ILEGAL MOVE '"+ uci + "' ("+move.uci()+") - it will try again" + str(self.agentIlegalPlays))
                    pass

        #print("RANDOM WAS PLAYED")
        move = self.generateRandomMove()
        aux = 0
        while(agent == self.mapFromMoveAgent(move) and aux<20):
            move=self.generateRandomMove()
            print("repeated")
            aux = aux + 1
        self.board.push_uci(move.uci())
        return

    '''
    Plays game with previous function
    '''
    def testOneAgentAgainstRandom(self,agent):
        i = 0
        self.initialize()
        #print("BOARD INICIAL")
        #print(self.board)
        self.agentDecisions = 0.0
        self.agentCorrectDecisions = 0.0
        self.agentIlegalPlays = 0.0
        while not self.board.is_game_over(claim_draw=True):
            print("\n\nITERAÇÃO="+str(i))
            print(len(self.board.move_stack))
            if i==200:
                #print("Não acabou antes das 200 rondas, YOUR AGENT SUCKS! Terminar ciclo")
                break
            if self.board.turn == chess.WHITE:  # StockFish
                self.oneAgentWStockfish(agent)

            else:  # RandomPlayer
                uci = self.generateRandomMove().uci()
                self.board.push_uci(uci)
            #print(self.board)

            #display(HTML(self.board._repr_svg_()))
            i = i + 1
        self.engine.quit()
        #print("In this game your played ",end='')
        #print((self.agentCorrectDecisions/self.agentDecisions)*100,end="% ")
        #print(self.agentCorrectDecisions)
        #print(self.agentDecisions)
        return (self.agentCorrectDecisions, self.agentDecisions, self.agentIlegalPlays)
        print("of the plays stockfish would have played!")

    '''
    This is a player with everyone of our team. Chooses one play
    '''
    def allAgentsWKingsDecision(self):
        self.totalAgentChoice += 1
        state = boardToList(self.board)

        # ordered choices for agents by kingDecision
        agentYs = self.kingsDecision.predict(state)[0].argsort()[::-1]


        stock_move = self.generateStockFishMove()
        stock_agent =self.mapFromMoveAgent(stock_move)

        
        for agentY in agentYs:
            agent = self.kingsDecision.yToAgent(agentY)
            print("The King nominated " + str(type(agent)) + " to play!")
            playsY = agent.predict(state)[0]
            ys = playsY.argsort()[::-1]
            for y in ys:
                try:
                    uci = agent.yToUci(y, state)
                    print("\t the " + str(type(agent)) + " decided to play " + uci)
                    print("stockfish: " + stock_move.uci())
                    self.board.push_uci(uci)
                    if agent == stock_agent:
                        self.correctAgentChoice += 1
                        if stock_move.uci() == uci:
                            self.correctChoiceByAgents += 1
                    return
                except:
                    print("\t\t the " + str(type(agent)) + " chose an invalid play " + uci)
                    pass

        move = self.generateRandomMove()
        self.board.push_uci(move.uci())
        return

    '''
    This is a Test MultiAgent. Joins our team together and decide with KingsDecision
    '''
    def testAllAgentsWKingsDecision(self):
        self.initialize()
        self.totalAgentChoice = 0
        self.correctAgentChoice = 0
        self.correctChoiceByAgents = 0
        i = 0
        while not self.board.is_game_over(claim_draw=True):
            print("\n\nITERAÇÃO="+str(i))
            if self.board.turn == chess.WHITE:  #StockFish
                self.allAgentsWKingsDecision()

            else: #RandomPlayer
                move = self.generateRandomMove()
                self.board.push_uci(move.uci())
            i += 1

        self.correctAgent = (self.correctAgentChoice/self.totalAgentChoice) * 100
        self.correctMoveWhenCorrectAgent = (self.correctChoiceByAgents/self.correctAgentChoice) * 100
        #print("\n\nKingsDecision chose the agents correctly {:.1f}%".format(self.correctAgent))
        #print("Agents chose the move correctly {:.1f}% when KingDecision chose the Agent correctly".format(self.correctMoveWhenCorrectAgent))
        self.engine.quit()

    '''
    This is a Player Test MultiAgent. Uses Stockfish to decide which
    Agent plays or random plays. It's to test results.
    '''
    def allAgentsWStockFish(self):
        state = boardToList(self.board)
        #STOCKFISH PART
        move = self.generateStockFishMove()
        agent =self.mapFromMoveAgent(move)
        playsY = agent.predict(state)[0]
        ys = playsY.argsort()[::-1]
        for y in ys:
            try:
                print("evaluating y=" + str(y))
                uci = agent.yToUci(y, state)
                self.board.push_uci(uci)
                print("played " + uci)
                return
            except:
                pass

    '''
    Play game with Previous function/player
    '''
    def testAllAgentsAgainstRandom(self):
        i = 0
        self.initialize()
        while not self.board.is_game_over(claim_draw=True):
            print("ITER,AÇÃO=" + str(i))
            print(len(self.board.move_stack))
            print(self.board)
            if i==200:
                print("Não acabou antes das 200 rondas. Terminar ciclo")
                break
            if self.board.turn == chess.WHITE:  # StockFish
                self.allAgentsWStockFish()

            else:  # RandomPlayer
                uci = self.generateRandomMove().uci()
                self.board.push_uci(uci)
            i = i + 1
        self.engine.quit()


    def testKingsDecision(self):
        self.initialize()
        i = 0
        while not self.board.is_game_over(claim_draw=True):
            print("ITERAÇÃO=" + str(i))
            if self.board.turn == chess.WHITE:  #StockFish
                move = self.generateStockFishMove()
                state = boardToList(self.board)
                agent = self.mapFromMoveAgent(move)
                if type(agent)!=type(""):#we don't have all agents,yet
                    decidedAgentY = self.kingsDecision.predict(state).argmax()
                    decidedAgent = self.kingsDecision.yToAgent(decidedAgentY)
                    if agent == decidedAgent:
                        print("correct decision it chose " + str(type(decidedAgent)))
                    else:
                        print("wrong decision it chose " + str(type(decidedAgent)) + " but should have been " + str(type(agent)))

            else: #RandomPlayer
                move = self.generateRandomMove()

            self.board.push_uci(move.uci())
            i = i + 1
        self.engine.quit()






'''
jeffrey = AgentsPlayGround()
jeffrey.load_network_tuner()
#jeffrey.loadStorage()
jeffrey.loadWeights()



print("king")
jeffrey.king.network.summary()

print("horses")
jeffrey.horses.network.summary()

print("bishops")
jeffrey.bishops.network.summary()


print("queen")
jeffrey.queen.network.summary()

print("pawns")
jeffrey.pawns.network.summary()

print("rooks")
jeffrey.rooks.network.summary()


for agent in jeffrey.Agents:
    if agent !=None:
        x = 0
        print(type(agent))
        agent.loadStorage()
        for i in range(100):
            if agent.predict(agent.train_x[i]).argmax() == agent.train_y[i]:
                x = x + 1
        print(x / 100)
'''

