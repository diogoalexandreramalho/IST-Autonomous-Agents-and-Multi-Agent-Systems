import chess.engine
import random
#import play
import sys
sys.path.append("../")
from board import SimpleBoard






autopilot = 0



class random_player():
    def move(self, board):
        move = random.choice(list(board.generate_legal_moves()))
        return move


class human_player():

    def move(self, board, suggested_move=None):

        def get_move(prompt):
            global autopilot
            uci = input(prompt)
            if uci and uci.startswith('t'):
                autopilot = int(uci[1:])

            if uci and uci[0] == 'q':
                raise KeyboardInterrupt()
            elif suggested_move and (autopilot or (uci and uci[0] == 's')):
                return suggested_move
            return uci

        if autopilot and suggested_move:
            autopilot -= 1
            return suggested_move

        uci = get_move("%s's move [q to quit, s-suggestion (%s), tN-autoplay]> " % (play.who(board.turn), suggested_move))
        legal_uci_moves = [move.uci() for move in board.legal_moves]
        while uci not in legal_uci_moves:
            print("Legal moves: " + (",".join(sorted(legal_uci_moves))))
            uci = get_move("%s's move[q to quit]> " % play.who(board.turn))
        return uci


class stockfish_player():
    def __init__(self,engine):
            #self.engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        self.engine = engine

    def move(self,board):
        result = self.engine.play(board, chess.engine.Limit(time=0.001), root_moves=board.generate_legal_moves())
        return result.move



class enhanced_human_player():
    def __init(self,engine):
        self.engine = engine
    def move(self, board):
        # note: use analyse() with multipv set to number of plays, to get a score to the board
        analysis = self.engine.analysis(board, chess.engine.Limit(time=0.1), root_moves=board.generate_legal_moves())
        return human_player(board, suggested_move=analysis.wait().move.uci())




