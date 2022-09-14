import chess
from bitarray import bitarray
import numpy as np





#for bitmap
bitMapIndex ={
    'r':0,
    'n':1,
    'b':2,
    'q':3,
    'k':4,
    'p':5,
    'R':6,
    'N':7,
    'B':8,
    'Q':9,
    'K':10,
    'P':11,
}

'''
HERE WE DEFINE SOME FUNCTIONS WHICH TRANSFORM A BOARD INTO
A STATE REPRESENTATION

'''


'''
THIS USES BITMAPS. IT'S QUITE EFFICIENT NOT ONLY THE OPERATIONS,
BUT THE TRANSFORMATION, SINCE WE USE THE FEN'S DESCRIPTION OF A BOARD
REF:https://www.chess.com/blog/the_real_greco/representations-of-chess-fen-pgn-and-bitboards
'''
def boardToBitmap(board):


    pieces = np.zeros((12,64))
    pos = 0
    for c in str(board.board_fen()):
        if c == ' ' or c == '.' or c == '\n' or c=='/':
            continue
        else:
            if(c.isnumeric()):
                pos = pos + int(c)
            else:
                pieces[bitMapIndex[c]][pos] = 1
                pos = pos + 1
    return pieces

'''
#EXAMPLE:
board = chess.Board()
pieces = boardToBitmap(board)
#for i in range(12):
#    print(pieces[i])
print(pieces[index['p']])
'''
