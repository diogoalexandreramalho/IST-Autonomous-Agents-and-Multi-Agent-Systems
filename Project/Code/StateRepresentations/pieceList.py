import chess
import numpy as np

#If you touch this you'll break the Agents code.
listIndex ={
    'R':(0,7),
    'N':(1,6),
    'B':(2,5),
    'Q':(3,-1),
    'K':(4,-1),
    'r':(16,23),
    'n':(17,22),
    'b':(18,21),
    'q':(19,-1),
    'k':(20,-1),
}


def twoPieces(pieces,x,pos):
    if pieces[x[0]]==0:
        pieces[x[0]]=pos
    else:
        if(x[1]==-1):
            print("BoardToState problem in twopieces")
            print(pieces)
            exit(-1)
        pieces[x[1]]=pos


def pawn(pieces,x,pos):
    pieces[x]=pos

'''
lista com 32 peças em que cada posição tem a sua posição no tabuleiro de 1-64 (inclusive)
se alguma posição estiver a 0, então a peça está morta
[ R1 N1 B1 Q K B2 N2 R2 P P P P P P P P r1 n1 b1 q k b2 n2 r2 p p p p p p p p]
(capitalized pieces are white)

The first state is 
[ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 57. 58.
 59. 60. 61. 62. 63. 64. 49. 50. 51. 52. 53. 54. 55. 56.]
 ATTENTION: chess.SQUARE_NAMES start at 0, so chess.SQUARE_NAMES[0] = 'a1'

input is a class board
or a Fen description of a board, in which case isFen should be true
'''
def boardToList(board, isFen = False):
    pieces = np.zeros((32,),dtype=np.int8)
    line = 7
    index = 0
    blackCount = 24
    whiteCount = 8
    if(isFen == False):
        board = board.board_fen()
    for c in str(board):
        if c=='/':
            line = line - 1
            index = 0
        elif c == ' ' or c == '.' or c == '\n':
            continue
        else:
            if(c.isnumeric()):
                index = index + int(c)
            else:
                if(c=='p'):
                    pawn(pieces,blackCount,line*8+index+1)
                    blackCount = blackCount + 1
                elif(c=='P'):
                    pawn(pieces,whiteCount,line*8+index+1)
                    whiteCount = whiteCount + 1
                else:
                    twoPieces(pieces,listIndex[c],line*8+index+1)
                index = index + 1
    return pieces



'''
#EXAMPLE:
board = chess.Board()
print(boardToList(board))
board = chess.Board()
print(boardToList(board.board_fen(),isFen=True))
'''

def numberToSquare(number):
    '''
    Gets a number like 64 and outputs h8
    também podemos aceder a chess.SQUARE_NAMES
    '''
    number = number - 1 #the squares start at 1, but we need to do operations with them starting at 0
    lineNum= str(number//8+1)
    letter = str(chr(number%8+97))
    return letter + lineNum


def squareToNumber(square):
    return (ord(square[0])-96)*square[1]

def isOrderInListCorrect(square1,square2):
    '''
    If I have to knights, one in a and another in b, who comes first in the state list?
    returns true if square1 is before square2
    Arguments are string positions
    Don't compare equal squares.
    '''
    if square1[1] > square2[1]:
        return True
    elif square1[1]<square2[1]:
        return False
    else:
        if(square1[0] < square2[0]):
            return True
        elif(square1[0]>square2[0]):
            return False
    exit(-1)

def correctOrder(squares):
    '''
    It's a follow up of the previous function
    but corrects the order in the list of two squares
    '''
    if not isOrderInListCorrect(squares[0],squares[1]):
        return [squares[1],squares[0]]
    return squares
