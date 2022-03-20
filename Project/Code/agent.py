import chess

""" To be done: way of detecting the other captured me """
class GenericPiece:
    def __init__(self, start_position, board, engine):
        self.position = start_position #chess.Square
        self.board = board


    """ Should be more tied to internal, specific move representation """
    def generate_posible_moves(self):
        yield from self.board.generate_legal_moves_from(self.position)
        yield chess.Move.null()

    """ My learnt evaluation """
    def evaluate_moves(self, moves):
        pass

    """ RL with teacher
        Possible problem: this will need to somehow account for the null play.
        What is the value of the null play? The best play of _another_ Piece?

        Will need something like:                           (this is alternatives)
        engine.analyse(self.board, chess.engine.Limit(...), multipv= ?, root_moves=self.generate_possible_moves())
        The problem is that it just ignores the null move, and the analysis comes out of order
        (besides the atual score being not that deterministic to begin with, depends on strict limits)
    """
    def learn_from_teacher(self, teacher_evaluation):
        pass

   """
   Probably should call the teacher, and record the choice somewhere
   """
    def apply_move(self, move):
        if move.from_square == self.position:
            self.position = move.to_square
            self.board.push(move)

    """ Save what was learnt """
    def dump_state(self):
        pass

    """ Restore what was learnt """
    def restore_state(self, state):
        pass
