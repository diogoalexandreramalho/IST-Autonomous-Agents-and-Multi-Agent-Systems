import chess
from typing import Optional, Iterator
from chess import Move




class SimpleBoard(chess.Board):
    Bitboard = int
    BB_ALL = 0xffff_ffff_ffff_ffff
    aliases = ["SimpleBoard"]
    uci_variant = "chess"  # Unofficial
    xboard_variant = "chess" # Unofficial

    def __init__(self, fen: Optional[str] = chess.STARTING_FEN, chess960: bool = False) -> None:
        super().__init__(fen, chess960=chess960)
        self.castling_rights = chess.BB_EMPTY

    def twoPawnStep(self, move):  # Which piece is making the movement?
        #return False
        squareFrom = chess.square(ord(move.uci()[0]) - 97, int(move.uci()[1]) - 1)
        return self.piece_at(squareFrom).piece_type == chess.PAWN \
               and ord(move.uci()[0]) == ord(move.uci()[2]) \
               and abs(int(move.uci()[1]) - int(move.uci()[3])) == 2


    def generate_pseudo_legal_moves(self, from_mask: chess.Bitboard = chess.BB_ALL, to_mask: chess.Bitboard = chess.BB_ALL) -> Iterator[chess.Move]:
        yield from \
          filter(lambda m: m.promotion==None and not self.is_en_passant(m) and not self.twoPawnStep(m),
            map(lambda m: m if m.promotion != chess.QUEEN else \
                     chess.Move(m.from_square, m.to_square),
              super().generate_pseudo_legal_moves(from_mask, to_mask)))

    def generate_legal_moves_from(self, from_square: chess.Square) -> Iterator[chess.Move]:
        yield from self.generate_legal_moves(from_mask=chess.BB_SQUARES[from_square])


    def generate_legal_moves(self, from_mask: Bitboard = BB_ALL, to_mask: Bitboard = BB_ALL) -> Iterator[Move]:
        yield from \
              filter(lambda m: m.promotion==None and not self.is_en_passant(m) and not self.twoPawnStep(m),
                map(lambda m: m if m.promotion != chess.QUEEN else \
                         chess.Move(m.from_square, m.to_square),
                  super().generate_legal_moves(from_mask, to_mask)))


