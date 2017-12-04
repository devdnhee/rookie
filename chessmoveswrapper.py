"""Module wrapping the chessmoves engine.

The chessmoves engine provides c-bindings for move generation. This module
enhances it's functionality by checking for repetitions and the fifty-move rule.
It requires chessmoves to have an 'enhanced' api, which also returns some flags
along with every possible move.
"""

# TODO: add rules for (not) claiming draws

# Import the 'enhanced' version of the chessmoves library.
# This library also computes a set of flags for each possible move,
# indicating whether the resulting position is mate, stalemate, check or not,
# and whether the move was a pawn move or capture or not.
import mychessmoves as chessmoves
import chess


def _is_insufficient_material(fen):
    """Computes wheter a given position is a draw due to insufficient material.
    """
    board_fen = fen.split()[0]

    # A board position is a draw due to insufficient material in the following
    # cases:
    # - king versus king
    # - king versus knight and king
    # - king and bisschop(s) versus king and bisschop(s), where all bisschops
    #       stand on the same square color.

    # To check for these conditions, we scan over the FEN string, keeping
    # track of the current square, whether we encountered a knight, and
    # whether we encountered a bisschop, and if so, which square color it's on.
    row, col = 0, 0  # Keeps track of the square
    # Bisschop presence and corresponding square color are encoded by the
    # following variable, having the value -1 if no bisschops have been
    # encountered, 0 if only black square bisschops have been encountered,
    # or 1 if only white square bisschops have been encountered.
    bisschop_square = -1
    knight = False  # Keeps track of the presence of a knight

    for c in board_fen:  # Scan over the FEN string
        # If any pawns, rooks or queens are present, mate is still possible.
        if c in ('p', 'P', 'r', 'R', 'q', 'Q'):
            return False
        # Kings are always present, we just need to keep track of the position
        elif c in ('k', 'K'):
            col += 1
        # If we encounter a knight fot the first time, we keep track of this.
        # If we already encountered a knight or a bisschop, mate is still possible.
        elif c in ('n', 'N'):
            if knight or bisschop_square >= 0:
                return False
            else:
                knight = True
                col +=1
        # If we encounter a bisschop, we make sure it is on the same color square
        # as previous bisschops, and that we have not encountered any knights.
        elif c in ('b', 'B'):
            sq = (row+col) % 2

            if bisschop_square == -1:
                bisschop_square = sq
            if (sq != bisschop_square) or knight:
                return False
            col += 1
        # Other symbols only require us to keep track of our position.
        elif c == '/':  # new row
            row += 1
            col = 0
        else:  # advance c columns
            col+= int(c)

    return True


class ChessMovesEngine():
    """Implements move generation in C using a modified version of Chessmoves."""

    WHITE = 'w'
    BLACK = 'b'
    WHITE_WIN = '1-0'
    DRAW = '1/2-1/2'
    BLACK_WIN='0-1'
    UNDECIDED='*'

    def __init__(self, epd):
        if epd is None:
            # Fen describing the board state, excluding the half- and fullmove counters
            self.board = chessmoves.startPosition[:-4]
        else:
            self.board=epd
        # Halfmove clock
        self.fifty_move_counter = 100
        # Fullmove clock
        self.ply = 0
        # Dict storing previous positions
        self.repetition_map = {self.board: 1}

        self.flags = '0-'  # some flags describing the position and the last move
        self.moves = None  # current possible moves
        self.result = '*'
        
        # Additional code -> check if we have to adapt result (if epd was a
        # mate or draw position) with ugly code (using other chess module) but
        # too lazy to do it neatly (should check c-code for it)
        """
        if len(self.get_moves())==0:
            b=chess.Board.from_epd(epd)[0]
            if b.is_checkmate():
                if self.turn()=='w':
                    self.result='0-1'
                else:
                    self.result='1-0'
            else:
                self.result='1/2-1/2'
        """

    def __str__(self):
        return '<ChessMovesEngine %s>' % self.get_fen()

    def turn(self):
        return self.board.split()[1][0]

    def change_turn(self):
        s=list(self.board)
        s[-5]='w' if s[-5]=='b' else 'b'
        self.board="".join(s)

    def _compute_moves(self):
        """Compute all possible moves, and the resulting board states.
        """
        turn = self.turn()
        # Do the move generation in C
        all_moves = chessmoves.moves(self.board, 'myuci')
        #print all_moves
        # For each move, determine the result
        self.moves = {}
        for move, next_board in all_moves.items():
            uci_notation = move[:-2]
            move_flags = move[-2:]

            # Check for mate
            if move_flags[1] == '#':
                result = '1-0' if turn == self.WHITE else '0-1'
            # Check for stalemate
            elif move_flags[1] == '=':
                result = '1/2-1/2'
            # Check for 50 move rule
            elif move_flags[0] == '0' and self.fifty_move_counter == 1:
                result = '1/2-1/2'
            # Check for repetitions
            elif self.repetition_map.get(next_board, 0) == 2:
                result = '1/2-1/2'
            # Check for insufficient material
            elif _is_insufficient_material(next_board):
                result = '1/2-1/2'
            else:
                result = '*'

            self.moves[uci_notation] = (next_board, move_flags, result)

    def get_moves(self):
        # Generate the moves for the current position in C
        # The result is a dict mapping each move (in uci notation) to a tuple
        # containing a pair of flags (indicating a pawn move or capture, and
        # a mate, draw or check, respectively) and the next board position.
        if self.moves is None:
            self._compute_moves()
        return self.moves.keys()

    def get_moves_and_states(self):
        if self.moves is None:
            self._compute_moves()
        return self.moves

    def is_move_terminal(self, move):
        next_pos, flags, result = self.moves[move]
        return result != '*'

    def is_game_over(self):
        return self.result != '*'

    def get_result(self):
        return self.result

    def get_move_result(self, move):
        return self.moves[move][2]

    def get_fen(self):
        halfmove_clock = 100 - self.fifty_move_counter
        fullmove_number = 1 + (self.ply // 2)
        return self.board + ' %d %d' % (halfmove_clock, fullmove_number)

    def get_move_fen(self, move):
        next_pos, flags, result = self.moves[move]
        next_fifty_move_counter = (self.fifty_move_counter-1) if flags[0] == '0' else 100
        next_halfmove_clock = 100 - next_fifty_move_counter
        next_ply = self.ply + 1
        next_fullmove_number = 1 + (next_ply // 2)
        return next_pos + ' %d %d' % (next_halfmove_clock, next_fullmove_number)

    def do_move(self, move):
        # Perform the move:
        # - Change the current board state.
        # - Change correspondig flags
        # - Change repetition and move counters
        assert self.result == '*'

        self.board, self.flags, self.result = self.moves[move]
        self.moves = None

        assert self.flags[0] in ('0', '1')
        if self.flags[0] == '1':    # Reset 50 move counter
            self.fifty_move_counter = 101
        self.fifty_move_counter -= 1
        self.ply += 1

        self.repetition_map[self.board] = self.repetition_map.get(self.board, 0) + 1

        terminal = (
                self.flags[1] in ('#', '=')
                or self.fifty_move_counter == 0
                or self.repetition_map[self.board] == 3
                or _is_insufficient_material(self.board))
        assert terminal == (self.result != '*')


def init_board():
    return ChessMovesEngine()

def test():
    fen='7k/3R4/4K3/8/8/8/8/8 w - - 0 1'
    epd='7k/3R4/4K3/8/8/8/8/8 w - -'
    engine=ChessMovesEngine(epd)
    print 'FEN notation board: {}'.format(engine.get_fen())
    assert fen==engine.get_fen()
    engine._compute_moves()
    print engine.moves
    ucis=['e6f5','h8g8','d7d5','g8f8','f5g6','f8g8','d5d8']
    for u in ucis:
        engine._compute_moves()
        print engine.get_move_result(u)
        engine.do_move(u)
    print chess.Board(engine.get_fen()) 
    assert engine.get_result()==ChessMovesEngine.WHITE_WIN

    fen='7k/3R4/4K3/8/8/8/8/8 w - - 0 1'
    epd='7k/3R4/4K3/8/8/8/8/8 w - -'
    engine=ChessMovesEngine(epd)
    print 'FEN notation board: {}'.format(engine.get_fen())
    assert fen==engine.get_fen()
    print engine.turn()
    engine._compute_moves()
    print engine.moves
    engine.change_turn()
    engine._compute_moves()
    print engine.moves

if __name__=='__main__':
    test()
