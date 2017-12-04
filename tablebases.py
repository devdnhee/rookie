import chess
import numpy as np
from chess_utils import map_side_to_int
import chess.gaviota as gav

TB=gav.open_tablebases('Gaviota')

def probe_result(epd):
    '''
    return 1 if white wins, 0 if draw, -1 if black wins
    returns None if not in tablebases
    '''
    if sum([int(c.isalpha()) for c in epd])<=(5+1):
        board=chess.Board.from_epd(epd)[0]
        turn=board.turn
        return map_side_to_int(turn)*TB.probe_wdl(board)
    else:
        return None
