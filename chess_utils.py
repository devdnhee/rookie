"""
This module contains some chess related functions to extract information about
the environment (the chessboard) for the agent and other utils absent in the
chess package.
"""

from __future__ import absolute_import
import chess
import random

piece_configurations={
    'KRk':[chess.Piece(chess.KING,chess.WHITE),
          chess.Piece(chess.ROOK,chess.WHITE),
          chess.Piece(chess.KING,chess.BLACK)],
    'Kkr':[chess.Piece(chess.KING,chess.WHITE),
          chess.Piece(chess.ROOK,chess.BLACK),
          chess.Piece(chess.KING,chess.BLACK)],
    'Kk':[chess.Piece(chess.KING,chess.WHITE),
          chess.Piece(chess.KING,chess.BLACK)]
}

def uci_to_squares(u):
    c1=ord(u[0])-ord('a')
    r1=ord('8')-ord(u[1])
    c2=ord(u[2])-ord('a')
    r2=ord('8')-ord(u[3])
    return ((r1,c1),(r2,c2))

def get_piece(state,(r1,c1)):
    line=(state.split()[0]).split('/')[r1]
    f=0
    for j in xrange(len(line)):
        if ord('0')<=ord(line[j])<=ord('9'):
            f+=int(line[j])
        else:
            if f==c1:
                return line[j]
            elif f>c1:
                return None 
            else:
                f+=env.current_state

def rotate(epd):
    array=epd.split(' ')
    board=array[0]
    color=array[1]
    castle=array[2]
    ep=array[3]

    b=''.join([rotate_string_map(s) for s in board])
    bd=b[::-1]

    color='w' if color=='b' else 'b'

    if castle!='-':
        k='k' if castle.find('K')!=-1 else ''
        q='q' if castle.find('Q')!=-1 else ''
        K='K' if castle.find('k')!=-1 else ''
        Q='Q' if castle.find('q')!=-1 else ''
        castle=K+Q+k+q 

    if ep!='-':
        fil=chr(ord('a')+ord('h')-ord(ep[0]))
        rank='6' if ep[1]=='3' else '3'
        ep=fil+rank

    return ' '.join([bd,color,castle,ep])

def rotate_string_map(s):
    if s.islower(): return s.upper()
    elif s.isupper(): return s.lower()
    else: return s

def is_terminal_state(board):
	"""This method tells if game could be finished in this position."""
	if board.is_game_over(claim_draw=True): return True
	else: return False 


def random_board_initialization(pieces, turn=chess.WHITE):
	"""Initialize a board randomly, given a piece configuration.
	NOTE: this method may only be appropriate when setting up endgames without
		too many pawns, as otherwise pawns may be put in an unrealistic setting,
		which is not ideal for learning and training.
	
	Arguments:
	pieces -- a list of Pieces, defining the piece conf on the board
	turn -- the side to move
	"""

	board=chess.Board.empty()
	board.turn=turn

	# make piece histograms to easily count if piece conf is valid
	w_piece_types=[p.piece_type for p in pieces if p.color==chess.WHITE]
	w_hist=[w_piece_types.count(i) for i in xrange(chess.PAWN,chess.KING+1)]
	b_piece_types=[p.piece_type for p in pieces if p.color==chess.BLACK]
	b_hist=[b_piece_types.count(i) for i in xrange(chess.PAWN,chess.KING+1)]
	
	# booleans indicating a certain problem with the piece conf
	king_prob=w_hist[chess.KING-1]!=1 or b_hist[chess.KING-1]!=1
	pawn_prob=w_hist[chess.PAWN-1]>8 or b_hist[chess.PAWN-1]>8
	too_many_pieces_prob=(
		sum(w_hist[chess.KNIGHT-1:chess.QUEEN])>15-w_hist[chess.PAWN-1]
		or sum(b_hist[chess.KNIGHT-1:chess.QUEEN])>15-b_hist[chess.PAWN-1]
		or (len([i for i in w_hist[chess.KNIGHT-1:chess.QUEEN]
			if i>10-w_hist[chess.PAWN-1]]) > 0)
		or (len([i for i in b_hist[chess.KNIGHT-1:chess.QUEEN]
			if i>10-b_hist[chess.PAWN-1]]) > 0)
		)

	if king_prob or pawn_prob or too_many_pieces_prob:
		raise Exception('There is a problem with the piece configuration')
	else:
		# put the pieces on the board in a random way
		while not board.is_valid():
			board.clear()
			free_sq_list=range(0,64)
			for p in pieces:
				if p.piece_type==chess.PAWN:
					# pawns should not be put on back ranks
					new_list=[i for i in free_sq_list if i>=8 and i <56]
					sq=new_list[int(random.uniform(0,len(new_list)))]
					free_sq_list.remove(sq)			
				else:
					rand=random.uniform(0,len(free_sq_list))
					sq=free_sq_list[int(rand)]
					free_sq_list.remove(sq)
				board.set_piece_at(sq,p)
        board.turn=turn
	return board	

def map_side_to_int(side):
	"""returns +1 for white and -1 for black"""
	return 2*int(side)-1
			

def rotate_board(board, hor=None, ver=None):
    sq=board.pieces(chess.KING,chess.BLACK).pop()
    if hor==None or ver==None:
        hor=(sq%8>=4)
        ver=(sq/8<4)
    h_mir=lambda sq: (sq/8)*8+7-(sq%8)
    v_mir=lambda sq: (7-sq/8)*8+(sq%8)
    temp=lambda sq,v: v_mir(sq) if v else sq
    new_sq=lambda sq,v,h: h_mir(temp(sq,v)) if h else temp(sq,v)
    if hor or ver:
        board_r=chess.Board.empty()
        board_r.turn=board.turn
        for c in xrange(2):
            for p in xrange(1,7):
                pset=board.pieces(p,int(c))
                while len(pset)>0:
                    sq=pset.pop()
                    board_r.set_piece_at(new_sq(sq,ver,hor),board.piece_at(sq))
    else:
        board_r=board
    return board_r, hor, ver

def rotated_push(board,move):
    p=board.piece_at(move.from_square)
    if p.symbol()=='k':
        sq=move.to_square
        hor=(sq%8>=4)
        ver=(sq/8<4)
        if hor or ver:
            move=rotated_move(move,hor,ver)
            board=rotate_board(board,hor,ver)[0]
    board.push(move)
    return board

def rotated_pop(board):
    board.pop()
    return rotate_board(board)[0] 

def rotated_move(move,h,v):
    h_mir=lambda sq: (sq/8)*8+7-(sq%8)
    v_mir=lambda sq: (7-sq/8)*8+(sq%8)
    temp=lambda sq,v: v_mir(sq) if v else sq
    new_sq=lambda sq,v,h: h_mir(temp(sq,v)) if h else temp(sq,v)
    from_sq=new_sq(move.from_square,v,h)
    to_sq=new_sq(move.to_square,v,h)
    return chess.Move(from_sq,to_sq) 
