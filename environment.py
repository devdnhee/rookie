import chess
import copy
import numpy as np
from chessmoveswrapper import ChessMovesEngine
from chess_utils import random_board_initialization, map_side_to_int,piece_configurations
import cPickle as cp
import tablebases as tb
import settings

def load_DS(ds_f):
    """loading in a file full of epd-strings to the settings DSET. Random
    starting positions are sampled from this dataset to generate chess
    positions for episodes.
    """
    with open(ds_f,'rb') as f:
        settings.params['DSET']=cp.load(f)
    settings.params['S']=len(settings.params['DSET'])
    print 'SIZE starpos: ', settings.params['S']

class Environment:
    """
    class describing the chess board (the environment in the RL paradigm)

    Attributes:
        current_state: describing the state of the chessboard in epd-format, so
            game contextual elements like number of moves are hidden (but
            stored in chess board)
        board: the chess position with all its information. Only copies of this
            instance will be shared to protect its consistency
        turn: whose turn it is (also stored in board)
        draw_r: reward earned after white obtains a draw position
        mate_r: reward earned after white obtains a mate position
        move_r: reward earned after each halfmove of white 
    """

    @staticmethod
    def draw_state(s):
        """
        utility function, draw the chess board to the respective
        state
        """
        print chess.Board.from_epd(s)[0]

    @staticmethod
    def terminal_state(s):
        """
        state terminal? 
        """ 
        if chess.Board.from_epd(s)[0].is_game_over(claim_draw=True): return True
        else: return False

    def __init__(self,state=None,draw_r=0,move_r=0,mate_r=1,repet_r=-0.0,dset=True,use_tb=False):
        """
        Constructor
        """
        assert mate_r>0.5
        self.hist=dict()
        self.draw_r=draw_r
        self.move_r=move_r
        self.mate_r=mate_r
        self.repet_r=0.0
        self.rewards={'DRAW':draw_r,'MATE':mate_r,'MOVE':move_r,'REPETITION':repet_r}
        self.dset=dset
        self.use_tb=use_tb
        if state is None:
            self.reset()
        else:
            self.engine=ChessMovesEngine(state)
            self.current_state=self.engine.board
        self.hist[self.current_state]=1
        self.actions=self.engine.get_moves()
        self.dset=dset

    def is_terminal(self):
        """
        decide if the episode should be finished
        """
        return (self.engine.is_game_over() or
            (self.use_tb and tb.probe_result(self.current_state) is not
             None)) 

    def is_game_over(self):
        return self.engine.is_game_over()

    def result(self):
        """return string result"""
        if not self.use_tb:
            return self.engine.get_result()
        else:            
            res=tb.probe_result(self.current_state)
            if res==1:
                return '1-0'
            elif res==0:
                return '1/2-1/2'
            elif res==-1:
                return '0-1'
            else: return self.engine.get_result()

    def int_result(self):
        """DEPRECEATED AND NOT UP TO DATE. return int result"""
        if self.tb:
            return tb.probe_result(self.current_state)
        else:
            res=self.engine.get_result()
            if res=='1-0':
                return 1 
            elif res=='1/2-1/2':
                return 0
            elif res=='0-1':
                return -1

    def outcome(self):
        """return outcome"""
        res=self.result()
        if res=='1-0':
            return 1
        elif res=='0-1':
            return -1
        elif res=='1/2-1/2':
            return 0
        else:
            return None

    def int_outcome(self):
        res=self.outcome()
        return res if res is not None else 0

    def action_outcome(self,a):
        res=self.engine.get_move_result(a)
        if res=='1-0':
            return 1
        elif res=='0-1':
            return -1
        elif res=='1/2-1/2':
            return 0
        else:
            return 0 

    def draw(self):
        """
        draw Environment for graphical interpretation
        """
        print chess.Board.from_epd(self.current_state)[0] 

    def get_rewards(self,sit):
        return self.rewards[sit]

    def get_turn(self):
        """
        transform to boolean because easier to calculate with
        """
        return self.engine.turn()==ChessMovesEngine.WHITE

    def reset(self,state=None):
        """
        resets the position for a new episode
        """
        no_success=True
        while no_success:
            if state is None:
                # this is probably pretty slow, but acceptable as only done once
                # every chess game
                # the reason it's slow is because the method in utils is not
                # adapted
                if not self.dset:
                    turn=int(np.round(np.random.uniform())) 
                    side=[chess.WHITE,chess.BLACK][turn]
                    board=random_board_initialization(piece_configurations['KRk'],turn=side)
                    self.current_state=board.epd()
                else:
                    # get position from dataset and play some random moves.
                    board=settings.params['DSET'][np.random.randint(settings.params['S'])] 
                    self.current_state=board
                    cunt=0
                    self.engine=ChessMovesEngine(self.current_state)
                    self.actions=self.engine.get_moves()
                    max_count=settings.params['RAND']
                    while cunt<max_count and not self.is_terminal() and len(self.actions)>0:
                        cunt+=1
                        rand=np.random.randint(len(self.actions))
                        a=self.actions[rand]
                        self.perform_action(a)
                        if cunt!=max_count:
                            self.engine=ChessMovesEngine(self.current_state)
                            self.actions=self.engine.get_moves()
            else:
                self.current_state=state 
            self.engine=ChessMovesEngine(self.current_state)
            self.actions=self.engine.get_moves()
            if len(self.actions)>0:
                no_success=False
            if self.is_terminal():
                no_success=True

    def is_win(self):
        res=self.result()
        return res=='1-0' or res=='0-1'

    def reward(self,a):
        """
        returns the reward compatible with action
        (compatible with color, so eg. black gets a positive reward when the
        reward would've been negative for white)
        """
        mr=0
        dr=0
        rr=0

        if a is None:
            self.draw()
        
        try:
            result=self.engine.get_move_result(a)
        except:
            self.draw()
            print a
            raise
        if result==ChessMovesEngine.WHITE_WIN or result==ChessMovesEngine.BLACK_WIN:
            mr=self.mate_r
        elif result==ChessMovesEngine.DRAW:
            dr=self.draw_r
        elif self.use_tb:
            s_n=self.engine.moves[a][0]
            if sum([int(c.isalpha()) for c in s_n])<=(5+1):
                mr=self.mate_r*tb.probe_result(s_n) 
        if self.hist.get(self.current_state,0)>1:
            pass
        s=mr+dr+self.move_r
        # inverted, because reward is for the one playing last move
        return map_side_to_int(self.get_turn())*s 

    def perform_action(self,a):
        """
        action is a uci move
        go to the next state
        return:
            (reward,next state)
        """
        r=self.reward(a)
        self.engine.do_move(a)
        self.current_state=self.engine.board
        if self.current_state in self.hist:
            r+=map_side_to_int(self.get_turn())*self.repet_r
            self.hist[self.current_state]+=1
        else:
            self.hist[self.current_state]=1
        self.actions=self.engine.get_moves()
        return r, self.current_state 

    def get_as_pairs(self):
        """
        get action state pairs obtainable from current state
        """
        return [(a,self.engine.moves[a][0]) for a in self.actions]

def test():
    """
    MAY NOT BE UP TO DATE
    """
    #b=chess.Board.from_epd('6k1/8/8/5K2/8/8/8/3R4 w - -')[0]
    epd='6k1/8/8/5K2/8/8/8/3R4 w - -'
    env=Environment(state=epd,draw_r=-1,move_r=-0.01,repet_r=-0.33,dset=False)
    print env.get_as_pairs()
    env.draw() 
    ucis=['d1d7','g8h8','f5g6','h8g8','g6f5','g8h8','f5g6','h8g8','d7d8']
    for m in ucis:
        print env.perform_action(m)
        env.draw()
    assert env.is_terminal()
    env.reset()
    env.draw()

    epd='7k/8/5K2/8/8/8/6R1/8 w - -'
    env=Environment(state=epd,draw_r=-1,move_r=-0.01)
    env.draw() 
    m='g2g7'
    print env.perform_action(m)
    env.draw()

    for i in xrange(10):
        env=Environment()
        env.draw()
        print('\n')

    epd='6k1/2p4p/2P5/1P6/4K3/8/8/8 w - -'
    env=Environment(state=epd)
    env.draw()
    ucis=['b5b6','c7b6']
    for m in ucis:
        print env.perform_action(m)
        print env.result(),env.is_win()
        print env.is_terminal()
        env.draw()

if __name__=='__main__':
    test()
