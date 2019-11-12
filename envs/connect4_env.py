# -*- coding: utf-8 -*-
# Code inspired by https://github.com/haje01/gym-tictactoe

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

import logging



CODE_MARK_MAP = {0: ' ', 1: 'O', 2: 'X'}
NO_REWARD = 0
O_REWARD = 1
X_REWARD = 1
n_cols = 7
n_rows = 6

def tomark(code):
    return CODE_MARK_MAP[code]


def tocode(mark):
    return 1 if mark == 'O' else 2


def next_mark(mark):
    return 'X' if mark == 'O' else 'O'


def agent_by_mark(agents, mark):
    for agent in agents:
        if agent.mark == mark:
            return agent


def after_action_state(state, action):
    """Execute an action and returns resulted state.
    Args:
        state (tuple): Board status + mark
        action (int): Action to run
    Returns:
        tuple: New state
    """

    board, mark = state
#    nboard = list(board[:])
    row = get_row(board, action)
    nboard[action, row] = tocode(mark)
#    nboard = tuple(nboard)
    return nboard, next_mark(mark)


class Connect4Env(gym.Env):
    metadata = {'render.modes': ['human']}
    symbols = ['O', ' ', 'X']

    def __init__(self, show_number=False):
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Discrete(7*6)  # Grid dimensions
        self.set_start_mark('O')
        self.seed()
        self.reset()

    def set_start_mark(self, mark):
        self.start_mark = mark

    def step(self, action):
        """Step environment by action.
        Args:
            action (int): Location
        Returns:
            list: Obeservation
            int: Reward
            bool: Done
            dict: Additional information
        """
        assert self.action_space.contains(action)
        col = action  # If it starts at 1

        if self.done:
            return self._get_obs(), 0, True, None

        reward = NO_REWARD

        # place
        row = get_row(self.board, col)

        self.board[row, col] = tocode(self.mark)

        status = check_game_status(self.board, row, col)
#        logging.debug("check_game_status board {} mark '{}'"
#                      " status {}".format(self.board, self.mark, status))

        if status >= 0:
            self.done = True
            if status in [1, 2]:
                # always called by self
                reward = O_REWARD if self.mark == 'O' else X_REWARD

        # switch turn
        self.mark = next_mark(self.mark)
        return self._get_obs(), reward, self.done, None

    def reset(self):
        self.board = np.zeros((6, 7))
        self.mark = self.start_mark
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        return self.board, self.mark

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human':
            self._show_board(print)  # NOQA
            print('')
#        else:
#            self._show_board(logging.info)
#            logging.info('')

    def show_episode(self, human, episode):
        self._show_episode(print if human else logging.warning, episode)

    def _show_episode(self, showfn, episode):
        showfn("==== Episode {} ====".format(episode))

    def _show_board(self, showfn):
        """Draw tictactoe board."""
        print(self.board)

    def show_turn(self, human, mark):
        self._show_turn(print if human else logging.info, mark)

    def _show_turn(self, showfn, mark):
        showfn("{}'s turn.".format(mark))

    def show_result(self, human, mark, reward):
        self._show_result(print if human else logging.info, mark, reward)

    def _show_result(self, showfn, mark, reward):
        print('Finished')
#        status = check_game_status(self.board)
#        assert status >= 0
#        if status == 0:
#            showfn("==== Finished: Draw ====")
#        else:
#            msg = "Winner is '{}'!".format(tomark(status))
#            showfn("==== Finished: {} ====".format(msg))
#        showfn('')

    def available_actions(self):
        return [col for col in range(7) if get_row(self.board, col) != -1]


def get_row(board, col):
    for row in range(5, 0, -1):
        if board[row, col] == 0:
            return row      
    return -1


def checkWin(board, row, col, marker):
    boardHeight, boardWidth = board.shape
    
    i, j = row, col

    # Right_diag:
    if isWin(board, [[i, j], [i-1, j+1], [i-2, j+2], [i-3, j+3]], marker):
        return True
    # left_diag:
    if isWin(board, [[i, j], [i-1, j-1], [i-2, j-2], [i-3, j-3]], marker):
        return True
    # right:
    if isWin(board, [[i, j], [i, j+1], [i, j+2], [i, j+3]], marker):
        return True
    # left:
    if isWin(board, [[i, j], [i, j-1], [i, j-2], [i, j-3]], marker):
        return True
    # down:
    if isWin(board, [[i, j], [i-1, j], [i-2, j], [i-3, j]], marker):
        return True
    return False

def isWin(board, mask, marker):
    print(mask)
    win = [marker]*4
    min_dim, max_dim = [0, 0], [board.shape[0]-1, board.shape[1]-1]
    if ~np.all([(min_dim[0] <= m[0] <= max_dim[0]) &
                (min_dim[1] <= m[1] <= max_dim[1]) for m in mask]):
        return False
#    print(board, [board[m[0], m[1]] for m in mask])
    if [board[m[0], m[1]] for m in mask] == win:
        return True
    return False
    
def isfull(board):
    return np.count_nonzero(board) == 42


def check_game_status(board, row, col):
    """Return game status by current board status.
    Args:
        board (list): Current board state
    Returns:
        int:
            -1: game in progress
            0: draw game,
            1 or 2 for finished game(winner mark code).

    """
    if checkWin(board, row, col, 1):
        return 1
    if checkWin(board, row, col, 2):
        return 2
    if isfull(board):
        return 0
    return -1


