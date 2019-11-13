# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 20:16:13 2019

@author: figu
"""

import random

#from gym_tictactoe.env import TicTacToeEnv, agent_by_mark, check_game_status,\
#    after_action_state, tomark, next_mark


class BaseAgent(object):
    def __init__(self, mark):
        self.mark = mark

    def act(self, state, ava_actions):
        for action in ava_actions:
            nstate = after_action_state(state, action)
            col = action
            row = get_row(nstate[0], col)
            gstatus = check_game_status(nstate[0], row, col)
            if gstatus > 0:
                if tomark(gstatus) == self.mark:
                    return action
        return random.choice(ava_actions)


def play(max_episode=10):
    episode = 0
    start_mark = 'O'
    env = Connect4Env()
    agents = [BaseAgent('O'),
              BaseAgent('X')]

    while episode < max_episode:
        env.set_start_mark(start_mark)
        state = env.reset()
        _, mark = state
        done = False
        while not done:
            env.show_turn(True, mark)

            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            action = agent.act(state, ava_actions)
            state, reward, done, info = env.step(action)
            env.render()

            if done:
                env.show_result(True, mark, reward)
                break
            else:
                _, mark = state

        # rotate start
        start_mark = next_mark(start_mark)
        episode += 1


if __name__ == '__main__':
    play()