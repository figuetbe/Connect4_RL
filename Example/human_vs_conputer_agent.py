# -*- coding: utf-8 -*-
# Code inspired by https://github.com/haje01/gym-tictactoe
import sys

import click
import random

#from connect4_env.py import Connect4Env, agent_by_mark, next_mark


class HumanAgent(object):
    def __init__(self, mark):
        self.mark = mark

    def act(self, ava_actions):
        while True:
            uloc = input("Enter location[1-7], q for quit: ")
            if uloc.lower() == 'q':
                return None
            try:
                action = int(uloc) - 1
                if action not in ava_actions:
                    raise ValueError()
            except ValueError:
                print("Illegal location: '{}'".format(uloc))
            else:
                break

        return action

class Computer(object):
    def __init__(self, mark):
        self.mark = mark

    def act(self, ava_actions):
        while True:
            action = random.choice(ava_actions)
            
            if action not in ava_actions:
                raise ValueError()
            else:
                break

        return action

@click.command(help="Play human agent.")
@click.option('-n', '--show-number', is_flag=True, default=False,
              show_default=True, help="Show location number in the board.")
def play(show_number):
    env = Connect4Env(show_number=show_number)
    agents = [HumanAgent('O'),
              Computer('X')]
    episode = 0
    while True:
        state = env.reset()
        _, mark = state
        done = False
        env.render()
        while not done:
            agent = agent_by_mark(agents, next_mark(mark))
            env.show_turn(True, mark)
            ava_actions = env.available_actions()
            action = agent.act(ava_actions)
            if action is None:
                sys.exit()

            state, reward, done, info = env.step(action)

            print('')
            env.render()
            if done:
                env.show_result(True, mark, reward)
                break
            else:
                _, mark = state
        episode += 1


if __name__ == '__main__':
    play()