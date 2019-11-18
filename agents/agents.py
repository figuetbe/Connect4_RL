# -*- coding: utf-8 -*-
"""
This file contains the different agents to play / learn a game

@author: figu
"""
import random
import numpy as np
from envs import connect4_env as c4
from collections import defaultdict
from tqdm import tqdm as _tqdm
tqdm = _tqdm


DEFAULT_VALUE = 0
EPISODE_CNT = 17000
BENCH_EPISODE_CNT = 3000
MODEL_FILE = 'best_td_agent.dat'
EPSILON = 0.08
ALPHA = 0.4
NO_REWARD = 0
O_REWARD = 1
X_REWARD = -1

st_values = {}
st_visits = defaultdict(lambda: 0)


class HumanAgent(object):
    def __init__(self, mark):
        self.mark = mark

    def act(self, ava_actions):
        while True:
            uloc = input("Enter location[1-7], q for quit: ")
            if uloc.lower() == 'q':
                return None
            try:
                action = int(uloc)-1
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
            print(action)
            if action not in ava_actions:
                raise ValueError()
            else:
                break
        return action


class BaseAgent(object):
    def __init__(self, mark):
        self.mark = mark

    def act(self, state, ava_actions):
        for action in ava_actions:
            nstate = c4.after_action_state(state, action)
            col = action
            row = c4.get_row(np.array(nstate[0]), col)
            gstatus = c4.check_game_status(np.array(nstate[0]), row, col)
            if gstatus > 0:
                if c4.tomark(gstatus) == self.mark:
                    return action
        return random.choice(ava_actions)


class TDAgent(object):
    def __init__(self, mark, epsilon, alpha):
        self.mark = mark
        self.alpha = alpha
        self.epsilon = epsilon
        self.episode_rate = 1.0

    def act(self, state, ava_actions):
        return self.egreedy_policy(state, ava_actions)

    def egreedy_policy(self, state, ava_actions):
        """Returns action by Epsilon greedy policy.
        Return random action with epsilon probability or best action.
        Args:
            state (tuple): Board status + mark
            ava_actions (list): Available actions
        Returns:
            int: Selected action.
        """
        e = random.random()
        if e < self.epsilon * self.episode_rate:
            action = self.random_action(ava_actions)
        else:
            action = self.greedy_action(state, ava_actions)
        return action

    def random_action(self, ava_actions):
        return random.choice(ava_actions)

    def greedy_action(self, state, ava_actions):
        """Return best action by current state value.
        Evaluate each action, select best one. Tie-breaking is random.
        Args:
            state (tuple): Board status + mark
            ava_actions (list): Available actions
        Returns:
            int: Selected action
        """
        assert len(ava_actions) > 0

        ava_values = []
        for action in ava_actions:
            nstate = c4.after_action_state(state, action)
            nval = self.ask_value(nstate)
            ava_values.append(nval)
            vcnt = st_visits[str(nstate)]

        # select best action for 'O' or 'X'
        if self.mark == 'O':
            indices = best_val_indices(ava_values, max)
        else:
            indices = best_val_indices(ava_values, min)

        # tie breaking by random choice
        aidx = random.choice(indices)
        action = ava_actions[aidx]

        return action

    def ask_value(self, state):
        """Returns value of given state.
        If state is not exists, set it as default value.
        Args:
            state (tuple): State.
        Returns:
            float: Value of a state.
        """
        if str(state) not in st_values:
            gstatus = c4.check_game_status2(state[0])
            val = DEFAULT_VALUE
            # win
            if gstatus > 0:
                val = O_REWARD if self.mark == 'O' else X_REWARD
            set_state_value(state, val)
        return st_values[str(state)]

    def backup(self, state, nstate, reward):
        """Backup value by difference and step size.
        Execute an action then backup Q by best value of next state.
        Args:
            state (tuple): Current state
            nstate (tuple): Next state
            reward (int): Immediate reward from action
        """
        val = self.ask_value(state)
        nval = self.ask_value(nstate)
        diff = nval - val
        val2 = val + self.alpha * diff

        set_state_value(state, val2)


def best_val_indices(values, fn):
    best = fn(values)
    return [i for i, v in enumerate(values) if v == best]


def reset_state_values():
    global st_values, st_visits
    st_values = {}
    st_visits = defaultdict(lambda: 0)


def set_state_value(state, value):
    st_visits[str(state)] += 1
    st_values[str(state)] = value


def learn_td(max_episode, epsilon, alpha):
    _learn(max_episode, epsilon, alpha)
    return st_values, st_visits


def _learn(max_episode, epsilon, alpha):
    """Learn by episodes.
    Make two TD agent, and repeat self play for given episode count.
    Update state values as reward coming from the environment.
    Args:
        max_episode (int): Episode count.
        epsilon (float): Probability of exploration.
        alpha (float): Step size.
    """
#    reset_state_values()

    env = c4.Connect4Env()
    agents = [TDAgent('O', epsilon, alpha),
              TDAgent('X', epsilon, alpha)]

    start_mark = 'O'
    for i in tqdm(range(max_episode)):
        episode = i + 1

        # reset agent for new episode
        for agent in agents:
            agent.episode_rate = episode / float(max_episode)

        env.set_start_mark(start_mark)
        state = env.reset()
        _, mark = state
        done = False
        while not done:
            agent = c4.agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            action = agent.act(state, ava_actions)

            # update (no rendering)
            nstate, reward, done, info = env.step(action)
            agent.backup(state, nstate, reward)

            if done:
                # set terminal state value
                set_state_value(state, reward)

            _, mark = state = nstate

        # rotate start
        start_mark = c4.next_mark(start_mark)


def load_training(filename1: str, filename2: str):
    st_values, st_visits = learn_td(100, 0.08, 0.8)
    np.load.__defaults__ = (None, True, True, 'ASCII')
    P = np.load(filename1)
    st_visits = defaultdict(lambda: 0)
    st_visits.update(P.item())

    P = np.load(filename2)
    st_values = {}
    st_values.update(P.item())
    np.load.__defaults__ = (None, False, True, 'ASCII')
    return st_values, st_visits


def save_training(filename1: str, filename2: str):
    np.save(filename1, np.array(dict(st_values)))
    np.save(filename2, np.array(dict(st_visits)))
