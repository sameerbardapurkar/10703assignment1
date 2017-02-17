# coding: utf-8
"""Define the Queue environment from problem 3 here."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from gym import Env, spaces
from gym.envs.registration import register
import random
import copy
import numpy as np

class QueueEnv(Env):
    """Implement the Queue environment from problem 3.

    Parameters
    ----------
    p1: float
      Value between [0, 1]. The probability of queue 1 receiving a new item.
    p2: float
      Value between [0, 1]. The probability of queue 2 receiving a new item.
    p3: float
      Value between [0, 1]. The probability of queue 3 receiving a new item.

    Attributes
    ----------
    nS: number of states
    nA: number of actions
    P: environment model
    """
    metadata = {'render.modes': ['human']}

    SWITCH_TO_1 = 0
    SWITCH_TO_2 = 1
    SWITCH_TO_3 = 2
    SERVICE_QUEUE = 3

    def __init__(self, p1, p2, p3):
        self.state = self._reset()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete(
            [(1, 3), (0, 5), (0, 5), (0, 5)])
        self.nS = 648
        self.nA = 4
        self.P = {}
        probabilities = (1 - p1, 1 - p2, 1 - p3)
        for server in range(1,4):
            for q1 in range(0,6):
                for q2 in range(0,6):
                    for q3 in range(0,6):
                        state = (server, q1, q2, q3)
                        state_id = self._getStateID(state)
                        transition = {}
                        for action in range(0,4):
                            transition[action] = self._getState(state, action,
                                probabilities)
                        self.P[state_id] = transition

    def _getState(self, state, action, probabilities):
        return_state = []
        template = (1.0, state, 0.0, False)
        '''
        if(state[1] == 0 and state[2] == 0 and state[3] == 0):
            if(action == 3):
                ret_state = copy.deepcopy(template)
                ret_state[3] = True
                return ret_state
            else:
                ret_state = copy.deepcopy(template)
                ret_state[1][0] = action + 1
                ret_state[3] = True
                return ret_state
        '''
        if(action != 3):
            for i in (0, 1):
                for j in (0, 1):
                    for k in (0, 1):
                        ret_state = list(copy.deepcopy(template))
                        ret_state[1] = list(ret_state[1])
                        ret_state[1][0] = action + 1
                        ret_state[1][1] += i
                        ret_state[1][2] += j
                        ret_state[1][3] += k
                        ret_state[1][1]  = min(ret_state[1][1], 5)
                        ret_state[1][2]  = min(ret_state[1][2], 5)
                        ret_state[1][3]  = min(ret_state[1][3], 5)
                        ret_state[1] = self._getStateID(ret_state[1])
                        ret_state[0] = abs(i - probabilities[0])*abs(j - probabilities[1])*abs(k - probabilities[2])
                        #ret_state[1] = tuple(ret_state[1])
                        return_state.append(tuple(ret_state))
        else:
            for i in (0, 1):
                for j in (0, 1):
                    for k in (0, 1):
                        ret_state = list(copy.deepcopy(template))
                        ret_state[1] = list(ret_state[1])
                        if(ret_state[1][state[0]] > 0):
                            ret_state[1][state[0]] -= 1
                            ret_state[2] = 1.0
                        ret_state[1][1] += i
                        ret_state[1][2] += j
                        ret_state[1][3] += k
                        ret_state[1][1]  = min(ret_state[1][1], 5)
                        ret_state[1][2]  = min(ret_state[1][2], 5)
                        ret_state[1][3]  = min(ret_state[1][3], 5)
                        ret_state[1] = self._getStateID(ret_state[1])
                        ret_state[0] = abs(i - probabilities[0])*abs(j - probabilities[1])*abs(k - probabilities[2])
                        #ret_state[1] = tuple(ret_state[1])
                        return_state.append(tuple(ret_state))
        return self._checkTerminal(return_state)

    def _checkTerminal(self, state):
        state1 = list(state)
        for i in range(0, len(state1)):
            sub_state = (state1[i][1])
            if(sub_state == 0):
                state1[i] = list(state1[i])
                state1[i][3] = True
                state1[i] = tuple(state1[i])
        return tuple(state1)
    def _reset(self):
        """Reset the environment.

        The server should always start on Queue 1.

        Returns
        -------
        (int, int, int, int)
          A tuple representing the current state with meanings
          (current queue, num items in 1, num items in 2, num items in
          3).
        """
        server = 1
        q1 = random.randint(0,5)
        q2 = random.randint(0,5)
        q3 = random.randint(0,5)
        rand_init_state = (server, q1, q2, q3)
        self.steps_beyond_done = None
        self.state = self._getStateID(rand_init_state)
        return self._getStateID(rand_init_state)

    def _getStateID(self, state):
        return state[3] + state[2]*6 + state[1]*36 + (state[0]-1)*216

    def _step(self, action):
        """Execute the specified action.

        Parameters
        ----------
        action: int
          A number in range [0, 3]. Represents the action.

        Returns
        -------
        (state, reward, is_terminal, debug_info)
          State is the tuple in the same format as the reset
          method. Reward is a floating point number. is_terminal is a
          boolean representing if the new state is a terminal
          state. debug_info is a dictionary. You can fill debug_info
          with any additional information you deem useful.
        """
        state_id = self.state
        transition = self.P[state_id][action]
        probs = []
        for i in range(0, len(transition)):
            probs.append(transition[i][0])
        index = np.where(np.random.multinomial(1,probs))[0][0]
        next_state = []
        next_state.append(transition[index][1])
        next_state.append(transition[index][2])
        next_state.append(transition[index][3])
        next_state.append(None)
        return tuple(next_state)

    def _render(self, mode='human', close=False):
        pass

    def _seed(self, seed=None):
        """Set the random seed.

        Parameters
        ----------
        seed: int, None
          Random seed used by numpy.random and random.
        """
        #self.np_random, seed = seeding.np_random(seed)
        #return[seed]
        pass

    def query_model(self, state, action):
        """Return the possible transition outcomes for a state-action pair.

        This should be in the same format at the provided environments
        in section 2.

        Parameters
        ----------
        state
          State used in query. Should be in the same format at
          the states returned by reset and step.
        action: int
          The action used in query.

        Returns
        -------
        [(prob, nextstate, reward, is_terminal), ...]
          List of possible outcomes
        """
        return self.P[self._getStateID(state)][action]
    
    def get_action_name(self, action):
        if action == QueueEnv.SERVICE_QUEUE:
            return 'SERVICE_QUEUE'
        elif action == QueueEnv.SWITCH_TO_1:
            return 'SWITCH_TO_1'
        elif action == QueueEnv.SWITCH_TO_2:
            return 'SWITCH_TO_2'
        elif action == QueueEnv.SWITCH_TO_3:
            return 'SWITCH_TO_3'
        return 'UNKNOWN'


register(
    id='Queue-1-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .9,
            'p3': .1})

register(
    id='Queue-2-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .1,
            'p3': .1})
