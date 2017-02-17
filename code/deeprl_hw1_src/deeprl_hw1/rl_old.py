# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import gym
import deeprl_hw1.lake_envs as drl
import copy
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

left = drl.LEFT
down = drl.DOWN
right = drl.RIGHT
up = drl.UP
actions = [left, down, right, up]
actions_dict = {}
actions_dict[left] = 'L'
actions_dict[right] = 'R'
actions_dict[down] = 'D'
actions_dict[up] = 'U'
def evaluate_policy(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Evaluate the value of a policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value_func = np.zeros(env.nS)
    stop_iterating = False;
    count = 0
    for n in range(0, max_iterations):
      count += 1
      if(stop_iterating):
        break
      stop_iterating = True;
      for i in range(0, env.nS):
        transition = env.P[i][policy[i]]
        new_value = 0
        for j in range(0, len(transition)):
          probability = transition[j][0]

          reward = transition[j][2]
          nextstate = transition[j][1]
          new_value += probability*(reward + gamma*value_func[nextstate])
        if(math.fabs(new_value - value_func[i]) > tol):
          stop_iterating = False;
          value_func[i] = new_value
    return value_func, count



def value_function_to_policy(env, gamma, value_func):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """
    policy = np.zeros(env.nS, dtype='int')
    for i in range(0, env.nS):
      max_value = float('-inf')
      for j in actions:
        transition = env.P[i][j]
        new_value = 0
        for k in range(0, len(transition)):
          probability = transition[k][0]
          reward = transition[k][2]
          nextstate = transition[k][1]
          new_value += probability*(reward + gamma*value_func[nextstate])
        if(new_value > max_value):
          policy[i] = j
          max_value = new_value
    return policy


def improve_policy(env, gamma, value_func, policy):
    """Given a policy and value function improve the policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    policy_changed = False;
    new_policy = copy.deepcopy(policy)
    for i in range(0, env.nS):
      max_value = float('-inf')
      for j in actions:
        transition = env.P[i][j]
        new_value = 0
        for k in range(0, len(transition)):
          probability = transition[k][0]
          reward = transition[k][2]
          nextstate = transition[k][1]
          new_value += probability*(reward + gamma*value_func[nextstate])
        if(new_value > max_value):
          max_value = new_value
          new_policy[i] = j
      if(new_policy[i] != policy[i]):
        policy_changed = True; 
    return policy_changed, new_policy


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    You should use the improve_policy and evaluate_policy methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    policy_changed = True
    val_count = 0
    pol_count = 0
    while(policy_changed):
      pol_count += 1
      value_func, val_count_temp = evaluate_policy(env, gamma, policy)
      val_count += val_count_temp
      (policy_changed, new_policy) = improve_policy(env, gamma, value_func,
                                                    policy)
      policy = copy.deepcopy(new_policy)
    return policy, value_func, pol_count, val_count


def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    policy = np.zeros(env.nS, dtype='int')
    stop_iterating = True;
    value_func = np.zeros(env.nS)
    num_iter = 0
    for n in range(0, max_iterations):
      value_func_old = copy.deepcopy(value_func)
      num_iter += 1
      for i in range(0, env.nS):
        max_value = float('-inf')
        for j in actions:
          transition = env.P[i][j]
          new_value = 0
          for k in range(0, len(transition)):
            probability = transition[k][0]
            reward = transition[k][2]
            nextstate = transition[k][1]
            #print(reward, nextstate)
            new_value += probability*(reward + gamma*value_func_old[nextstate])
            
          if(new_value > max_value):
            max_value = new_value
        #print(max_value, value_func_old[i])
        value_func[i] = copy.deepcopy(max_value)
      if(is_same(value_func, value_func_old)):
        break
      #print("****************************************************")    
    return value_func, num_iter

def is_same(arr1, arr2, tol = 1e-3):
  same = True
  for i in range(0, len(arr1)):
    if(math.fabs(arr1[i] - arr2[i]) > tol):
      same = False
      break
  return same
def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)
    print(str_policy)

def take_a_walk(env, gamma, policy, num_times=1):
  returns = 0
  for n in range(0, num_times):
    terminate = 0
    discount = 1
    state = env.reset()
    while(terminate == 0):
      (state, reward, terminate, info) = env.step(policy[state])
      returns += discount*reward
      discount = discount*gamma
  returns = returns/num_times
  return returns
#Main body


print("Q2 Part(a)")
print("**********************************************")
print("4X4")
print("**********************************************")
env = gym.make('Deterministic-4x4-FrozenLake-v0')
env.render()
print("**********************************************")
#Run policy iteration and print and render its output
gamma = 0.9
start_time = time.time()
(policy, value_func, pol_count, val_count) = policy_iteration(env, gamma)
print("a. Finished policy iteration in %s s" % (time.time() - start_time))
print("   carried out %d policy improvement steps and %d policy evaluations" 
      % (pol_count, val_count))
print("b.")
print_policy(policy.reshape(4,4), actions_dict)
print(value_func.reshape(4,4))
plt.imshow(value_func.reshape(4,4))
plt.colorbar()
plt.savefig('/home/sameer/10-703/assignment1/results/q_2_parta_c_4_pol_iter.eps',
         bbox_inches='tight')
plt.clf()

#Run value iteration and print and render its output
gamma = 0.9
start_time = time.time()
(val_func_alternate, num_iter) = value_iteration(env, gamma)
print("d. Finished value iteration in %s s" % (time.time() - start_time))
print("   %d iterations were required" % num_iter)
plt.imshow(val_func_alternate.reshape(4,4))
plt.colorbar()
plt.savefig('/home/sameer/10-703/assignment1/results/q_2_parta_e_4_val_iter.eps',
         bbox_inches='tight')
plt.clf()
policy_alternate = value_function_to_policy(env, gamma, val_func_alternate)
print("h.")
print_policy(policy_alternate.reshape(4,4), actions_dict)

#make an agent take a walk in the environment
print("i. ")
gamma = 0.9
returns = take_a_walk(env, gamma, policy)
print(returns)


print("Q2 Part(a)")
print("**********************************************")
print("8X8")
print("**********************************************")
env = gym.make('Deterministic-8x8-FrozenLake-v0')
env.render()
print("**********************************************")
#Run policy iteration and print and render its output
gamma = 0.9
start_time = time.time()
(policy, value_func, pol_count, val_count) = policy_iteration(env, gamma)
print("a. Finished policy iteration in %s s" % (time.time() - start_time))
print("   carried out %d policy improvement steps and %d policy evaluations" 
      % (pol_count, val_count))
print("b.")
print_policy(policy.reshape(8,8), actions_dict)
print(value_func.reshape(8,8))
plt.imshow(value_func.reshape(8,8))
plt.colorbar()
plt.savefig('/home/sameer/10-703/assignment1/results/q_2_parta_c_8_pol_iter.eps',
         bbox_inches='tight')
plt.clf()

#Run value iteration and print and render its output
gamma = 0.9
start_time = time.time()
(val_func_alternate, num_iter) = value_iteration(env, gamma)
print("d. Finished value iteration in %s s" % (time.time() - start_time))
print("   %d iterations were required" % num_iter)
plt.imshow(val_func_alternate.reshape(8,8))
plt.colorbar()
plt.savefig('/home/sameer/10-703/assignment1/results/q_2_parta_e_8_val_iter.eps',
         bbox_inches='tight')
plt.clf()
policy_alternate = value_function_to_policy(env, gamma, val_func_alternate)
print("h.")
print_policy(policy_alternate.reshape(8,8), actions_dict)

#make an agent take a walk in the environment
print("i. ")
gamma = 0.9
returns = take_a_walk(env, gamma, policy)
print(returns)

print("Q2 Part(b)")
print("**********************************************")
print("4X4")
print("**********************************************")
env = gym.make('Stochastic-4x4-FrozenLake-v0')
env.render()
print("**********************************************")
#Run policy iteration and print and render its output
gamma = 0.9
start_time = time.time()
(policy, value_func, pol_count, val_count) = policy_iteration(env, gamma)
print("a. Finished policy iteration in %s s" % (time.time() - start_time))
print("   carried out %d policy improvement steps and %d policy evaluations" 
      % (pol_count, val_count))
print("b.")
print_policy(policy.reshape(4,4), actions_dict)
print(value_func.reshape(4,4))
plt.imshow(value_func.reshape(4,4))
plt.colorbar()
plt.savefig('/home/sameer/10-703/assignment1/results/q_2_partb_c_4_pol_iter.eps',
         bbox_inches='tight')
plt.clf()

#Run value iteration and print and render its output
gamma = 0.9
start_time = time.time()
(val_func_alternate, num_iter) = value_iteration(env, gamma)
print("d. Finished value iteration in %s s" % (time.time() - start_time))
print("   %d iterations were required" % num_iter)
plt.imshow(val_func_alternate.reshape(4,4))
plt.colorbar()
plt.savefig('/home/sameer/10-703/assignment1/results/q_2_partb_e_4_val_iter.eps',
         bbox_inches='tight')
plt.clf()
policy_alternate = value_function_to_policy(env, gamma, val_func_alternate)
print("h.")
print_policy(policy_alternate.reshape(4,4), actions_dict)

#make an agent take a walk in the environment
print("i. ")
gamma = 0.9
returns = take_a_walk(env, gamma, policy, 100)
print(returns)


print("Q2 Part(b)")
print("**********************************************")
print("8X8")
print("**********************************************")
env = gym.make('Stochastic-8x8-FrozenLake-v0')
env.render()
print("**********************************************")
#Run policy iteration and print and render its output
gamma = 0.9
start_time = time.time()
(policy, value_func, pol_count, val_count) = policy_iteration(env, gamma)
print("a. Finished policy iteration in %s s" % (time.time() - start_time))
print("   carried out %d policy improvement steps and %d policy evaluations" 
      % (pol_count, val_count))
print("b.")
print_policy(policy.reshape(8,8), actions_dict)
print(value_func.reshape(8,8))
plt.imshow(value_func.reshape(8,8))
plt.colorbar()
plt.savefig('/home/sameer/10-703/assignment1/results/q_2_partb_c_8_pol_iter.eps',
         bbox_inches='tight')
plt.clf()

#Run value iteration and print and render its output
gamma = 0.9
start_time = time.time()
(val_func_alternate, num_iter) = value_iteration(env, gamma)
print("d. Finished value iteration in %s s" % (time.time() - start_time))
print("   %d iterations were required" % num_iter)
plt.imshow(val_func_alternate.reshape(8,8))
plt.colorbar()
plt.savefig('/home/sameer/10-703/assignment1/results/q_2_partb_e_8_val_iter.eps',
         bbox_inches='tight')
plt.clf()
policy_alternate = value_function_to_policy(env, gamma, val_func_alternate)
print("h.")
print_policy(policy_alternate.reshape(8,8), actions_dict)

#make an agent take a walk in the environment
print("i. ")
gamma = 0.9
returns = take_a_walk(env, gamma, policy, 100)
print(returns)

print("Q2 Part(c)")
print("**********************************************")
print("4X4")
print("**********************************************")
env = gym.make('Deterministic-4x4-neg-reward-FrozenLake-v0')
env.render()
print("**********************************************")
#Run policy iteration and print and render its output
gamma = 0.9
start_time = time.time()
(policy, value_func, pol_count, val_count) = policy_iteration(env, gamma)
print("a. Finished policy iteration in %s s" % (time.time() - start_time))
print("   carried out %d policy improvement steps and %d policy evaluations" 
      % (pol_count, val_count))
print("b.")
print_policy(policy.reshape(4,4), actions_dict)
print(value_func.reshape(4,4))
plt.imshow(value_func.reshape(4,4))
plt.colorbar()
plt.savefig('/home/sameer/10-703/assignment1/results/q_2_partc_c_4_pol_iter.eps',
         bbox_inches='tight')
plt.clf()

#Run value iteration and print and render its output
gamma = 0.9
start_time = time.time()
(val_func_alternate, num_iter) = value_iteration(env, gamma)
print("d. Finished value iteration in %s s" % (time.time() - start_time))
print("   %d iterations were required" % num_iter)
plt.imshow(val_func_alternate.reshape(4,4))
plt.colorbar()
plt.savefig('/home/sameer/10-703/assignment1/results/q_2_partc_e_4_val_iter.eps',
         bbox_inches='tight')
plt.clf()
policy_alternate = value_function_to_policy(env, gamma, val_func_alternate)
print("h.")
print_policy(policy_alternate.reshape(4,4), actions_dict)

#make an agent take a walk in the environment
print("i. ")
gamma = 0.9
returns = take_a_walk(env, gamma, policy)
print(returns)