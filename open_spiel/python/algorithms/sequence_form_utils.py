# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Useful sequence form functions used in the MMD implementation."""

import numpy as np
from open_spiel.python import policy


_DELIMITER = " -=- "


def get_isa_key(info_state, action):
  return info_state + _DELIMITER + str(action)


def get_action_from_key(isa_key):
  _, action_str = isa_key.split(_DELIMITER)
  return int(action_str)


def get_infostate_from_key(isa_key, player):
  assert not is_root(isa_key, player), "Cannot use this method for root nodes."
  infostate, _ = isa_key.split(_DELIMITER)
  return infostate


def is_root(key, player):
  empty_is_key = f"***EMPTY_INFOSET_P{player}***"
  empty_isa_key = f"***EMPTY_INFOSET_ACTION_P{player}***"
  return True if key in [empty_is_key, empty_isa_key] else False


def construct_vars(game):
  """Construct useful sequence from variables from game.

  Args:
      game: The spiel game to solve (must be zero-sum, sequential, and have
        chance node of deterministic or explicit stochastic).

  Returns:
      An 8 tuple of sequence form variables from _construct_vars by
      recursively
      traversing the game tree.

  """

  initial_state = game.new_initial_state()
  npl = game.num_players()

  empty_is_keys = [f"***EMPTY_INFOSET_P{player}***" for player in range(npl)]
  empty_isa_keys = [
      f"***EMPTY_INFOSET_ACTION_P{player}***" for player in range(npl)
  ]

  # initialize variables
  infosets = [{empty_is_keys[p]: 0} for p in range(npl)]
  infoset_actions_to_seq = [{empty_isa_keys[p]: 0} for p in range(npl)]
  infoset_action_maps = [
      {empty_is_keys[p]: [empty_isa_keys[p]]} for p in range(npl)
  ]

  # infoset_action_maps = [{}, {}]
  payoff_dict = dict()

  infoset_parent_map = [{empty_isa_keys[p]: None} for p in range(npl)]
  infoset_actions_children = [{empty_isa_keys[p]: []} for p in range(npl)]

  _construct_vars(initial_state, infosets, infoset_actions_to_seq,
                  infoset_action_maps, infoset_parent_map, 1.0,
                  empty_is_keys[:], empty_isa_keys[:],
                  payoff_dict, infoset_actions_children)

  payoff_mat = _construct_numpy_vars(payoff_dict, infoset_actions_to_seq)
  return (infosets, infoset_actions_to_seq,
          infoset_action_maps, infoset_parent_map,
          payoff_mat, infoset_actions_children)


def uniform_random_seq(game, infoset_actions_to_seq):
  """Generate uniform random sequence.

  The sequence generated is equivalent to a uniform random tabular policy.

  Args:
      game: the spiel game to solve (must be zero-sum, sequential, and have
        chance mode of deterministic or explicit stochastic).
      infoset_actions_to_seq: a list of dicts, one per player, that maps a
        string of (infostate, action) pair to an id.

  Returns:
      A list of NumPy arrays, one for each player.
  """
  policies = policy.TabularPolicy(game)
  initial_state = game.new_initial_state()
  sequences = [
      np.ones(len(infoset_actions_to_seq[i])) for i in range(game.num_players())
  ]
  _policy_to_sequence(
      initial_state,
      policies,
      sequences,
      infoset_actions_to_seq,
      [1 for _ in range(game.num_players())],
  )
  return sequences


def _construct_vars(state, infosets, infoset_actions_to_seq,
                    infoset_action_maps, infoset_parent_map, chance_reach,
                    parent_is_keys, parent_isa_keys, payoff_dict,
                    infoset_actions_children):
  """Recursively builds maps and the sequence form payoff matrix.

  Args:
      state: pyspiel (OpenSpiel) state
      infosets: a list of dicts, one per player, that maps infostate to an id.
        The dicts are filled by this function and should initially only
        contain root values.
      infoset_actions_to_seq: a list of dicts, one per player, that maps a
        string of (infostate, action) pair to an id. The dicts are filled by
        this function and should initially only contain the root values.
      infoset_action_maps: a list of dicts, one per player, that maps each
        info_state to a list of (infostate, action) string.
      infoset_parent_map: a list of dicts, one per player, that maps each
        info_state to an (infostate, action) string.
      chance_reach: the contribution of chance's reach probability (should
        start at 1).
      parent_is_keys: a list of parent information state keys for this state
      parent_isa_keys: a list of parent (infostate, action) keys
      payoff_dict: a dict that maps sequences of players' (infostate, action)
        tuples, e.g., ((infostate, action), ...) to the chance weighted reward.
      infoset_actions_children: a list of dicts, one for each player, mapping
        (infostate, action) keys to reachable infostates for each player
  """

  if state.is_terminal():
    returns = state.returns()
    idx = tuple(parent_isa_keys_i for parent_isa_keys_i in parent_isa_keys)
    payoff_dict.setdefault(idx, 0)
    payoff_dict[idx] += np.asarray(returns) * chance_reach
    return

  if state.is_chance_node():
    for action, prob in state.chance_outcomes():
      new_state = state.child(action)
      _construct_vars(new_state, infosets, infoset_actions_to_seq,
                      infoset_action_maps, infoset_parent_map,
                      prob * chance_reach, parent_is_keys, parent_isa_keys,
                      payoff_dict, infoset_actions_children)
    return

  player = state.current_player()
  info_state = state.information_state_string(player)
  legal_actions = state.legal_actions(player)

  # Add to the infostate maps
  if info_state not in infosets[player]:
    infosets[player][info_state] = len(infosets[player])
  if info_state not in infoset_action_maps[player]:
    infoset_action_maps[player][info_state] = []

  # Add to infoset to parent infoset action map
  if info_state not in infoset_parent_map[player]:
    infoset_parent_map[player][info_state] = parent_isa_keys[player]

  # add as child to parent
  if parent_isa_keys[player] in infoset_actions_children[player]:
    if info_state not in infoset_actions_children[player][
        parent_isa_keys[player]]:
      infoset_actions_children[player][parent_isa_keys[player]].append(
          info_state)
  else:
    infoset_actions_children[player][parent_isa_keys[player]] = [info_state]

  new_parent_is_keys = parent_is_keys[:]
  new_parent_is_keys[player] = info_state

  for action in legal_actions:
    isa_key = get_isa_key(info_state, action)
    if isa_key not in infoset_actions_to_seq[player]:
      infoset_actions_to_seq[player][isa_key] = len(
          infoset_actions_to_seq[player])
    if isa_key not in infoset_action_maps[player][info_state]:
      infoset_action_maps[player][info_state].append(isa_key)

    new_parent_isa_keys = parent_isa_keys[:]
    new_parent_isa_keys[player] = isa_key
    new_state = state.child(action)
    _construct_vars(new_state, infosets, infoset_actions_to_seq,
                    infoset_action_maps, infoset_parent_map, chance_reach,
                    new_parent_is_keys, new_parent_isa_keys, payoff_dict,
                    infoset_actions_children)


def _construct_numpy_vars(payoff_dict, infoset_actions_to_seq):
  """Convert sequence form payoff dict to numpy array.

  Args:
      payoff_dict: a dict that maps sequences of players' (infostate, action)
        tuples, e.g., ((infostate, action), ...) to the chance weighted reward.
      infoset_actions_to_seq: a list of dicts, one per player, that maps a
        string of (infostate, action) pair to an id.

  Returns:
      A numpy array corresponding to the chance weighted rewards
      i.e. the sequence form payoff tensor.

  """
  npl = len(infoset_actions_to_seq)
  pls = range(npl)  # player list
  sequence_sizes = tuple(len(infoset_actions_to_seq[i]) for i in pls)
  payoff_tensor = np.zeros((npl,) + sequence_sizes)
  for player_isa_seqs, payoffs in payoff_dict.items():
    idx = tuple(infoset_actions_to_seq[i][player_isa_seqs[i]] for i in pls)
    payoff_tensor[(slice(None),) + idx] = np.asarray(payoffs)
  return payoff_tensor


def construct_constraint_vars(infoset_parent_map, infoset_actions_to_seq,
                              infoset_action_maps):
  """Construct useful sequence form variables from game.

  Args:
      infoset_parent_map: a list of dicts, one per player, that maps each
        info_state to an (infostate, action) string.
      infoset_actions_to_seq: a list of dicts, one per player, that maps a
        string of (infostate, action) pair to an id.
      infoset_action_maps: a list of dicts, one per player, that maps each
        info_state to a list of (infostate, action) string.

  Returns:
      A dict mapping player to a tuple containing a numpy array of coefficients,
      each of dimension # of player sequences, as well as a sparse vector
      containing the constants, i.e., dict[player] = (A, b) as in Ax = b.
  """
  npl = len(infoset_actions_to_seq)
  constraint_dict = {}

  for player in range(npl):
    num_seqs = len(infoset_actions_to_seq[player].values())

    root_con = np.zeros(num_seqs)
    root_con[0] = 1.0
    constraints = [root_con]

    for info_state in infoset_action_maps[player]:
      if is_root(info_state, player):
        continue

      parent_isa_key = infoset_parent_map[player][info_state]
      parent_seq_id = infoset_actions_to_seq[player][parent_isa_key]

      # seq ids for children
      children_isa_keys = infoset_action_maps[player][info_state]
      children_seq_ids = [
          infoset_actions_to_seq[player][isa_key]
          for isa_key in children_isa_keys
      ]

      constraint = np.zeros(num_seqs)
      constraint[parent_seq_id] = -1.0
      constraint[children_seq_ids] = 1.0
      constraints.append(constraint)

    constant = np.zeros(len(constraints))
    constant[0] = 1.0
    constraint_dict[player] = (np.stack(constraints), constant)

  return constraint_dict


def sequence_to_policy(sequences, game, infoset_actions_to_seq,
                       infoset_action_maps):
  """Convert sequence form policies to the realization-equivalent tabular ones.

  Args:
      sequences: list of two sequence form policies, one for each player.
      game: a spiel game with two players.
      infoset_actions_to_seq: a list of dicts, one per player, that maps a
        string of (infostate, action) pair to an id.
      infoset_action_maps: a list of dicts, one per player, that maps each
        info_state to a list of (infostate, action) string.

  Returns:
      A TabularPolicy object.
  """

  policies = policy.TabularPolicy(game)
  for player in range(2):
    for info_state in infoset_action_maps[player]:
      if is_root(info_state, player):
        continue

      state_policy = policies.policy_for_key(info_state)
      total_weight = 0
      num_actions = 0

      for isa_key in infoset_action_maps[player][info_state]:
        total_weight += sequences[player][infoset_actions_to_seq[player]
                                          [isa_key]]
        num_actions += 1

      unif_pr = 1.0 / num_actions
      for isa_key in infoset_action_maps[player][info_state]:
        rel_weight = sequences[player][infoset_actions_to_seq[player][isa_key]]
        _, action_str = isa_key.split(_DELIMITER)
        action = int(action_str)
        pr_action = rel_weight / total_weight if total_weight > 0 else unif_pr
        state_policy[action] = pr_action
  return policies


def policy_to_sequence(game, policies, infoset_actions_to_seq):
  """Converts a TabularPolicy object for a two-player game.

  The converted policy is its realization-equivalent sequence form one.

  Args:
      game: a two-player open spiel game.
      policies: a TabularPolicy object.
      infoset_actions_to_seq: a list of dicts, one per player, that maps a
        string of (infostate, action) pair to an id.

  Returns:
      A list of numpy arrays, one for each player.
  """
  initial_state = game.new_initial_state()
  sequences = [
      np.ones(len(infoset_actions_to_seq[0])),
      np.ones(len(infoset_actions_to_seq[1]))
  ]
  _policy_to_sequence(initial_state, policies, sequences,
                      infoset_actions_to_seq, [1, 1])
  return sequences


def _policy_to_sequence(state, policies, sequences, infoset_actions_to_seq,
                        parent_seq_val):
  """Converts a TabularPolicy object to its equivalent sequence form.

  This method modifies the sequences inplace and should not be called directly.

  Args:
      state: an openspiel state.
      policies: a TabularPolicy object.
      sequences: list of numpy arrays to be modified.
      infoset_actions_to_seq: a list of dicts, one per player, that maps a
        string of (infostate, action) pair to an id.
      parent_seq_val: list of parent sequence values, this method should be
        called with initial value of [1,1].
  """

  if state.is_terminal():
    return

  if state.is_chance_node():
    for action, _ in state.chance_outcomes():
      new_state = state.child(action)
      _policy_to_sequence(new_state, policies, sequences,
                          infoset_actions_to_seq, parent_seq_val)
    return

  player = state.current_player()
  info_state = state.information_state_string(player)
  legal_actions = state.legal_actions(player)
  state_policy = policies.policy_for_key(info_state)
  for action in legal_actions:
    isa_key = get_isa_key(info_state, action)
    # update sequence form
    sequences[player][infoset_actions_to_seq[player]
                      [isa_key]] = parent_seq_val[player] * state_policy[action]
    new_parent_seq_val = parent_seq_val[:]
    new_parent_seq_val[player] = sequences[player][
        infoset_actions_to_seq[player][isa_key]]
    new_state = state.child(action)
    _policy_to_sequence(new_state, policies, sequences, infoset_actions_to_seq,
                        new_parent_seq_val)
