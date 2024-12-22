# Copyright 2019 DeepMind Technologies Limited
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

"""MCTS example."""

import collections
import random
import sys
print(sys.path)

from absl import app
from absl import flags
import numpy as np
import random
import sys
import time
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator
from open_spiel.python.algorithms.alpha_zero import model as az_model
from open_spiel.python.bots import gtp
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel
import re
from mnk import make_whole_board
import io
from contextlib import redirect_stdout
import subprocess
import os
from dataclasses import dataclass
from typing import Optional
_KNOWN_PLAYERS = [
    # A generic Monte Carlo Tree Search agent.
    "mcts",

    # A generic random agent.
    "random",

    # You'll be asked to provide the moves.
    "human",

    # Run an external program that speaks the Go Text Protocol.
    # Requires the gtp_path flag.
    "gtp",

    # Run an alpha_zero checkpoint with MCTS. Uses the specified UCT/sims.
    # Requires the az_path flag.
    "az"
]
def parse_and_sort(data_list):
    result_list = []
    
    for data in data_list:
        # Wyciągamy ruch (x(0,0))
        move = data.split(': ')[0]
        
        # Dzielimy resztę na klucz-wartość
        items = data.split(': ')[1:]  # Pomijamy ruch
        keys = ['player', 'prior', 'value', 'sims', 'outcome', 'children']

        values = []
        for item in items:
            value = item.split(',')[0].strip()
            values.append(value)

        # Dodajemy liczbę dzieci jako ostatnią wartość
        values[-1] = values[-1].replace('children', '').strip()
        print(data)
        # Tworzymy słownik
        result_dict = dict(zip(keys, values))
        print(result_dict)
        # Dodajemy ruch do słownika
        result_dict['move'] = move
        
        # Przekształcamy wartości na odpowiednie typy
        result_dict['player'] = str(result_dict['player'])
        result_dict['prior'] = float(result_dict['prior'])
        result_dict['value'] = float(result_dict['value'])
        result_dict['sims'] = float(result_dict['sims'])
        result_dict['children'] = int(result_dict['children'])
        if result_dict['outcome'] == 'none':
            result_dict['outcome'] = None

        # Przenosimy ruch na początek słownika
        result_dict = {'move': result_dict.pop('move'), **result_dict}

        result_list.append(result_dict)

    # Sortowanie tablicy słowników malejąco po value
    result_list = sorted(result_list, key=lambda x: x['value'], reverse=True)
    
    return result_list
flags.DEFINE_string("game", "mnk", "Name of the game.")
flags.DEFINE_integer("m", 5, "Number of rows.")
flags.DEFINE_integer("n", 5, "Number of columns.")
flags.DEFINE_integer("k", 4, "Number of elements forming a line to win.")
flags.DEFINE_enum("player1", "mcts", _KNOWN_PLAYERS, "Who controls player 1.")
flags.DEFINE_enum("player2", "mcts", _KNOWN_PLAYERS, "Who controls player 2.")  # IB: oryginalnie było random
flags.DEFINE_string("gtp_path", None, "Where to find a binary for gtp.")
flags.DEFINE_multi_string("gtp_cmd", [], "GTP commands to run at init.")
flags.DEFINE_string("az_path", None,
                    "Path to an alpha_zero checkpoint. Needed by an az player.")
flags.DEFINE_integer("uct_c", 1, "UCT's exploration constant.")
flags.DEFINE_integer("rollout_count", 1, "How many rollouts to do.")
flags.DEFINE_integer("max_simulations",60000, "How many simulations to run.")
flags.DEFINE_integer("num_games", 1, "How many games to play.")
flags.DEFINE_integer("seed", None, "Seed for the random number generator.")
flags.DEFINE_bool("random_first", False, "Play the first move randomly.")
flags.DEFINE_bool("solve", True, "Whether to use MCTS-Solver.")
flags.DEFINE_bool("quiet", False, "Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, "Show the MCTS stats of possible moves.")

FLAGS = flags.FLAGS

q = ["x(2,2),o(3,2)"]
q_max_len = 10
flag_end = False
pattern = r'[xo]\(\d+,\d+\)'
def _opt_print(*args, **kwargs):
  if not FLAGS.quiet:
    print(*args, **kwargs)


def _init_bot(bot_type, game, player_id):
  """Initializes a bot by type."""
  rng = np.random.RandomState(FLAGS.seed)
  if bot_type == "mcts":
    evaluator = mcts.RandomRolloutEvaluator(FLAGS.rollout_count, rng)
    return mcts.MCTSBot(
        game,
        FLAGS.uct_c,
        FLAGS.max_simulations,
        evaluator,
        random_state=rng,
        solve=FLAGS.solve,
        verbose=False)
  if bot_type == "az":
    model = az_model.Model.from_checkpoint(FLAGS.az_path)
    evaluator = az_evaluator.AlphaZeroEvaluator(game, model)
    return mcts.MCTSBot(
        game,
        FLAGS.uct_c,
        FLAGS.max_simulations,
        evaluator,
        random_state=rng,
        child_selection_fn=mcts.SearchNode.puct_value,
        solve=FLAGS.solve,
        verbose=FLAGS.verbose)
  if bot_type == "random":
    return uniform_random.UniformRandomBot(player_id, rng)
  if bot_type == "human":
    return human.HumanBot()
  if bot_type == "gtp":
    bot = gtp.GTPBot(game, FLAGS.gtp_path)
    for cmd in FLAGS.gtp_cmd:
      bot.gtp_cmd(cmd)
    return bot
  raise ValueError("Invalid bot type: %s" % bot_type)


def _get_action(state, action_str):
  for action in state.legal_actions():
    print("legal: ", state.action_to_string(state.current_player(), action))
    if action_str == state.action_to_string(state.current_player(), action):
      return action
  return None
@dataclass
class NodeData:
    position: tuple[int, int]
    player: int
    prior: float
    value: float
    sims: int
    outcome: Optional[str]
    children: int 
def get_value(data_str: str) -> NodeData:
    value = float(data_str.split(':')[4].split(',')[0])
    return value
def _play_game(game, bots, initial_actions):
  """Plays one game."""
  ra=0
  state = game.new_initial_state()
  global q
  q = sorted(q, key=len)
  print(q[0])
  #print(q[0])
  #_opt_print("Initial state:\n{}".format(state))

  history = []

  if FLAGS.random_first:
    assert not initial_actions
    initial_actions = [state.action_to_string(
        state.current_player(), random.choice(state.legal_actions()))]
  if len(q) == 0:
    return 
  #print(q[0].count('x') + q[0].count('o'))
  if q[0].count('x') + q[0].count('o') >= q_max_len:
    
    flag_end = True
    return -1
  #print("q[0]:", q[0])
  moves = ruchy = re.findall(pattern, q[0])
  for action_str in moves:
    action = _get_action(state, action_str)
    if action is None:
      sys.exit("Invalid action: {}".format(action_str))

    history.append(action_str)
    for bot in bots:
      bot.inform_action(state, state.current_player(), action)
    state.apply_action(action)
    #_opt_print("Forced action", action_str)
    #_opt_print("Next state:\n{}".format(state))
  if state.is_terminal():
    print("jebło")
    q.append(q[0]+"                                       ")
    q.remove(q[0])
  while not state.is_terminal():
    current_player = state.current_player()
    # The state can be three different types: chance node,
    # simultaneous node, or decision node
    if state.is_chance_node():
      # Chance node: sample an outcome
      outcomes = state.chance_outcomes()
      num_actions = len(outcomes)
      _opt_print("Chance node, got " + str(num_actions) + " outcomes")
      #print(outcomes)
      action_list, prob_list = zip(*outcomes)
      action = np.random.choice(action_list, p=prob_list)
      action_str = state.action_to_string(current_player, action)
      _opt_print("Sampled action: ", action_str)
    elif state.is_simultaneous_node():
      raise ValueError("Game cannot have simultaneous nodes.")
    else:
      # Decision node: sample action for the single current player
      #print("dx")
      bot = bots[current_player]
      action = bot.step(state)
      # if ra==1:
      #   moves = state.legal_actions(current_player)
      #   if q[0]=="":
      #     if random.random() > 0.5:
      #       m1,m2 = random.sample(moves,2)
      #       m1 = state.action_to_string(current_player, m1)
      #       m2 = state.action_to_string(current_player, m2)
      #       q.append(m1)
      #       q.append(m2)
      #       q.remove(q[0])
      #     else:
      #       m1,m2 = random.sample(moves,2)
      #       m1 = state.action_to_string(current_player, m1)
      #       q.append(m1)
      #       q.remove(q[0])
      #   else:
      #     if random.random() > 0.5:
      #       m1,m2 = random.sample(moves,2)
      #       m1 = state.action_to_string(current_player, m1)
      #       m2 = state.action_to_string(current_player, m2)
      #       q.append(q[0] + "," +m1)
      #       q.append(q[0] + "," +m2)
      #       q.remove(q[0])
      #     else:
      #       m1,m2 = random.sample(moves,2)
      #       m1 = state.action_to_string(current_player, m1)
      #       m2 = state.action_to_string(current_player, m2)
      #       q.append(q[0] + "," +m1)
      #       q.remove(q[0])
      # if ra==1:
      #   return  



      actions = [field.split(":")[0] for field in bot.my_policy.split("\n")[:2]]
      print(q[0])
      # print(bot.my_policy.split("\n"))
      # my_list = parse_and_sort(bot.my_policy.split("\n"))
      # print(my_list)
      val1 = get_value( bot.my_policy.split("\n")[:2][0])
      val2= get_value( bot.my_policy.split("\n")[:2][1])
      #sys.exit()
      #print(q[0],q[0].count('x'), q[0].count('o') )
      #print(actions)
      print(val1, val2)
      if q[0]=="":
        if val1 !=0 and val2/val1 > 0.99:
          q.append(actions[0])
          q.append(actions[1])
          q.remove(q[0])
        else:
           q.append(actions[0])
           q.remove(q[0])
      else:
        if val1 !=0 and val2/val1 > 0.99:
          q.append(q[0] + "," + actions[0])
          q.append(q[0] + "," + actions[1])
          q.remove(q[0])
        else:
          q.append(q[0] + "," + actions[0])
          q.remove(q[0])
      

      return
      action_str = state.action_to_string(current_player, action)
      _opt_print("Player {} sampled action: {}".format(current_player,
                                                       action_str))

    for i, bot in enumerate(bots):
      if i != current_player:
        bot.inform_action(state, current_player, action)
    history.append(action_str)
    state.apply_action(action)

    _opt_print("Next state:\n{}".format(state))

  # Game is now done. Print return for each player
  returns = state.returns()
  # print("Returns:", " ".join(map(str, returns)), ", Game actions:",
  #       " ".join(history))

  for bot in bots:
    bot.restart()

  return returns, history


def main(argv):
  print("xd")
  for i in range(0,10):
    start = time.time()
    global q
    q = ["x(2,2),o(3,2)"]  # set of initial moves
    flag_end = False
    game = pyspiel.load_game(FLAGS.game, {"m": FLAGS.m, "n": FLAGS.n, "k": FLAGS.k})
    if game.num_players() > 2:
      sys.exit("This game requires more players than the example can handle.")
    bots = [
        _init_bot(FLAGS.player1, game, 0),
        _init_bot(FLAGS.player2, game, 1),
    ]
    while not(flag_end):
      
      if -1 ==_play_game(game, bots, argv[1:]):
        break

    counter = 0
    os.mkdir(f"/home/iwob/mcts_start__5_5_4{i}/")
    for move in q:
      
      output_file_path = f"/home/iwob/mcts_start__5_5_4{i}/" + f"output_{move.replace(' ', '')}.txt"
      counter += 1
      f = io.StringIO()
      with redirect_stdout(f):
        make_whole_board(5, 5, 4, im=move)
      with open(output_file_path, "w") as output_file:
          output_file.write(f.getvalue())
    end = time.time()
    print(f"i:{i},",end-start)

if __name__ == "__main__":
  app.run(main)
