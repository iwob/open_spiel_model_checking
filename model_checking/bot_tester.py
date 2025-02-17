import re
import sys
from absl import app
from absl import flags
import numpy as np
import random
import time
import datetime
import textwrap
import logging

from action_selectors import *
from solvers import Solver, SolverMCMAS
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.bots import gtp
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel
from game_mnk import GameMnk, GameInterface
from game_nim import GameNim
import os
from dataclasses import *
from typing import Optional
from pathlib import Path
import shutil


_KNOWN_GAMES = ["mnk", "nim"]
_KNOWN_PLAYERS = [
    # A generic Monte Carlo Tree Search agent.
    "mcts",

    # A generic random agent.
    "random",

    # Learner based on creating a q-values table.
    "q-learner",

    # You'll be asked to provide the moves.GameTreeNode
    "human",

    # Run an external program that speaks the Go Text Protocol.
    # Requires the gtp_path flag.
    "gtp",

    # Run an alpha_zero checkpoint with MCTS. Uses the specified UCT/sims.
    # Requires the az_path flag.
    "az"
]
_KNOWN_SELECTORS = ["most2", "all", "k-best", "1-best", "2-best", "3-best", "4-best", "5-best", "10-best", "none"]


flags.DEFINE_enum("game", "mnk", _KNOWN_GAMES, help="Name of the game.")
flags.DEFINE_integer("m", 5, help="(Game: mnk) Width of the board (i.e., number of columns).")
flags.DEFINE_integer("n", 5, help="(Game: mnk) Height of the board (i.e., number of rows).")
flags.DEFINE_integer("k", 4, help="(Game: mnk) Number of elements forming a line to win.")
flags.DEFINE_string("piles", None, help="(Game: nim) Piles in the format as in the example: '1;3;5;7'.")
flags.DEFINE_string("formula", None, help="Formula to be verified. Player names and variables in the formula are problem-specific.")
flags.DEFINE_string("coalition", None, help="Player coalition provided as integers divided by commas, e.g. '1,2'.")
flags.DEFINE_string("initial_moves", None, help="Initial actions to be specified in the game-specific format.")
flags.DEFINE_enum("player1", "mcts", _KNOWN_PLAYERS, help="Who controls player 1.")
flags.DEFINE_enum("player2", "mcts", _KNOWN_PLAYERS, help="Who controls player 2.")
flags.DEFINE_enum("action_selector1", "most2", _KNOWN_SELECTORS, help="Action selector for the coalition. If action_selector2 is none, it will be also used for the anti-coalition.")
flags.DEFINE_enum("action_selector2", "none", _KNOWN_SELECTORS, help="Action selector for the anti-coalition.")
flags.DEFINE_string("gtp_path", None, help="Where to find a binary for gtp.")
flags.DEFINE_multi_string("gtp_cmd", [], help="GTP commands to run at init.")
flags.DEFINE_string("az_path", None, help="Path to an alpha_zero checkpoint. Needed by an az player.")
flags.DEFINE_string("mcmas_path", None, required=False, help="Path to the MCMAS executable.")
flags.DEFINE_string("output_file", None, required=False, help="Path to the file in which the results of this run will be stored.")
flags.DEFINE_string("submodels_dir", None, required=False, help="Path to the directory in which will be stored the generated submodels.")
flags.DEFINE_integer("uct_c", 1, help="UCT's exploration constant.")
flags.DEFINE_integer("rollout_count", 1, help="How many rollouts to do.")
flags.DEFINE_integer("max_simulations", 60000, help="How many simulations to run.")
flags.DEFINE_float("selector_epsilon", 0.95, required=False, help="Seed for the random number generator.")
flags.DEFINE_integer("selector_k", 3, required=False, help="How many best actions will be selected by selector.")
flags.DEFINE_integer("num_games", 1, help="How many games to play.")
flags.DEFINE_integer("seed", None, help="Seed for the random number generator.")
flags.DEFINE_integer("max_game_depth", 10, help="Maximum number of moves from the initial position that can be explored in the game tree.")
flags.DEFINE_bool("random_first", False, help="Play the first move randomly.")
flags.DEFINE_bool("solve", True, help="Whether to use MCTS-Solver.")
flags.DEFINE_bool("quiet", False, help="Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, help="Show the MCTS stats of possible moves.")
FLAGS = flags.FLAGS



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
    elif bot_type == "az":
        from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator
        from open_spiel.python.algorithms.alpha_zero import model as az_model
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
    elif bot_type == "random":
        return uniform_random.UniformRandomBot(player_id, rng)
    elif bot_type == "human":
        return human.HumanBot()
    elif bot_type == "gtp":
        bot = gtp.GTPBot(game, FLAGS.gtp_path)
        for cmd in FLAGS.gtp_cmd:
            bot.gtp_cmd(cmd)
        return bot
    raise ValueError("Invalid bot type: %s" % bot_type)


def _get_action_id(state, action_str):
    for action in state.legal_actions():
        if action_str == state.action_to_string(state.current_player(), action):
            return action
    return None


def _execute_initial_moves(state: pyspiel.State, bots: list, moves: list):
    for action_str in moves:
        action = _get_action_id(state, action_str)
        if action is None:
            sys.exit("Invalid action: {}".format(action_str))

        for i, bot in enumerate(bots):
            # According to docstrings: "This should not be called for the bot that generated the
            #  action as it already knows the action it took."
            if i != state.current_player():
                bot.inform_action(state, state.current_player(), action)
        state.apply_action(action)


def get_name(data_str: str) -> str:
    return data_str.split(": player:")[0]

my_policy_value_pattern_sims = re.compile(r",\s+sims:\s+([+-]?[0-9]+),")
my_policy_value_pattern_value = re.compile(r",\s+value:\s+([+-]?[0-9]+\.[0-9]+),")
def get_value(data_str: str) -> float:
    value = float(my_policy_value_pattern_value.search(data_str).group(1))
    return value

def _restart_bots(bots):
    for b in bots:
        b.restart()

def evaluate_bot(bots, test_set, game_utils, game, use_value):
    correct_guesses = 0
    action_selector = SelectKBestActions(k=1)
    for input, exp_outputs in test_set:
        initial_moves = game_utils.get_moves_from_history_str(input)
        state = game.new_initial_state()
        _restart_bots(bots)
        _execute_initial_moves(state, bots, initial_moves)
        print(f"Scenario:\n{state}")
        bot = bots[state.current_player()]

        if use_value:
            bot_selection = bot.step(state)
            # print(f"bot_selection = {state.action_to_string(state.current_player(), bot_selection)}")
            actions_list = [(get_value(line), get_name(line)) for line in bot.my_policy.split("\n")]
            actions_list = sorted(actions_list, reverse=True)  # sort by value
            _, action_name = action_selector(actions_list, state.current_player())[0]
            action_id = _get_action_id(state, action_name)
        else:
            action_id = bot.step(state)
            action_name = state.action_to_string(state.current_player(), action_id)
        if action_name in exp_outputs:
            print(f"Selected action: {action_name} (id: {action_id}) -- SUCCESS")
            correct_guesses += 1
        else:
            print(f"Selected action: {action_name} (id: {action_id}) -- FAIL")

    print(f"* Accuracy: {correct_guesses / len(test_set)}")


def main(argv):
    game_utils = GameMnk(3, 3, 3)
    game = game_utils.load_game()

    test_set = [
        ("", {"x(1,1)"}),
        ("x(0,1)", {"o(1,1)"}),
        ("x(2,2)", {"o(1,1)"}),
        ("x(1,1),o(1,0),x(0,2)", {"o(2,0)"}),
        ("x(1,1),o(1,2),x(0,2)", {"o(2,0)"}),
        ("x(0,0),o(1,0),x(2,0),o(0,1)", {"x(1,1)"}),
        ("x(1,1),o(2,1),x(0,1),o(1,2)", {"x(0,0)", "x(0,2)"}),
    ]
    bots_variants = [
        [_init_bot("mcts", game, 0), _init_bot("mcts", game, 1),]
    ]

    for bots in bots_variants:
        evaluate_bot(bots, test_set, game_utils, game, use_value=True)
        print("-"*40)
        evaluate_bot(bots, test_set, game_utils, game, use_value=False)



if __name__ == "__main__":
    app.run(main)