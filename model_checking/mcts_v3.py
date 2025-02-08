import re
import sys
from absl import app
from absl import flags
import numpy as np
import random
import time
import datetime
import textwrap

from action_selectors import *
from solvers import Solver, SolverMCMAS
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator
from open_spiel.python.algorithms.alpha_zero import model as az_model
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
import heapq
import shutil


@dataclass(init=True, order=True)
class NodeData:
    priority: int
    moves_str: str = field(compare=False)
    tree_root: dict = field(compare=False)
    _moves: list[str] = field(default=None, init=False, compare=False)
    game_state: pyspiel.State = field(default=None, init=False, compare=False)
    specification_path: str = field(default=None, init=False, compare=False)
    verification_results: dict = field(default=None, init=False, compare=False)

    def get_moves_list(self, game_utils: GameInterface):
        if self._moves is None:
            self._moves = game_utils.get_moves_from_history(self.moves_str)
        return self._moves

    def is_terminal_state(self):
        return self.game_state.is_terminal()

    def current_player(self):
        return self.game_state.current_player()


_KNOWN_GAMES = ["mnk", "nim"]

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

_KNOWN_SELECTORS = ["most2", "all", "k-best", "1-best", "2-best", "3-best", "4-best", "5-best", "10-best", "none"]

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


flags.DEFINE_boolean("encode_tree_in_spec", False, help="If true, then only a single specification file will be generated.")
flags.DEFINE_enum("game", "mnk", _KNOWN_GAMES, help="Name of the game.")
flags.DEFINE_integer("m", 5, help="(Game: mnk) Number of rows.")
flags.DEFINE_integer("n", 5, help="(Game: mnk) Number of columns.")
flags.DEFINE_integer("k", 4, help="(Game: mnk) Number of elements forming a line to win.")
flags.DEFINE_string("piles", None, help="(Game: nim) Piles in the format as in the example: '1;3;5;7'.")
flags.DEFINE_string("formula", None, help="Formula to be verified. Player names and variables in the formula are problem-specific.")
flags.DEFINE_string("coalition", None, help="Player coalition provided as integers divided by commas, e.g. '1,2'.")
flags.DEFINE_string("initial_moves", None, help="Initial actions to be specified in the game-specific format.")
flags.DEFINE_enum("player1", "mcts", _KNOWN_PLAYERS, help="Who controls player 1.")
flags.DEFINE_enum("player2", "mcts", _KNOWN_PLAYERS, help="Who controls player 2.")  # IB: oryginalnie było random
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

GAME_TREE_LEAF_ID = "SUBMODEL"
GAME_TREE_CUR_PLAYER_ID = "CUR_PLAYER"
RESERVED_TREE_IDS = {GAME_TREE_LEAF_ID, GAME_TREE_CUR_PLAYER_ID}

my_policy_value_pattern = re.compile(r",\s+value:\s+([+-]?[0-9]+\.[0-9]+),")

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
        # _opt_print("legal: ", state.action_to_string(state.current_player(), action))
        if action_str == state.action_to_string(state.current_player(), action):
            return action
    return None


def _restart_bots(bots):
    for b in bots:
        b.restart()


def _execute_initial_moves(state, bots, moves):
    for action_str in moves:
        action = _get_action(state, action_str)
        if action is None:
            sys.exit("Invalid action: {}".format(action_str))

        for bot in bots:
            bot.inform_action(state, state.current_player(), action)
        state.apply_action(action)


def move_down_tree(game_tree: dict, moves: list):
    """Moves down the tree creating nodes as necessary and returning a resulting node."""
    for m in moves:
        if m not in game_tree:
            game_tree[m] = {}
        game_tree = game_tree[m]
    return game_tree


def add_data_to_game_tree(game_tree: dict, moves: list, key: str, data):
    """Traverses the game tree to add some data under the specified key to a certain node."""
    game_tree = move_down_tree(game_tree, moves)
    game_tree[key] = data
    return game_tree


def _add_node_to_game_tree(game_utils, game_tree, node, key=GAME_TREE_LEAF_ID):
    """Traverses the game tree to add a leaf node (submodel)."""
    return add_data_to_game_tree(game_tree, node.get_moves_list(game_utils), key, node)


def _play_game_single_step(game_utils: GameInterface, state: pyspiel.State, bots, action_selector,
                           node: NodeData, nodes_queue, game_tree, coalition, max_game_depth):
    """Plays one game."""
    ra = 0  # indicates if bots are random, used in the commented code in the game loop

    # Add information about current player in the node
    add_data_to_game_tree(node.tree_root, [], GAME_TREE_CUR_PLAYER_ID, state.current_player())

    if state.is_terminal():
        node.game_state = state
        _add_node_to_game_tree(game_utils, game_tree, node)
        return

    if node.priority >= max_game_depth:
        # Too long sequence of moves, stop processing this sequence (it will constitute one
        # of the subproblem leaves in the game tree).
        node.game_state = state
        _add_node_to_game_tree(game_utils, game_tree, node)
        return

    # The purpose of this loop is to potentially automatically pass over chance nodes, but
    # this functionality is not currently implemented. The loop will always execute only once.
    while not state.is_terminal():
        # The state can be three different types: chance node, simultaneous node, or decision node
        if state.is_chance_node():
            raise ValueError("Game cannot have chance nodes.")
        elif state.is_simultaneous_node():
            raise ValueError("Game cannot have simultaneous nodes.")
        else:
            # Decision node: sample action for the single current player
            # print("dx")
            current_player = state.current_player()
            bot = bots[current_player]
            action = bot.step(state)  # for MCTS step() runs a given number of MCTS simulations (by default here: 60000)
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

            # IB ---------
            # p = bot.get_policy(state)  #return_probs=True
            # print("policy:", p)
            # ------------

            # print(bot.my_policy.split("\n"))
            # my_list = parse_and_sort(bot.my_policy.split("\n"))
            # print(my_list)
            # Example content of my_policy:
            # x(1,1): player: 0, prior: 0.043, value:  0.405, sims: 45770, outcome: none,  22 children
            # x(2,3): player: 0, prior: 0.043, value:  0.278, sims:  2910, outcome: none,  22 children
            # x(3,3): player: 0, prior: 0.043, value:  0.278, sims:  2875, outcome: none,  22 children
            # ...
            # x(0,2): player: 0, prior: 0.043, value: -0.085, sims:    59, outcome: none,  22 children
            # x(0,3): player: 0, prior: 0.043, value: -0.194, sims:    36, outcome: none,  22 children
            # x(3,4): player: 0, prior: 0.043, value: -0.364, sims:    22, outcome: none,  22 children
            # It is sorted by value, so we take the two highest values

            def get_name(data_str: str) -> str:
                return data_str.split(": player:")[0]
            def get_value(data_str: str) -> float:
                value = float(my_policy_value_pattern.search(data_str).group(1))
                return value

            # Here we use a custom my_policy field, which is a custom modification of OpenSpiel which
            # doesn't give that information on the outside
            actions_list = [(get_value(line), get_name(line)) for line in bot.my_policy.split("\n")]
            actions_list = sorted(actions_list, reverse=True)  # sort by value, by default values are sorted by prior

            actions_to_add = action_selector(actions_list, state.current_player(), coalition)

            for val, a in actions_to_add:
                n0 = NodeData(node.priority + 1,
                              moves_str=game_utils.add_move_to_history(node.moves_str, a),
                              tree_root=move_down_tree(node.tree_root, [a]))
                heapq.heappush(nodes_queue, n0)
            return


def generate_game_tree(game_utils: GameInterface, game: pyspiel.Game, bots, action_selector, coalition,
                       initial_moves: str = None, max_game_depth: int = 5):
    if initial_moves is None:
        initial_moves = ""
    game_tree = {}
    nodes_queue = [NodeData(0, moves_str=initial_moves, tree_root=game_tree)]
    while len(nodes_queue) > 0:
        _opt_print("current nodes_queue:")
        _opt_print("\n".join([f"\"{x}\"" for x in nodes_queue]) + "\n")

        node = heapq.heappop(nodes_queue)

        # State of the game is saved in the node, however information received by the bots is not.
        # To ensure that there will be no errors for various types of games, we create a new state
        # and manually simulate all the moves and inform players about them.
        state = game.new_initial_state()
        _restart_bots(bots)
        _execute_initial_moves(state, bots, game_utils.get_moves_from_history(node.moves_str))
        _opt_print(f"State after initial moves:\n{state}\n")

        _play_game_single_step(game_utils, state, bots, action_selector, node, nodes_queue, game_tree, coalition, max_game_depth=max_game_depth)
    return game_tree


def generate_specification(game_utils: GameInterface, node: NodeData, formula: str):
    return game_utils.formal_subproblem_description(node.game_state, history=node.moves_str, formulae_to_check=formula)


def _save_specification(game_utils: GameInterface, node: NodeData, output_dir, formula, cur_index=0):
    filename = f"{game_utils.get_name()}_s{cur_index}_{textwrap.shorten(node.moves_str, width=55)}.ispl"
    output_file = os.path.join(output_dir, filename)
    node.specification_path = output_file
    script = generate_specification(game_utils, node, formula)
    with open(output_file, "w") as f:
        f.write(script)
    return cur_index + 1


def save_specifications(game_utils: GameInterface, game_tree: dict, output_dir, formula: str,
                        exclude_terminal_states=False, cur_index=0):
    if GAME_TREE_LEAF_ID in game_tree:
        if exclude_terminal_states and game_tree[GAME_TREE_LEAF_ID].is_terminal_state():
            return cur_index
        return _save_specification(game_utils, game_tree[GAME_TREE_LEAF_ID], output_dir, formula, cur_index=cur_index)
    else:
        for a in game_tree:
            if a not in RESERVED_TREE_IDS:
                cur_index = save_specifications(game_utils, game_tree[a], output_dir, formula,
                                                cur_index=cur_index,
                                                exclude_terminal_states=exclude_terminal_states)
        return cur_index


def verify_submodel_node_solver(node: NodeData, solver: Solver) -> NodeData:
    if node.specification_path is None:
        raise Exception("Specification path not present in the leaf! Most likely the 'save_specifications' function needs to be used on the game tree.")
    _, meta = solver.verify_from_file(node.specification_path)
    node.verification_results = meta
    return node


def verify_submodel_node_rewards(node: NodeData, coalition) -> NodeData:
    """Uses expert knowledge that if the sum of rewards of coalition players is greater than the
     anti-coalition players, then the formula will be satisfied."""
    rewards = node.game_state.rewards()
    sum_coalition = [rewards[i] for i in node.game_state.game.num_players() if i in coalition]
    sum_anti_coalition = [rewards[i] for i in node.game_state.game.num_players() if i not in coalition]
    if sum_coalition > sum_anti_coalition:
        node.verification_results = {"decision": 1, "status": "auto"}
    else:
        node.verification_results = {"decision": 0, "status": "auto"}
    return node


def verify_submodel_node(node: NodeData, solver: Solver, coalition, exclude_terminal_states=False):
    if exclude_terminal_states and node.is_terminal_state():
        # Check if the sum of coalition rewards is bigger than anti-coalition
        return verify_submodel_node_rewards(node, coalition)
    else:
        # Perform verification using a solver
        return verify_submodel_node_solver(node, solver)

def verify_submodels(game_tree: dict, solver: Solver, coalition, exclude_terminal_states=False):
    """Traverses the game tree and adds at each submodel leaf the result of verification."""
    if GAME_TREE_LEAF_ID in game_tree:
        node = game_tree[GAME_TREE_LEAF_ID]
        verify_submodel_node(node, solver, coalition, exclude_terminal_states=exclude_terminal_states)
    else:
        for a in game_tree:
            if a not in RESERVED_TREE_IDS:
                verify_submodels(game_tree[a], solver, coalition, exclude_terminal_states=exclude_terminal_states)


def minimax_submodels_aggregation(game_tree, coalition, solve_submodels=False, solver: Solver=None,
                                  exclude_terminal_states=False):
    """Returns the overall result of the formula verification by using the minimax algorithm, where
     coalition wants to satisfy the formula and anti-coalition wants to falsify it."""
    if GAME_TREE_LEAF_ID in game_tree:
        # If it's a leaf node, propagate the result upwards
        node = game_tree[GAME_TREE_LEAF_ID]
        if solve_submodels:
            if solver is None:
                raise Exception("In order to solve submodels in minimax you need to pass solver as an argument.")
            verify_submodel_node(node, solver, coalition, exclude_terminal_states=exclude_terminal_states)
        return node.verification_results['decision']

    if GAME_TREE_CUR_PLAYER_ID not in game_tree:
        raise Exception("Reached a node in with unknown player! Aborting minimax.")
    cur_player = game_tree[GAME_TREE_CUR_PLAYER_ID]

    if cur_player in coalition:
        best_value = -float('inf')
        for key in game_tree:
            if key in RESERVED_TREE_IDS:
                continue
            value = minimax_submodels_aggregation(game_tree[key], coalition)
            best_value = max(best_value, value)
        return best_value
    else:
        best_value = float('inf')
        for key in game_tree:
            if key in RESERVED_TREE_IDS:
                continue
            value = minimax_submodels_aggregation(game_tree[key], coalition)
            best_value = min(best_value, value)
        return best_value


def MCSA_naive(game_utils: GameInterface, game: pyspiel.Game, solver: Solver, bots: list,
               action_selector: ActionSelector, formula: str, coalition: set,
               run_results_dir, results_dict, initial_moves: str=""):
    """MCSA = Model Checking by Submodel Aggregation. This is a naive version which works in stages and starts
    a new stage only after the previous one is finished:
    1. Generate game tree.
    2. Save specification files on disk.
    3. Verify all specification files.
    4. Use the min-max algorithm to compute aggregate answer for the whole game tree.
    """
    # Generate a game tree with submodels and terminal states as leaves
    start = time.time()
    print("Generating game tree...")
    game_tree = generate_game_tree(game_utils, game, bots, action_selector, coalition,
                                   initial_moves=initial_moves,
                                   max_game_depth=FLAGS.max_game_depth)
    end = time.time()
    results_dict["time_rl"] = end - start

    # Traverse tree and generate submodel specification files
    save_specifications(game_utils, game_tree, run_results_dir, formula)

    # Verify submodels
    start = time.time()
    print("Verifying submodels...")
    verify_submodels(game_tree, solver, coalition)
    end = time.time()
    results_dict["time_solver"] = end - start

    # Aggregate verification results
    result = minimax_submodels_aggregation(game_tree, coalition)

    return result, game_tree


def MCSA_single_solver_call(game_utils: GameInterface, game: pyspiel.Game, solver: Solver, bots: list,
                            action_selector: ActionSelector, formula: str, coalition: set,
                            run_results_dir, results_dict, initial_moves: str=""):
    """MCSA = Model Checking by Submodel Aggregation. In this MCSA variant game tree is directly translated into a
    single specification file, which is then solved by the solver. In effect, the only contribution of MCSA is
    reduction of the model by removing actions with high probability of success for their respective players.
    Stages of the algorithm:
    1. Generate game tree.
    2. Save a single specification file on disk corresponding to the game tree.
    3. Verify the specification file.
    """
    # Generate a game tree with submodels and terminal states as leaves
    start = time.time()
    print("Generating game tree...")
    game_tree = generate_game_tree(game_utils, game, bots, action_selector, coalition,
                                   initial_moves=initial_moves,
                                   max_game_depth=FLAGS.max_game_depth)
    end = time.time()
    results_dict["time_rl"] = end - start

    # Traverse tree and generate submodel specification files
    print("Verifying submodel...")
    script = game_utils.formal_subproblem_description_game_tree(game_tree, initial_moves, formula)
    script_path = os.path.join(run_results_dir, "game_tree_spec.ispl")
    with open(script_path, "w") as f:
        f.write(script)

    # Verify submodels
    start = time.time()
    result, meta = solver.verify_from_file(script_path)
    end = time.time()
    results_dict["time_solver"] = end - start

    return result, game_tree


def collect_game_tree_stats(game_tree, results_dict):
    if GAME_TREE_LEAF_ID in game_tree:
        node = game_tree[GAME_TREE_LEAF_ID]
        results_dict["num_submodels"] = 1 + results_dict.get("num_submodels", 0)
    else:
        for a in game_tree:
            if a not in RESERVED_TREE_IDS:
                collect_game_tree_stats(game_tree[a], results_dict)


def create_single_run_report(results_dict):
    timestamp = str(datetime.datetime.now())
    results_dict["timestamp"] = timestamp
    path = Path(results_dict["submodels_dir"]) / f"summary_{results_dict['name']}.txt"
    with path.open("w") as f:
        for k in sorted(results_dict.keys()):
            f.write(f"{k} = {results_dict[k]}\n")

def create_final_report(collected_results, output_file):
    timestamp = str(datetime.datetime.now())
    with output_file.open("w") as f:
        total_times = []
        solver_times = []
        rl_times = []
        num_result_one = 0
        num_result_zero = 0
        for i, d in enumerate(collected_results):
            if d["decision"]:
                num_result_one += 1
            else:
                num_result_zero += 1
            total_times.append(d["time_total"])
            solver_times.append(d["time_solver"])
            rl_times.append(d["time_rl"])
            f.write("# " + "-" * 30 + "\n")
            f.write(f"# Run {i}\n")
            f.write("# " + "-" * 30 + "\n")
            for k in sorted(d.keys()):
                f.write(f"{k} = {d[k]}\n")

        f.write("# " + "-" * 30 + "\n")
        f.write(f"# Total\n")
        f.write("# " + "-" * 30 + "\n")
        f.write(f"avg.time_rl = {sum(rl_times) / len(rl_times)}\n")
        f.write(f"avg.time_solver = {sum(solver_times) / len(solver_times)}\n")
        f.write(f"avg.time_total = {sum(total_times) / len(total_times)}\n")
        f.write(f"sum.result_0 = {num_result_zero}\n")
        f.write(f"sum.result_1 = {num_result_one}\n")
        f.write(f"timestamp = {timestamp}")


def main(argv):
    if FLAGS.mcmas_path is None:
        mcmas_path = "/home/iwob/Programs/MCMAS/mcmas-linux64-1.3.0"
    else:
        mcmas_path = FLAGS.mcmas_path
    solver = SolverMCMAS(mcmas_path, time_limit=1000*3600*4)

    if FLAGS.game == "mnk":
        # Example initial moves for mnk; "x(2,2),o(3,2)"
        if FLAGS.submodels_dir is None:
            results_root = Path(f"results(v3)__mnk_{FLAGS.m}_{FLAGS.n}_{FLAGS.k}")
        else:
            results_root = Path(FLAGS.submodels_dir)
        game_utils = GameMnk(FLAGS.m, FLAGS.n, FLAGS.k)
    elif FLAGS.game == "nim":
        if FLAGS.submodels_dir is None:
            results_root = Path(f"results(v3)__nim_{FLAGS.piles}")
        else:
            results_root = Path(FLAGS.submodels_dir)
        game_utils = GameNim(FLAGS.piles)
    else:
        raise Exception("Unknown game!")

    if FLAGS.formula is None and FLAGS.coalition is None:
        formula, coalition = game_utils.get_default_formula_and_coalition()
    elif FLAGS.formula is not None and FLAGS.coalition is not None:
        formula, coalition = FLAGS.formula, {int(a) for a in FLAGS.coalition.split(",")}
    else:
        raise Exception("Coalition and formula needs to be both specified (or left empty for the default values).")

    if FLAGS.output_file is None:
        output_file = results_root / "summary_all.txt"
    else:
        output_file = Path(FLAGS.output_file)
    if output_file.exists():
        output_file.unlink()

    if results_root.exists():
        shutil.rmtree(results_root)

    def get_action_selector(name):
        if name == "most2":
            return SelectAtMostTwoActions(epsilon_ratio=FLAGS.selector_epsilon)
        elif name == "all":
            return SelectAllActions()
        elif name == "k-best":
            return SelectKBestActions(k=FLAGS.selector_k)
        elif name == "1-best":
            return SelectKBestActions(k=1)
        elif name == "2-best":
            return SelectKBestActions(k=2)
        elif name == "3-best":
            return SelectKBestActions(k=3)
        elif name == "4-best":
            return SelectKBestActions(k=4)
        elif name == "5-best":
            return SelectKBestActions(k=5)
        elif name == "10-best":
            return SelectKBestActions(k=10)

    if FLAGS.action_selector1 == "none":
        raise Exception("action_selector1 cannot be empty.")
    if FLAGS.action_selector2 == "none":  # dual
        action_selector = get_action_selector(FLAGS.action_selector1)
    else:
        as1 = get_action_selector(FLAGS.action_selector1)
        as2 = get_action_selector(FLAGS.action_selector2)
        action_selector = DualActionSelector(as1, as2)

    final_log = ""
    collected_results = []
    collected_subproblem_dirs = []
    game = game_utils.load_game()
    if game.num_players() > 2:
        sys.exit("This game requires more players than the example can handle.")

    bots = [
        _init_bot(FLAGS.player1, game, 0),
        _init_bot(FLAGS.player2, game, 1),
    ]

    initial_moves = "" if FLAGS.initial_moves is None else FLAGS.initial_moves

    if FLAGS.encode_tree_in_spec:
        algorithm = MCSA_single_solver_call
    else:
        algorithm = MCSA_naive

    for i in range(FLAGS.num_games):
        start = time.time()
        run_results_dir = results_root / f"mcts_{i}"
        collected_subproblem_dirs.append(run_results_dir)
        run_results_dir.mkdir(parents=True, exist_ok=False)
        results_dict = {"submodels_dir": run_results_dir, "name": f"mcts_{i}"}

        result, game_tree = algorithm(game_utils, game, solver, bots, action_selector, formula, coalition,
                                      run_results_dir, results_dict, initial_moves)

        end = time.time()
        results_dict["decision"] = result
        results_dict["time_total"] = end - start
        results_dict["action_selector1"] = FLAGS.action_selector1
        results_dict["action_selector2"] = FLAGS.action_selector2
        if FLAGS.game == "nim":
            results_dict["piles"] = FLAGS.piles
        elif FLAGS.game == "mnk":
            results_dict["m,n,k"] = str((FLAGS.m, FLAGS.n, FLAGS.k))
        if FLAGS.action_selector1 == "k-best" or FLAGS.action_selector2 == "k-best":
            results_dict["selector_k"] = FLAGS.selector_k
        results_dict["max_game_depth"] = FLAGS.max_game_depth
        results_dict["max_simulations"] = FLAGS.max_simulations
        results_dict["encode_tree_in_spec"] = FLAGS.encode_tree_in_spec
        results_dict["game"] = FLAGS.game
        results_dict["player1"] = FLAGS.player1
        results_dict["player2"] = FLAGS.player2
        results_dict["formula"] = formula
        results_dict["coalition"] = coalition
        collect_game_tree_stats(game_tree, results_dict)
        create_single_run_report(results_dict)
        print("FINAL ANSWER:", result, f" (time:{end - start})")
        collected_results.append(results_dict)
        final_log += f"mcts ({run_results_dir}): {end - start}\n"

    print()
    print("-" * 25)
    print(final_log)

    create_final_report(collected_results, output_file)


if __name__ == "__main__":
    app.run(main)
