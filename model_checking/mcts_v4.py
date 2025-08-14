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
from solvers import *
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms import ismcts
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.bots import gtp
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel
from game_mnk import GameMnk, GameInterface
from game_nim import GameNim
from game_kuhn_poker import GameKuhnPoker
from game_atl_model import GameInterfaceAtlModel
import os
from dataclasses import *
from typing import Optional
from pathlib import Path
import shutil


try:
    # Removing annoying default format of the absl logger
    # (https://stackoverflow.com/questions/59654893/python-absl-logging-without-timestamp-module-name)
    import absl.logging as abslogging
    abslogging.get_absl_handler().setFormatter(None)
except Exception:
    pass
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(init=True, order=True)
class GameTreeNode:
    value: float | None
    cur_player: int = field(init=True, compare=False)
    # A mapping from action name to the children nodes. Key: tuple (value, action_name)
    children: dict = field(default_factory=lambda: {}, init=True, compare=False)
    metadata: dict = field(default_factory=lambda: {}, init=True, compare=False)
    specification_path: str = field(default=None, init=False, compare=False)
    verification_results: dict = field(default=None, init=False, compare=False)
    is_leaf: bool = field(default=False, init=False, compare=False)
    is_terminal_state: bool = field(default=False, init=False, compare=False)

    def __getitem__(self, key):
        return self.children[key]

    def __setitem__(self, key: tuple, value):
        assert isinstance(value, GameTreeNode)
        self.children[key] = value

    def __iter__(self):
        yield from self.children.keys()


@dataclass(init=True, order=True)
class QueueNode:
    priority: int
    moves_str: str = field(compare=False)
    state: pyspiel.State = field(compare=False)
    _moves: list[str] = field(default=None, init=False, compare=False)

    def get_moves_list(self, game_utils: GameInterface):
        if self._moves is None:
            self._moves = game_utils.get_moves_from_history_str(self.moves_str)
        return self._moves

    def is_terminal_state(self):
        return self.state.is_terminal()

    def current_player(self):
        return self.state.current_player()



_KNOWN_GAMES = ["mnk", "nim", "kuhn_poker", "atl_model"]
_KNOWN_PLAYERS = [
    # A generic Monte Carlo Tree Search agent.
    "mcts",

    #Information Set Monte Carlo Tree Search (MCTS variant for imperfect information games)
    "ismcts",

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
_KNOWN_SELECTORS = ["most2", "all", "k-best", "1-best", "2-best", "3-best", "4-best", "5-best", "10-best", "none",
                    "0.25-total-value", "0.5-total-value", "0.75-total-value", "k-total-value"]
_KNOWN_SOLVERS = ["mcmas", "stv"]


flags.DEFINE_enum("game", "mnk", _KNOWN_GAMES, help="Name of the game.")
flags.DEFINE_integer("m", 5, help="(Game: mnk) Width of the board (i.e., number of columns).")
flags.DEFINE_integer("n", 5, help="(Game: mnk) Height of the board (i.e., number of rows).")
flags.DEFINE_integer("k", 4, help="(Game: mnk) Number of elements forming a line to win.")
flags.DEFINE_string("piles", None, help="(Game: nim) Piles in the format as in the example: '1;3;5;7'.")
flags.DEFINE_string("formula", None, help="Formula to be verified. Player names and variables in the formula are problem-specific.")
flags.DEFINE_string("coalition", None, help="Player coalition provided as integers divided by commas, e.g. '1,2'.")
flags.DEFINE_string("initial_moves", None, help="Initial actions to be specified in the game-specific format.")
flags.DEFINE_enum("player", "mcts", _KNOWN_PLAYERS, help="Who controls all agents. Is taken into account only if the player1 and player2 arguments are not provided.")
flags.DEFINE_enum("player1", None, _KNOWN_PLAYERS, help="Who controls player 1. Only applies to games with exactly 2 players.")
flags.DEFINE_enum("player2", None, _KNOWN_PLAYERS, help="Who controls player 2. Only applies to games with exactly 2 players.")
flags.DEFINE_enum("action_selector1", "most2", _KNOWN_SELECTORS, help="Action selector for the coalition. If action_selector2 is none, it will be also used for the anti-coalition.")
flags.DEFINE_enum("action_selector2", "none", _KNOWN_SELECTORS, help="Action selector for the anti-coalition.")
flags.DEFINE_enum("solver", "mcmas", _KNOWN_SOLVERS, help="Type of the solver to be used for verification.")
flags.DEFINE_string("atl_spec_path", None, help="Path to the ATL specification.")
flags.DEFINE_string("gtp_path", None, help="Where to find a binary for gtp.")
flags.DEFINE_multi_string("gtp_cmd", [], help="GTP commands to run at init.")
flags.DEFINE_string("az_path", None, help="Path to an alpha_zero checkpoint. Needed by an az player.")
flags.DEFINE_string("mcmas_path", None, required=False, help="Path to the MCMAS executable.")
flags.DEFINE_string("stv_path", None, required=False, help="Path to the STV executable.")
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
flags.DEFINE_bool("use_reward_in_terminal_states", False, help="If true, then a solver won't be called in terminal states. Instead, rewards will be summed between the players and if the reward of the coalition is greater than anti-coalition, then verification is assumed to be successful.")
flags.DEFINE_bool("random_first", False, help="Play the first move randomly.")
flags.DEFINE_bool("solve", True, help="Whether to use MCTS-Solver.")
flags.DEFINE_bool("quiet", False, help="Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, help="Show the MCTS stats of possible moves.")
FLAGS = flags.FLAGS

my_policy_value_pattern = re.compile(r",\s+sims:\s+([+-]?[0-9]+),")  # should guide much better
# my_policy_value_pattern = re.compile(r",\s+value:\s+([+-]?[0-9]+\.[0-9]+),")


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
    elif bot_type == "ismcts":
        evaluator = mcts.RandomRolloutEvaluator(FLAGS.rollout_count, rng)
        return ismcts.ISMCTSBot(
            game,
            evaluator,
            FLAGS.uct_c,
            FLAGS.max_simulations,
            random_state=rng,
            max_world_samples=ismcts.UNLIMITED_NUM_WORLD_SAMPLES,  # default
            final_policy_type=ismcts.ISMCTSFinalPolicyType.MAX_VISIT_COUNT,  # default
            use_observation_string=False,  # default
            allow_inconsistent_action_sets=False,  # default
            child_selection_policy=ismcts.ChildSelectionPolicy.PUCT)  # default
    elif bot_type == "q-learner":
        return tabular_qlearner.QLearner(player_id=player_id, num_actions=num_actions)
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
    raise Exception(f"No id was found for the action: '{action_str}'.")


def _restart_bots(bots: list):
    for b in bots:
        b.restart()


def _inform_bots(bots: list, state: pyspiel.State, action: int):
    for i, bot in enumerate(bots):
        # According to docstrings: "This should not be called for the bot that generated the
        #  action as it already knows the action it took."
        if i != state.current_player():
            bot.inform_action(state, state.current_player(), action)


def _execute_initial_moves(state: pyspiel.State, bots: list, moves: list):
    for action_str in moves:
        action = _get_action_id(state, action_str)
        if action is None:
            sys.exit("Invalid action: {}".format(action_str))
        # OpenSpiel docs: "The state is the state at which the `player_id` player decided to take
        #  the given `action` (but before it is applied to the state)."
        _inform_bots(bots, state, action)
        state.apply_action(action)


def verify_submodel_node_solver(game_tree: GameTreeNode, solver: Solver):
    if game_tree.specification_path is None:
        raise Exception("Specification path not present in the leaf!")
    dec, meta = solver.verify_from_file(game_tree.specification_path)
    game_tree.verification_results = meta
    return dec


def verify_submodel_node_rewards(node: QueueNode, game_tree: GameTreeNode, coalition) -> QueueNode:
    """Uses expert knowledge that if the sum of rewards of coalition players is greater than the
     anti-coalition players, then the formula will be satisfied."""
    rewards = node.state.rewards()
    sum_coalition = [rewards[i] for i in range(node.state.get_game().num_players()) if i in coalition]
    sum_anti_coalition = [rewards[i] for i in range(node.state.get_game().num_players()) if i not in coalition]
    if sum_coalition > sum_anti_coalition:
        game_tree.verification_results = {"decision": 1, "status": "auto"}
    else:
        game_tree.verification_results = {"decision": 0, "status": "auto"}
    return game_tree.verification_results["decision"]


def verify_submodel_node(node: QueueNode, game_tree: GameTreeNode, solver: Solver, coalition,
                         results_dict, use_reward_in_terminal_states=True):
    if use_reward_in_terminal_states and node.is_terminal_state():
        # Check if the sum of coalition rewards is bigger than anti-coalition
        return verify_submodel_node_rewards(node, game_tree, coalition)
    else:
        # Perform verification using a solver
        start = time.time()
        dec = verify_submodel_node_solver(game_tree, solver)
        end = time.time()
        results_dict["time_solver"] += end - start
        return dec


def generate_specification(game_utils: GameInterface, node: QueueNode, formula: str):
    return game_utils.formal_subproblem_description(node.state, history=node.moves_str, formulae_to_check=formula)


def get_filename(game_utils, node, num):
    if len(node.moves_str) <= 55:
        moves_str = node.moves_str
    else:
        moves_str = node.moves_str[:55] + "[...]"
    return f"{game_utils.get_name()}_s{num}_{moves_str}.ispl"

SPEC_FILE_COUNTER = 0
def save_specification_file(game_utils: GameInterface, node, game_tree, formula,
                            run_results_dir, use_reward_in_terminal_states=False):
    if not game_tree.is_terminal_state or (game_tree.is_terminal_state and not use_reward_in_terminal_states):
        global SPEC_FILE_COUNTER
        filename = get_filename(game_utils, node, SPEC_FILE_COUNTER)
        SPEC_FILE_COUNTER += 1
        script = generate_specification(game_utils, node, formula)
        game_tree.specification_path = os.path.join(run_results_dir, filename)
        with open(game_tree.specification_path, "w") as f:
            f.write(script)


def _debug_player_name(node: QueueNode):
    if node.state.current_player() == -1:
        return "chance"
    elif node.state.current_player() == -4:
        return "leaf"
    else:
        return str(node.state.current_player())


def MCSA_combined_run(game_utils: GameInterface, solver: Solver,
                      bots: list, action_selector: ActionSelector, formula: str, coalition: set,
                      node: QueueNode, game_tree: GameTreeNode, run_results_dir, results_dict,
                      max_game_depth, use_reward_in_terminal_states, unroll_chance_nodes):
    debug_indent = "\t" * node.priority
    logger.debug(f"{debug_indent}(Player: {_debug_player_name(node)}) Processing state:\n{textwrap.indent(str(node.state), debug_indent)}")

    if node.state.is_terminal() or node.priority >= max_game_depth:
        # Too long sequence of moves or a terminal state, stop processing this sequence and verify it.
        game_tree.is_leaf = True
        game_tree.is_terminal_state = node.state.is_terminal()
        save_specification_file(game_utils, node, game_tree, formula,
                                run_results_dir=run_results_dir,
                                use_reward_in_terminal_states=use_reward_in_terminal_states)
        verify_submodel_node(node, game_tree, solver, coalition, results_dict=results_dict, use_reward_in_terminal_states=use_reward_in_terminal_states)
        if game_tree.is_terminal_state and use_reward_in_terminal_states:
            logger.debug(f"{debug_indent}(Player: {_debug_player_name(node)}) Leaf state; verification: {game_tree.verification_results['decision']} (based on rewards: {node.state.rewards()})")
        else:
            logger.debug(f"{debug_indent}(Player: {_debug_player_name(node)}) Leaf state; verification: {game_tree.verification_results['decision']}")
        return game_tree.verification_results["decision"]


    # The state can be of three different types: chance node, simultaneous node, or decision node
    if node.state.is_chance_node():
        if unroll_chance_nodes:
            for action in node.state.legal_actions():
                action_str = node.state.action_to_string(node.state.current_player(), action)
                new_state = node.state.clone()
                # TODO: clone() method not implemented
                # new_bots = [b.clone() for b in bots]  # Probably not needed for the perfect information, but may be needed for the imperfect case
                _inform_bots(bots, node.state, action)
                new_state.apply_action(action)
                game_tree[action_str] = GameTreeNode(None, cur_player=new_state.current_player())
                new_node = QueueNode(node.priority + 1,
                                     moves_str=game_utils.add_move_to_history_str(node.moves_str, action_str),
                                     state=new_state)
                logger.debug(f"{debug_indent}(Player: {_debug_player_name(node)}) Exploring new action: {action}")
                dec = MCSA_combined_run(game_utils, solver, bots, action_selector, formula, coalition,
                                        new_node, game_tree[action_str],
                                        run_results_dir=run_results_dir,
                                        results_dict=results_dict,
                                        max_game_depth=max_game_depth,
                                        use_reward_in_terminal_states=use_reward_in_terminal_states,
                                        unroll_chance_nodes=unroll_chance_nodes)
                if not dec:
                    logger.debug(f"{debug_indent}(Player: {_debug_player_name(node)}) Opponent has a path to prevent coalition from winning, move to the previous layer")
                    return 0
                else:
                    logger.debug(f"{debug_indent}(Player: {_debug_player_name(node)}) Opponent continues search")
            return 1  # chance node acts as part of anti-coalition
        else:
            raise ValueError("Game cannot have chance nodes.")
    elif node.state.is_simultaneous_node():
        raise ValueError("Game cannot have simultaneous nodes.")
    else:
        current_player = node.state.current_player()
        bot = bots[current_player]
        action = bot.step(node.state)  # for MCTS step() runs a given number of MCTS simulations (by default here: 60000)

        # Example content of my_policy:
        # x(1,1): player: 0, prior: 0.043, value:  0.405, sims: 45770, outcome: none,  22 children
        # x(2,3): player: 0, prior: 0.043, value:  0.278, sims:  2910, outcome: none,  22 children
        # ...
        # x(0,3): player: 0, prior: 0.043, value: -0.194, sims:    36, outcome: none,  22 children
        # x(3,4): player: 0, prior: 0.043, value: -0.364, sims:    22, outcome: none,  22 children
        # It is sorted by value, so we take the two highest values

        def get_name(data_str: str) -> str:
            return data_str.split(": player:")[0].strip()
        def get_value(data_str: str) -> float:
            value = float(my_policy_value_pattern.search(data_str).group(1))
            return value

        # Here we use a custom my_policy field, which is a custom modification of OpenSpiel which
        # doesn't give that information on the outside
        actions_list = [(get_value(line), get_name(line)) for line in bot.my_policy.split("\n")]
        actions_list = sorted(actions_list, reverse=True)  # sort by value
        actions_to_explore = action_selector(actions_list, current_player, coalition)
        # Assumption: actions_to_explore are returned sorted by the action_selector

        for val, a in actions_to_explore:
            action_id = _get_action_id(node.state, a)
            new_state = node.state.clone()
            # TODO: clone() method not implemented
            # new_bots = [b.clone() for b in bots]  # Probably not needed for the perfect information, but may be needed for the imperfect case
            new_bots = bots
            _inform_bots(new_bots, node.state, action_id)
            new_state.apply_action(action_id)
            game_tree[a] = GameTreeNode(val, cur_player=new_state.current_player())
            new_node = QueueNode(node.priority + 1,
                                 moves_str=game_utils.add_move_to_history_str(node.moves_str, a),
                                 state=new_state)
            logger.debug(f"{debug_indent}(Player: {_debug_player_name(node)}) Exploring new action: {(val, a)}")
            dec = MCSA_combined_run(game_utils, solver, new_bots, action_selector, formula, coalition,
                                    new_node, game_tree[a],
                                    run_results_dir=run_results_dir,
                                    results_dict=results_dict,
                                    max_game_depth=max_game_depth,
                                    use_reward_in_terminal_states=use_reward_in_terminal_states,
                                    unroll_chance_nodes=unroll_chance_nodes)

            # Here we implement a minmax part depending on the decision returned by the lower layers
            if current_player in coalition:
                if dec:
                    logger.debug(f"{debug_indent}(Player: {_debug_player_name(node)}) Proponent has a winning path, move to the previous layer")
                    return 1
                else:
                    logger.debug(f"{debug_indent}(Player: {_debug_player_name(node)}) Proponent continues search")
            else:
                if not dec:
                    logger.debug(f"{debug_indent}(Player: {_debug_player_name(node)}) Opponent has a path to prevent coalition from winning, move to the previous layer")
                    return 0
                else:
                    logger.debug(f"{debug_indent}(Player: {_debug_player_name(node)}) Opponent continues search")
        approx = f"(full: {len(actions_to_explore)}/{len(actions_list)})" if len(actions_to_explore) == len(actions_list) else f"(approx: {len(actions_to_explore)}/{len(actions_list)})"
        if current_player in coalition:
            logger.debug(f"{debug_indent}(Player: {_debug_player_name(node)}) Proponent fails after exploring available actions; verification {approx}: False")
            return 0  # didn't manage to find a winning path
        else:
            logger.debug(f"{debug_indent}(Player: {_debug_player_name(node)}) Opponent fails after exploring available actions; verification {approx}: True")
            return 1  # didn't manage to find a not-winning path for proponents


def MCSA_combined(game_utils: GameInterface, game: pyspiel.Game, solver: Solver, bots: list,
                  action_selector: ActionSelector, formula: str, coalition: set, run_results_dir,
                  results_dict, initial_moves: str="", max_game_depth=5, use_reward_in_terminal_states=False,
                  unroll_chance_nodes=True):
    global SPEC_FILE_COUNTER
    SPEC_FILE_COUNTER = 0
    results_dict["time_solver"] = 0.0
    if isinstance(initial_moves, str):
        initial_moves = game_utils.get_moves_from_history_str(initial_moves)
    state = game.new_initial_state()
    _restart_bots(bots)
    _execute_initial_moves(state, bots, initial_moves)

    # Initial nodes are not part of the game tree and are not taken into account when doing minmax.
    # The reason for that is that the initial moves form a single path without any branching, and the
    # verification result can be back-propagated directly (other perspective: a different game with
    # the same rules but a different starting position).
    game_tree = GameTreeNode(0, state.current_player())
    init_node = QueueNode(len(initial_moves), ",".join(initial_moves), state)

    dec = MCSA_combined_run(game_utils, solver, bots, action_selector, formula, coalition,
                            init_node, game_tree,
                            run_results_dir=run_results_dir,
                            results_dict=results_dict,
                            max_game_depth=max_game_depth,
                            use_reward_in_terminal_states=use_reward_in_terminal_states,
                            unroll_chance_nodes=unroll_chance_nodes)
    return dec, game_tree


def collect_game_tree_stats(game_tree: GameTreeNode, results_dict):
    if game_tree.is_leaf and game_tree.verification_results["status"] != "auto":
        results_dict["num_submodels"] += 1
    else:
        for a in game_tree:
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
        num_submodels = []
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
            num_submodels.append(d["num_submodels"])
            f.write("# " + "-" * 30 + "\n")
            f.write(f"# Run {i}\n")
            f.write("# " + "-" * 30 + "\n")
            for k in sorted(d.keys()):
                f.write(f"{k} = {d[k]}\n")

        f.write("# " + "-" * 30 + "\n")
        f.write(f"# Total\n")
        f.write("# " + "-" * 30 + "\n")
        f.write(f"avg.num_submodels = {sum(num_submodels) / len(num_submodels)}\n")
        f.write(f"avg.time_rl = {sum(rl_times) / len(rl_times)}\n")
        f.write(f"avg.time_solver = {sum(solver_times) / len(solver_times)}\n")
        f.write(f"avg.time_total = {sum(total_times) / len(total_times)}\n")
        f.write(f"stddev.num_submodels = {np.std(num_submodels)}\n")
        f.write(f"stddev.time_rl = {np.std(rl_times)}\n")
        f.write(f"stddev.time_solver = {np.std(solver_times)}\n")
        f.write(f"stddev.time_total = {np.std(total_times)}\n")
        f.write(f"sum.result_0 = {num_result_zero}\n")
        f.write(f"sum.result_1 = {num_result_one}\n")
        f.write(f"timestamp = {timestamp}")


def main(argv):
    if FLAGS.mcmas_path is None:
        mcmas_path = "/home/iwob/Programs/MCMAS/mcmas-linux64-1.3.0"
    else:
        mcmas_path = FLAGS.mcmas_path
    if FLAGS.stv_path is None:
        stv_path = "/home/iwob/Programs/STV/stv"
    else:
        stv_path = FLAGS.stv_path

    if FLAGS.solver == "mcmas":
        solver = SolverMCMAS(mcmas_path, time_limit=1000*3600*4)
    elif FLAGS.solver == "stv":
        solver = SolverSTV(stv_path, time_limit=1000 * 3600 * 4)

    if FLAGS.game == "mnk":
        # Example initial moves for mnk; "x(2,2),o(3,2)"
        game_utils = GameMnk(FLAGS.m, FLAGS.n, FLAGS.k)
    elif FLAGS.game == "nim":
        game_utils = GameNim(FLAGS.piles)
    elif FLAGS.game == "kuhn_poker":
        game_utils = GameKuhnPoker()
    elif FLAGS.game == "atl_model":
        game_utils = GameInterfaceAtlModel(FLAGS.atl_spec_path)
    else:
        raise Exception("Unknown game!")

    if FLAGS.submodels_dir is None:
        results_root = Path(f"results(v4)__{game_utils.get_name()}")  # TODO: change name to constant and independent
    else:
        results_root = Path(FLAGS.submodels_dir)

    if FLAGS.formula is None and FLAGS.coalition is None:
        if FLAGS.game == "atl_model":
            formula, coalition = game_utils.formula, game_utils.coalition
        else:
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
        elif name == "0.5-total-value":
            return SelectTotalValuePercent(0.5)
        elif name == "0.25-total-value":
            return SelectTotalValuePercent(0.25)
        elif name == "0.75-total-value":
            return SelectTotalValuePercent(0.75)
        elif name == "k-total-value":
            return SelectTotalValuePercent(FLAGS.selector_k)

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
    game = game_utils.load_game_as_turn_game()

    if game.num_players() == 2 and (FLAGS.player1 is not None or FLAGS.player2 is not None):
        bots = [
            _init_bot(FLAGS.player1, game, 0),
            _init_bot(FLAGS.player2, game, 1),
        ]
    else:
        bots = [_init_bot(FLAGS.player, game, i) for i in range(game.num_players())]

    initial_moves = "" if FLAGS.initial_moves is None else FLAGS.initial_moves

    for i in range(FLAGS.num_games):
        start = time.time()
        run_results_dir = results_root / f"mcts_{i}"
        collected_subproblem_dirs.append(run_results_dir)
        run_results_dir.mkdir(parents=True, exist_ok=False)
        results_dict = {"submodels_dir": run_results_dir, "name": f"mcts_{i}"}

        result, game_tree = MCSA_combined(game_utils, game, solver, bots, action_selector, formula, coalition,
                                          run_results_dir, results_dict, initial_moves,
                                          max_game_depth=FLAGS.max_game_depth,
                                          use_reward_in_terminal_states=FLAGS.use_reward_in_terminal_states)

        end = time.time()
        results_dict["decision"] = result
        results_dict["time_total"] = end - start
        results_dict["time_rl"] = results_dict["time_total"] - results_dict["time_solver"]
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
        results_dict["game"] = FLAGS.game
        results_dict["player"] = FLAGS.player
        if FLAGS.player1 is not None and FLAGS.player2 is not None:
            results_dict["player1"] = FLAGS.player1
            results_dict["player2"] = FLAGS.player2
        results_dict["formula"] = formula
        results_dict["coalition"] = coalition
        results_dict["num_submodels"] = 0  # to be filled by collect_game_tree_stats
        collect_game_tree_stats(game_tree, results_dict)
        create_single_run_report(results_dict)
        print("FINAL ANSWER:", result, f" (time: {end - start})")
        collected_results.append(results_dict)
        final_log += f"mcts ({run_results_dir}): {end - start}\n"

    print()
    print("-" * 25)
    print(final_log)

    create_final_report(collected_results, output_file)


if __name__ == "__main__":
    app.run(main)
