import sys
from absl import app
from absl import flags
import numpy as np
import random
import time
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator
from open_spiel.python.algorithms.alpha_zero import model as az_model
from open_spiel.python.bots import gtp
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel
import re
from model_checking import runner
from game_mnk import GameMnk, GameInterface
from game_nim import GameNim
import io
import subprocess
import os
from dataclasses import *
from typing import Optional
from pathlib import Path
import heapq
import shutil


@dataclass(init=True, order=True)
class NodeData:
    priority: int
    player: int = field(compare=False)
    moves_str: str = field(compare=False)
    # prior: float
    # value: float


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


flags.DEFINE_enum("game", "mnk", _KNOWN_GAMES, help="Name of the game.")
flags.DEFINE_integer("max_game_depth", 10, help="Maximum number of moves from the initial position that can be explored in the game tree.")
flags.DEFINE_integer("m", 5, help="(Game: mnk) Number of rows.")
flags.DEFINE_integer("n", 5, help="(Game: mnk) Number of columns.")
flags.DEFINE_integer("k", 4, help="(Game: mnk) Number of elements forming a line to win.")
flags.DEFINE_string("piles", None, help="(Game: nim) Piles in the format as in the example: '1;3;5;7'.")
flags.DEFINE_string("initial_moves", None, help="Initial actions to be specified in the game-specific format.")
flags.DEFINE_enum("player1", "mcts", _KNOWN_PLAYERS, help="Who controls player 1.")
flags.DEFINE_enum("player2", "mcts", _KNOWN_PLAYERS, help="Who controls player 2.")  # IB: oryginalnie było random
flags.DEFINE_string("gtp_path", None, help="Where to find a binary for gtp.")
flags.DEFINE_multi_string("gtp_cmd", [], help="GTP commands to run at init.")
flags.DEFINE_string("az_path", None, help="Path to an alpha_zero checkpoint. Needed by an az player.")
flags.DEFINE_string("mcmas_path", None, required=False, help="Path to the MCMAS executable.")
flags.DEFINE_string("output_file", None, required=False, help="Path to the directory in which the results of this run will be stored.")
flags.DEFINE_integer("uct_c", 1, help="UCT's exploration constant.")
flags.DEFINE_integer("rollout_count", 1, help="How many rollouts to do.")
flags.DEFINE_integer("max_simulations", 60000, help="How many simulations to run.")
flags.DEFINE_integer("num_games", 1, help="How many games to play.")
flags.DEFINE_integer("seed", None, help="Seed for the random number generator.")
flags.DEFINE_float("epsilon_ratio", 0.99, required=False, help="Seed for the random number generator.")
flags.DEFINE_bool("random_first", False, help="Play the first move randomly.")
flags.DEFINE_bool("solve", True, help="Whether to use MCTS-Solver.")
flags.DEFINE_bool("quiet", False, help="Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, help="Show the MCTS stats of possible moves.")
flags.DEFINE_bool("solve_submodels", False, required=False, help="If true, only ispl files will be created.")
FLAGS = flags.FLAGS


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


def _execute_initial_moves(state, bots, moves):
    for action_str in moves:
        action = _get_action(state, action_str)
        if action is None:
            sys.exit("Invalid action: {}".format(action_str))

        for bot in bots:
            bot.inform_action(state, state.current_player(), action)
        state.apply_action(action)


def _play_game(game_utils: GameInterface, game, bots, nodes_queue, max_game_depth):
    """Plays one game."""
    print("current nodes_queue:")
    print("\n".join([f"\"{x}\"" for x in nodes_queue]) + "\n")

    node = heapq.heappop(nodes_queue)

    ra = 0  # indicates if bots are random, used in the commented code in the game loop

    # print("moves:", moves)
    state = game.new_initial_state()
    _opt_print("Starting state:\n")
    _opt_print(state)

    _execute_initial_moves(state, bots, game_utils.get_moves_from_history(node.moves_str))
    _opt_print("State after initial moves:\n")
    _opt_print(state)

    if state.is_terminal():
        return True, state

    if node.priority >= max_game_depth:
        # Too long sequence of moves, stop processing this sequence
        # (it will constitute one of the subproblems)
        return True, state

    while not state.is_terminal():
        current_player = state.current_player()
        if node.player is None:
            node.player = current_player
        # The state can be three different types: chance node,
        # simultaneous node, or decision node
        if state.is_chance_node():
            raise ValueError("Game cannot have chance nodes.")
        elif state.is_simultaneous_node():
            raise ValueError("Game cannot have simultaneous nodes.")
        else:
            # Decision node: sample action for the single current player
            # print("dx")
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

            # Here we use a custom my_policy field, which is a custom modification of OpenSpiel which
            # doesn't give that information on the outside
            # IB ---------
            # p = bot.get_policy(state)  #return_probs=True
            # print("policy:", p)
            # ------------
            # TODO: Potentially change the name of my_policy to something more informative
            actions = [field.split(": player:")[0] for field in bot.my_policy.split("\n")[:2]]
            # print(q[0])
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
            def get_value(data_str: str) -> float:
                value = float(data_str.split(':')[4].split(',')[0])
                return value
            bot_policy_lines = bot.my_policy.split("\n")
            val1 = get_value(bot_policy_lines[:2][0])  # there should always be at least one possible action for the agent
            val2 = get_value(bot_policy_lines[:2][1]) if len(bot_policy_lines) >= 2 else None
            # TODO: parameterize the number/percent of the best policy values branches
            # sys.exit()
            # print(q[0],q[0].count('x'), q[0].count('o') )
            # print(actions)
            # print(val1, val2)

            # for bot in bots:
            #     bot.inform_action(state, state.current_player(), action)
            # state.apply_action(action)

            # Add actions to the queue
            n0 = NodeData(game_utils.get_num_actions(actions[0]),
                          player=None,
                          moves_str=game_utils.add_move_to_history(node.moves_str, actions[0]))
            heapq.heappush(nodes_queue, n0)

            if val2 is not None and val1 != 0 and val2 / val1 > FLAGS.epsilon_ratio:
                n1 = NodeData(game_utils.get_num_actions(actions[1]),
                              player=None,
                              moves_str=game_utils.add_move_to_history(node.moves_str, actions[1]))
                heapq.heappush(nodes_queue, n1)

            return False, None
    return True, state


def main(argv):
    if FLAGS.mcmas_path is None:
        mcmas_path = "/home/iwob/Programs/MCMAS/mcmas-linux64-1.3.0"
    else:
        mcmas_path = FLAGS.mcmas_path

    if FLAGS.game == "mnk":
        # Example initial moves for mnk; "x(2,2),o(3,2)"
        results_root = Path(f"results(v3)__mnk_{FLAGS.m}_{FLAGS.n}_{FLAGS.k}")
        game_utils = GameMnk(FLAGS.m, FLAGS.n, FLAGS.k)
    elif FLAGS.game == "nim":
        results_root = Path(f"results(v3)__nim_{FLAGS.piles}")
        game_utils = GameNim(FLAGS.piles)
    else:
        raise Exception("Unknown game!")

    if FLAGS.output_file is None:
        output_file = Path("output") / "results.txt"
    else:
        output_file = Path(FLAGS.output_file)
    if output_file.exists():
        output_file.unlink()

    if results_root.exists():
        shutil.rmtree(results_root)

    final_log = ""
    collected_results = []
    collected_subproblem_dirs = []
    game = game_utils.load_game()
    if game.num_players() > 2:
        sys.exit("This game requires more players than the example can handle.")

    for i in range(FLAGS.num_games):
        start = time.time()
        run_results_dir = results_root / f"mcts_{i}"
        collected_subproblem_dirs.append(run_results_dir)
        run_results_dir.mkdir(parents=True, exist_ok=False)

        # For the initial element in the queue priority and player doesn't matter
        if FLAGS.initial_moves is None:
            nodes_queue = [NodeData(0, player=None, moves_str="")]
        else:
            nodes_queue = [NodeData(0, player=None, moves_str=FLAGS.initial_moves)]

        bots = [
            _init_bot(FLAGS.player1, game, 0),
            _init_bot(FLAGS.player2, game, 1),
        ]

        def save_specification(path, moves, game_state):
            output_file = run_results_dir / path
            script = game_utils.formal_subproblem_description(game_state, history=moves)
            with output_file.open("w") as of:
                of.write(script)

        game_state = None
        while len(nodes_queue) > 0:
            is_branch_terminated, game_state = _play_game(game_utils, game, bots, nodes_queue, max_game_depth=FLAGS.max_game_depth)
            if is_branch_terminated:
                save_specification(f"output_{nodes_queue[0]}.txt", nodes_queue[0], game_state)
                _opt_print("state is terminal")
                _opt_print("Removing q[0]:", nodes_queue[0])
                # del nodes_queue[0]  # remove terminal element


        end = time.time()
        text = f"mcts ({run_results_dir}): {end - start}\n"
        collected_results.append({"path": run_results_dir, "time_rl": end - start})
        final_log += text
        # print(text)
        # final_log += f"i:{i}, {end - start}\n"
        print(f"i:{i},", end - start)

    print()
    print("-" * 25)
    print(final_log)

    if FLAGS.solve_submodels:
        runner.process_experiment_with_multiple_runs(collected_subproblem_dirs,
                                                     game_utils,
                                                     solver_path=mcmas_path,
                                                     collected_results=collected_results)

        with output_file.open("w") as f:
            total_times = []
            num_result_one = 0
            num_result_zero = 0
            for i, result_dict in enumerate(collected_results):
                result_dict["total_time"] = result_dict["time_rl"] + result_dict["time_solver"]
                if str(result_dict["answer_solver"]) == "1":
                    num_result_one += 1
                elif str(result_dict["answer_solver"]) == "0":
                    num_result_zero += 1
                total_times.append(result_dict["total_time"])
                f.write("# " + "-"*30 + "\n")
                f.write(f"# Run {i}\n")
                f.write("# " + "-"*30 + "\n")
                for k, v in result_dict.items():
                    v = str(v).replace('\n', '\t')
                    f.write(f"{k} = {v}\n")

            f.write("# " + "-" * 30 + "\n")
            f.write(f"# Total\n")
            f.write("# " + "-" * 30 + "\n")
            f.write(f"avg.total_time = {sum(total_times) / len(total_times)}\n")
            f.write(f"sum.result_0 = {num_result_zero}\n")
            f.write(f"sum.result_1 = {num_result_one}\n")



if __name__ == "__main__":
    app.run(main)
