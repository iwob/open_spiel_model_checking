import time
import os
import sys
import numpy as np

import mcts_v3
from action_selectors import SelectKBestActions
from open_spiel.python.algorithms import mcts
from open_spiel.python.bots import uniform_random
from pathlib import Path
from game_mnk import GameMnk
from solvers import SolverMCMAS


def _init_bot(bot_type, game, player_id, rng):
    """Initializes a bot by type."""
    if bot_type == "mcts":
        rollout_count = 10
        evaluator = mcts.RandomRolloutEvaluator(rollout_count, rng)
        return mcts.MCTSBot(
            game,
            1,  # uct_c
            100000,  # max_simulations
            evaluator,
            random_state=rng,
            solve=True,  # Whether to use MCTS-Solver (improvement over traditional MCTS)
            verbose=False)
    if bot_type == "random":
        return uniform_random.UniformRandomBot(player_id, rng)
    raise ValueError("Invalid bot type: %s" % bot_type)


def verification_times_experiments():
    seed = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
    print("seed:", seed)
    results_dir = sys.argv[2] if len(sys.argv) >= 3 else "verification_tests"
    bot_type = "mcts"
    m, n, k = 5, 5, 4
    mcmas_path = "/home/iwob/Programs/MCMAS/mcmas-linux64-1.3.0"
    solver = SolverMCMAS(mcmas_path, time_limit=1000 * 3600 * 4)
    game_utils = GameMnk(m, n, k)
    formula, coalition = game_utils.get_default_formula_and_coalition()
    game = game_utils.load_game()
    rng = np.random.RandomState(seed)
    bots = [
        _init_bot(bot_type, game, 0, rng),
        _init_bot(bot_type, game, 1, rng)
    ]

    def simulate_run():
        state = game.new_initial_state()
        moves = []
        while not state.is_terminal():
            if state.is_chance_node():
                raise ValueError("Game cannot have chance nodes.")
            elif state.is_simultaneous_node():
                raise ValueError("Game cannot have simultaneous nodes.")
            else:
                current_player = state.current_player()
                bot = bots[current_player]
                action = bot.step(state)
                for bot in bots:
                    bot.inform_action(state, state.current_player(), action)
                moves.append(state.action_to_string(state.current_player(), action))
                state.apply_action(action)

        results = []
        for i in reversed(range(len(moves))):
            if i < 7:
                break
            print(f"moves (i={i}): {moves[:i]}")
            script = game_utils.formal_subproblem_description(state, moves[:i], formulae_to_check=formula)
            # path = Path(f"verification_tests/{','.join(moves[:i]).replace(',', '')}.ispl")
            path = Path(f"{results_dir}/{','.join(moves[:i]).replace(',', '')}.ispl")
            path.parent.mkdir(exist_ok=True)
            with path.open("w") as f:
                f.write(script)
            start = time.time()
            res = solver.verify_from_file(path)
            end = time.time()
            results.append((i, end - start))
            print("-" * 50)
            print("Result:", (i, end - start))
            print("-" * 50)
        return results

    for i in range(1):
        results = simulate_run()
        for r in results:
            print(f"{r[0]}: {r[1]}")



def check_different_heights_of_tree_encoding():
    seed = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
    print("seed:", seed)
    bot_type = "mcts"
    m, n, k = 3, 3, 3
    mcmas_path = "/home/iwob/Programs/MCMAS/mcmas-linux64-1.3.0"
    solver = SolverMCMAS(mcmas_path, time_limit=1000 * 3600 * 4)
    game_utils = GameMnk(m, n, k)
    formula, coalition = game_utils.get_default_formula_and_coalition()
    game = game_utils.load_game()
    rng = np.random.RandomState(seed)
    bots = [
        _init_bot(bot_type, game, 0, rng),
        _init_bot(bot_type, game, 1, rng)
    ]
    run_results_dir = Path("exp2")
    summary_file = run_results_dir / "summary.txt"
    run_results_dir.mkdir(exist_ok=True)

    results_dict = {}
    for game_depth in [2, 4, 6]:
        for num_actions in [1, 2, 3, 4]:
            print(f"Running: {game_depth}, {num_actions}")
            action_selector = SelectKBestActions(num_actions)
            game_tree = mcts_v3.generate_game_tree(game_utils, game, bots, action_selector, coalition,
                                                   initial_moves="",
                                                   max_game_depth=game_depth)
            script = game_utils.formal_subproblem_description_game_tree(game_tree, "", formula)
            script_path = run_results_dir / f"game_tree_spec_{game_depth}_{num_actions}.ispl"
            with script_path.open("w") as f:
                f.write(script)

            # Verify submodels
            start = time.time()
            result, meta = solver.verify_from_file(script_path)
            end = time.time()
            results_dict[(game_depth, num_actions)] = end - start

    text = ""
    for k, v in results_dict.items():
        text += f"{k.replace(' ', '')}\t{v}\n"
    with summary_file.open("w") as f:
        f.write(text)
    print(f"\n{text}")

if __name__ == "__main__":
    # verification_times_experiments()

    check_different_heights_of_tree_encoding()
