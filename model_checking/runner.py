import os
import subprocess
from collections import defaultdict
import re
import time
from pathlib import Path

from model_checking.game_mnk import GameInterface, GameMnk
from model_checking.game_nim import GameNim

# The solution in this file assumes that in each node of the game tree there
# are checked two formulas, regardless of which of the players is to move.


def parse_output(output):
    lines = output.split('\n')
    
    verification_results = {}
    execution_time = None
    # TODO: Currently there is an assumption that we check winning formula for both players in this node (see comment at the top of this file)
    for line in lines:
        if "Formula number 1:" in line:
            verification_results["Formula 1"] = "TRUE" if "is TRUE" in line else "FALSE"
        elif "Formula number 2:" in line:
            verification_results["Formula 2"] = "TRUE" if "is TRUE" in line else "FALSE"
        elif "execution time =" in line:
            execution_time = float(line.split('=')[-1].strip())
    
    return verification_results, execution_time


def parse_logs_correctly_fixed_v2(results, game_utils: GameInterface):
    game_tree = {}
    pattern_results = r"Formula 1: (\w+)\n\s+Formula 2: (\w+)"
    
    for result in results:
        log = result["log_entry"]
        results_match = re.search(pattern_results, log)
        
        if results_match:
            moves = game_utils.get_moves_from_history(result["history"])
            result_x = results_match.group(1) == "TRUE"
            result_o = results_match.group(2) == "TRUE"
            
            # Traverse through the tree and build nodes using full moves
            node = game_tree
            for move in moves:
                if move not in node:
                    node[move] = {}
                node = node[move]
            # Store the results at the leaf node
            node['result'] = (result_x, result_o)

    return game_tree

def minimax(node, is_maximizing_player):
    if 'result' in node:
        # If it's a leaf node, return the result for player X
        return 1 if node['result'][0] else -1 if node['result'][1] else 0
    
    if is_maximizing_player:
        best_value = -float('inf')
        for key in node:
            value = minimax(node[key], False)
            best_value = max(best_value, value)
        return best_value
    else:
        best_value = float('inf')
        for key in node:
            value = minimax(node[key], True)
            best_value = min(best_value, value)
        return best_value

def generate_latex_table(logs):
    # LaTeX table header
    table = "\\begin{tabular}{|c|c|c|}\n\\hline\n"
    table += "Moves sequence & Victory X & Victory O \\\\\n\\hline\n"

    for log in logs:
        # Using regex to extract filename and results of the formulas
        file_name_match = re.search(r"File: (.+)\.txt", log)
        formula_1_match = re.search(r"Formula 1: (TRUE|FALSE)", log)
        formula_2_match = re.search(r"Formula 2: (TRUE|FALSE)", log)
        
        if file_name_match and formula_1_match and formula_2_match:
            # Sequence of moves is a part of the filename between 'output_' and '.txt'
            sequence = file_name_match.group(1).replace("output_", "")
            win_x = "YES" if formula_1_match.group(1) == "TRUE" else "NO"
            win_o = "YES" if formula_2_match.group(1) == "TRUE" else "NO"
            
            # Adding row to the table
            table += f"{sequence} & {win_x} & {win_o} \\\\\n\\hline\n"

    table += "\\end{tabular}"
    
    return table


def run_MCMAS(file_path, mcmas_path):
    result = subprocess.run([mcmas_path, file_path], capture_output=True, text=True)
    output = result.stdout  # Capturing standard output
    print(output)
    return output


def verify_submodels_MCMAS(folder_path, mcmas_path):
    """Verifies all submodels in a given folder using MCMAS."""
    results = []
    # Iterating over all files in the directory
    for file_name in os.listdir(folder_path):
        print("(*) Processing file:", file_name)
        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path):
            output = run_MCMAS(file_path, mcmas_path)
            verification_results, execution_time = parse_output(output)

            # Read file to get the
            with open(file_path, "r") as f:
                content = f.read()
                pattern_history = r"--\s*History: (.*)\n"
                history = re.search(pattern_history, content)
                if not history:
                    raise Exception("Submodel file does not contain a '-- History: <moves>' section!")
                history = history.group(1)

            results.append({
                "file_name": file_name,
                "history": history,
                "verification_results": verification_results,
                "execution_time_mcmas": execution_time
            })

    return results


def process_experiment_with_multiple_runs(run_result_dirs, game_utils, solver_path, collected_results=None):
    """Handles collection and interpretation of results by invoking MCSA for each run conducted in a given experiment,
     where by an experiment we mean a particular model checking problem.

    :param run_result_dirs: A path to the root directory containing directories (run-directories) for separate experiment runs.
     Each run-directory contains several submodel .ispl files.
     :param game_utils: Game-specific utilities handling various important processing aspects.
    :return:
    """
    final_outputs = ""
    for i, folder_path in enumerate(run_result_dirs):
        start = time.time()
        debug = False
        logs = []
        if debug:
            # TODO: This will not work currently
            files = ["output_x(2,2),o(1,2).txt", "output_x(2,2),o(3,3),x(0,0).txt", "output_x(2,2),o(3,3),x(1,1).txt"]
            f1 = ["FALSE", "TRUE", "FALSE"]
            f2 = ["TRUE", "FALSE", "FALSE"]
            exs = [0, 0]
            for file, f1_, f2_, e in zip(files, f1, f2, exs):
                log_entry = (
                    f"File: {file}\n"
                    f"  Formula 1: {f1_}\n"
                    f"  Formula 2: {f2_}\n"
                    f"  Execution time: {e} s"
                )
                logs.append(log_entry)
        else:
            results = verify_submodels_MCMAS(folder_path, solver_path)
            for result in results:
                log_entry = (
                    f"File: {result['file_name']}\n"
                    f"  Formula 1: {result['verification_results'].get('Formula 1')}\n"
                    f"  Formula 2: {result['verification_results'].get('Formula 2')}\n"
                    f"  Execution time: {result['execution_time_mcmas']} s"
                )
                result["log_entry"] = log_entry
                logs.append(log_entry)

        # Build the game tree with parsed moves
        game_tree_corrected_fixed_v2 = parse_logs_correctly_fixed_v2(results, game_utils)

        # Apply minimax to the root of the corrected and fixed game tree
        result_for_x_corrected_fixed_v2 = minimax(game_tree_corrected_fixed_v2, True)
        # print(game_tree_corrected_fixed_v2, result_for_x_corrected_fixed_v2)
        end = time.time()
        output = f"results ({folder_path}): {end - start} {result_for_x_corrected_fixed_v2}\n"
        final_outputs += output
        print(f"results ({folder_path}):", end - start, result_for_x_corrected_fixed_v2)
        # print(generate_latex_table(logs))

        if collected_results is not None:
            for k, v in result.items():
                collected_results[i][k] = str(v).replace('\n', '\t')
            collected_results[i]["time_solver"] = end - start
            collected_results[i]["answer_solver"] = result_for_x_corrected_fixed_v2

    print("-" * 40)
    print(final_outputs)
    return final_outputs


def main(argv):
    if FLAGS.mcmas_path is None:
        mcmas_path = "/home/iwob/Programs/MCMAS/mcmas-linux64-1.3.0"
    else:
        mcmas_path = FLAGS.mcmas_path

    if FLAGS.game == "mnk":
        game_utils = GameMnk(FLAGS.m, FLAGS.n, FLAGS.k)
    elif FLAGS.game == "nim":
        game_utils = GameNim(FLAGS.piles)
    else:
        raise Exception("Unknown game!")

    run_result_dirs = []
    for f in sorted(Path(FLAGS.results_dir).iterdir()):
        print(f)
        if f.is_dir():
            run_result_dirs.append(f)

    process_experiment_with_multiple_runs(run_result_dirs, game_utils, solver_path=mcmas_path)



if __name__ == "__main__":
    from absl import app
    from absl import flags

    _KNOWN_GAMES = ["mnk", "nim"]

    flags.DEFINE_enum("game", None, enum_values=_KNOWN_GAMES, required=True, help="Name of the game.")
    flags.DEFINE_integer("m", 5, help="(Game: mnk) Number of rows.")
    flags.DEFINE_integer("n", 5, help="(Game: mnk) Number of columns.")
    flags.DEFINE_integer("k", 4, help="(Game: mnk) Number of elements forming a line to win.")
    flags.DEFINE_string("piles", None, help="(Game: nim) Piles in the format as in the example: '1;3;5;7'.")
    flags.DEFINE_string("results_dir", None, required=True, help="Directory containing all results generated by mcts.py, each of which represents a certain submodel.")
    flags.DEFINE_string("mcmas_path", None, required=False, help="Path to the MCMAS executable.")
    FLAGS = flags.FLAGS

    app.run(main)
