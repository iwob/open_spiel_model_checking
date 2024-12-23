import os
import subprocess
from collections import defaultdict
import re
import time
def run_mcmas(file_path):
    # Ścieżka do programu mcmas
    mcmas_path = "/home/iwob/Programs/MCMAS/mcmas-linux64-1.3.0"
    
    # Wywołanie polecenia i przechwycenie wyjścia
    result = subprocess.run([mcmas_path, file_path], capture_output=True, text=True)
    
    # Przechwycenie standardowego wyjścia
    output = result.stdout
    print(output)
    return output

def parse_output(output):
    lines = output.split('\n')
    
    verification_results = {}
    execution_time = None
    
    for line in lines:
        if "Formula number 1:" in line:
            verification_results["Formula 1"] = "TRUE" if "is TRUE" in line else "FALSE"
        elif "Formula number 2:" in line:
            verification_results["Formula 2"] = "TRUE" if "is TRUE" in line else "FALSE"
        elif "execution time =" in line:
            execution_time = float(line.split('=')[-1].strip())
    
    return verification_results, execution_time

def main(folder_path):
    results = []
    
    # Iteracja przez wszystkie pliki w folderze
    for file_name in os.listdir(folder_path):
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.isfile(file_path):
            output = run_mcmas(file_path)
            verification_results, execution_time = parse_output(output)
            
            results.append({
                "file_name": file_name,
                "verification_results": verification_results,
                "execution_time": execution_time
            })
    
    return results

# Przykładowe użycie




def parse_logs_correctly_fixed_v2(logs):
    game_tree = {}
    pattern_moves = r"output_(.*)\.txt"
    pattern_results = r"Formula 1: (\w+)\n\s+Formula 2: (\w+)"
    
    for log in logs:
        moves_match = re.search(pattern_moves, log)
        results_match = re.search(pattern_results, log)
        
        if moves_match and results_match:
            # Split the moves correctly
            moves = re.findall(r'[xo]\(\d,\d\)', moves_match.group(1))
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
    # Nagłówek tabeli LaTeX
    table = "\\begin{tabular}{|c|c|c|}\n\\hline\n"
    table += "Sekwencja ruchów & Wygrana X & Wygrana O \\\\\n\\hline\n"

    for log in logs:
        # Wyciąganie nazwy pliku i wyników formuł za pomocą regex
        file_name_match = re.search(r"Plik: (.+)\.txt", log)
        formula_1_match = re.search(r"Formula 1: (TRUE|FALSE)", log)
        formula_2_match = re.search(r"Formula 2: (TRUE|FALSE)", log)
        
        if file_name_match and formula_1_match and formula_2_match:
            # Sekwencja ruchów to część nazwy pliku pomiędzy 'output_' a '.txt'
            sequence = file_name_match.group(1).replace("output_", "")
            win_x = "TAK" if formula_1_match.group(1) == "TRUE" else "NIE"
            win_o = "TAK" if formula_2_match.group(1) == "TRUE" else "NIE"
            
            # Dodawanie wiersza do tabeli
            table += f"{sequence} & {win_x} & {win_o} \\\\\n\\hline\n"

    table += "\\end{tabular}"
    
    return table
for i in range(0,10):
    start = time.time()
    debug = 0
    folder_path = f"/home/iwob/mcts_start__5_5_4{i}/"  # Zmienna wskazująca na folder z plikami
    if debug != 1:
        results = main(folder_path)
    logs = []
    if debug:
        files = ["output_x(2,2),o(1,2).txt", "output_x(2,2),o(3,3),x(0,0).txt", "output_x(2,2),o(3,3),x(1,1).txt"]
        f1 = ["FALSE", "TRUE", "FALSE"]
        f2 = ["TRUE", "FALSE", "FALSE"]
        exs = [0, 0]
        for file, f1_, f2_,e in zip(files, f1, f2, exs):
            log_entry = (
            f"Plik: {file}\n"
            f"  Formula 1: {f1_}\n"
            f"  Formula 2: {f2_}\n"
            f"  Execution time: {e} s"
        )
            logs.append(log_entry)
    else:
        for result in results:
            log_entry = (
                f"Plik: {result['file_name']}\n"
                f"  Formula 1: {result['verification_results'].get('Formula 1')}\n"
                f"  Formula 2: {result['verification_results'].get('Formula 2')}\n"
                f"  Execution time: {result['execution_time']} s"
            )
            logs.append(log_entry)

# Wyświetlenie wynikówfiles = ["output_x(2,2),o(1,2).txt","output_x(2,2),o(3,3),x(0,0).txt", "output_x(2,2),o(3,3),x(1,1).txt"]
    f1 = ["FALSE","TRUE", "FALSE"]
    f2 = ["TRUE", "FALSE","FALSE"]
    exs = [0,0]
    

    # Build the game tree correctly now with properly parsed moves
    game_tree_corrected_fixed_v2 = parse_logs_correctly_fixed_v2(logs)

    # Apply minimax to the root of the corrected and fixed game tree
    result_for_x_corrected_fixed_v2 = minimax(game_tree_corrected_fixed_v2, True)
    #print(game_tree_corrected_fixed_v2, result_for_x_corrected_fixed_v2)
    end = time.time()
    print(f"i:{i}:",end-start,result_for_x_corrected_fixed_v2)
    #print(generate_latex_table(logs))