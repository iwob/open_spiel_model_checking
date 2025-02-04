#!/bin/bash

# python3 mcts_v3.py --game mnk -m 5 -n 5 -k 4 --num_games 10 --quiet 1 --seed 42 --max_game_depth 10 --output_file "output/results(v3)_mnk554.txt"



# python3 mcts_v2.py --game mnk -m 5 -n 5 -k 4 --num_games 10 --quiet 1 --seed 42 --max_game_depth 10 --solve_submodels --output_file "output/results(v2)_mnk554.txt"


# 1.02.2025

#python3 mcts_v3.py --game nim --piles "5;5;5" --num_games 10 --quiet 1 --seed 42 --max_game_depth 10 --output_file "output/results(v3)_nim_5;5;5.txt"



#python3 mcts_v2.py --game nim --piles "5;5;5" --num_games 10 --quiet 1 --seed 42 --max_game_depth 10 --solve_submodels --output_file "output/results(v2)_nim_5;5;5.txt"



# --- 2.02.2025


python3 mcts_v3.py --game mnk -m 5 -n 5 -k 4 --num_games 10 --quiet 1 --seed 42 --max_game_depth 8 --output_file "output/results(v3)_mnk554.txt" --action_selector1 2-best --action_selector2 2-best


python3 mcts_v3.py --game nim --piles "5;5;5" --num_games 10 --quiet 1 --seed 42 --max_game_depth 10 --output_file "output/results(v3)_nim_5;5;5.txt" --action_selector1 2-best --action_selector2 2-best 


# -----------------------


python3 mcts_v3.py --game mnk -m 5 -n 5 -k 4 --num_games 2 --quiet 1 --seed 42 --max_game_depth 5 --output_file "output/results(v3)_mnk554_all2.txt" --action_selector1 most2 --action_selector2 all --submodels_dir "results(v3)__mnk_5_5_4_all2"

python3 mcts_v3.py --game mnk -m 5 -n 5 -k 4 --num_games 2 --quiet 1 --seed 42 --max_game_depth 5 --output_file "output/results(v3)_mnk554_all1.txt" --action_selector1 all  --action_selector2 most2 --submodels_dir "results(v3)__mnk_5_5_4_all1"
