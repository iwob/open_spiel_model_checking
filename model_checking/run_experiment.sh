#!/bin/bash

# Open Spiel należy zbudować zgodnie z instrukcją budowania open_spiel.
# następnie należy uruchomić mcts.py który wygeneruje pliki .ispl z najbardziej prawdopodobnymi ruchami.
# W pliku mnk.py znajduje się funkcja która generuje plik ispl na podstawie sekwencji ruchów.
# Plik runner.py jest odpowiedzialny za zebranie wyników i zbudowanie drzewa gry.

python3 mcts.py

python3 runner.py --results_dir results__mnk_5_5_4
