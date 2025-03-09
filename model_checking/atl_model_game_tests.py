import numpy as np
from pathlib import Path
import pyspiel
from stv.parsers.parser_stv_v2 import Stv2Parser
from atl_model_game import AtlModelGame

# game = pyspiel.load_game("tic_tac_toe")

print("Starting...")
print(__file__)

parser = Stv2Parser()
file = Path(__file__).parent / "example_specifications" / "simple" / "simple.stv"
text = file.open().read()
stv_spec, formula = parser(text)

print("Model loaded")

game = AtlModelGame(stv_spec)

state = game.new_initial_state()
print(str(state) + '\n')

for action in state.legal_actions():
  print(f"{action} {state.action_to_string(action)}")


while not state.is_terminal():
    i = np.random.choice(state.legal_actions())
    print("Executing action: {}".format(state.game.actionable_steps[i]))
    state.apply_action(i)
    print(str(state) + '\n')
