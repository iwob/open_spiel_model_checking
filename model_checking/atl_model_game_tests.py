import numpy as np
from pathlib import Path
import pyspiel
from stv.parsers.parser_stv_v2 import Stv2Parser
from atl_model_game import AtlModelGame

# game = pyspiel.load_game("tic_tac_toe")

np.random.seed(10)

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
for a_id, _ in enumerate(state.agent_local_states):
    print("Legal actions for player {}:".format(a_id))
    for action in state.legal_actions(a_id):
        print(f"{action} {state.action_to_string(a_id, action)}")
    print()

MAX_ITER = 1
num_iter = 0
print("Start game")
print("State:")
print(str(state))
print()

while not state.is_terminal() and num_iter < MAX_ITER:

    if num_iter == 0:
        actions = [0, 5, 8]
    else:
        actions = []
        for a_id, a in enumerate(state.agent_local_states):
            j = np.random.choice(state.legal_actions(a_id))
            actions.append(j)
    print(f"Trying to execute actions: {actions} ({[state.get_action_name(a) for a in actions]})")
    state.apply_actions(actions)

    print("State:")
    print(str(state))
    print()
    num_iter += 1
