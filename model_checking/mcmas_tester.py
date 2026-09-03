import numpy as np
from pathlib import Path
import pyspiel

from model_checking.game_mcmas_model import GameInterfaceMcmasModel
from open_spiel.python.observation import make_observation
from mcmas.parsers.ispl_parser import ISPLParser
from mcmas_model_game import McmasModelGame



def simple_run_test():
    parser = ISPLParser()
    file = Path(__file__).parent / "example_specifications" / "mnk" / "mnk(3,3,3).ispl"
    model = parser.parse_file(file)
    print("Model loaded")
    print(model.agents)
    game = McmasModelGame.from_spec(model)

    state = game.new_initial_state()
    print(str(state) + '\n')
    for a_id, _ in enumerate(state.agent_local_states):
        print("Legal actions for player {}:".format(a_id))
        for action in state.legal_actions(a_id):
            print(f"{action} {state.action_to_string(a_id, action)}")
        print()

    MAX_ITER = 2
    num_iter = 0
    print("Start game")
    print("State:")
    print(str(state))
    print()
    self.assertEqual(True, True)
    while not state.is_terminal() and num_iter < MAX_ITER:
        print(f"ITERATION #{num_iter}")
        if num_iter == -1:
            actions = [0, 5, 7]
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



simple_run_test()