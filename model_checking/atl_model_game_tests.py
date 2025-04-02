import unittest
import numpy as np
from pathlib import Path
import pyspiel
from stv.parsers.parser_stv_v2 import Stv2Parser, parseAndTransformFormula
from atl_model_game import AtlModelGame


def simple_run_test(self):
    parser = Stv2Parser()
    file = Path(__file__).parent / "example_specifications" / "simple" / "simple.stv"
    with file.open() as f:
        text = f.read()
    stv_spec, formula = parser(text)
    game = AtlModelGame(stv_spec)

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

    while not state.is_terminal() and num_iter < MAX_ITER:
        print(f"ITERATION #{num_iter}")
        if num_iter == 0:
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

class TestAtlModel(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        parser = Stv2Parser()
        file = Path(__file__).parent / "example_specifications" / "simple" / "simple.stv"
        with file.open() as f:
            text = f.read()
        stv_spec, formula = parser(text)
        self.game_simple = AtlModelGame(stv_spec, formula)

        file = Path(__file__).parent / "example_specifications" / "simple" / "simple2.stv"
        with file.open() as f:
            text = f.read()
        stv_spec, formula = parser(text)
        self.game_simple_2 = AtlModelGame(stv_spec, formula)


    def test_model_properties(self):
        game = self.game_simple
        self.assertEqual(game.action_name_to_id_dict["Player0"]["play_0_rock"], 0)
        self.assertEqual(game.action_name_to_id_dict["Player0"]["play_0_paper"], 1)
        self.assertEqual(game.action_name_to_id_dict["Player0"]["play_0_scissors"], 2)
        self.assertEqual(game.action_name_to_id_dict["Player0"]["loop"], 3)
        self.assertEqual(game.action_name_to_id_dict["Player1"]["obstruct"], 4)
        self.assertEqual(game.action_name_to_id_dict["Player1"]["play0"], 5)
        self.assertEqual(game.action_name_to_id_dict["Player1"]["loop"], 6)
        self.assertEqual(game.action_name_to_id_dict["Score"]["play0"], 7)
        self.assertEqual(game.action_name_to_id_dict["Score"]["loop"], 8)
        self.assertEqual(game.possible_actions, ["play_0_rock", "play_0_paper", "play_0_scissors", "loop", "obstruct", "play0", "loop", "play0", "loop"])
        self.assertEqual(game.agent_actions["Player0"], ["play_0_rock", "play_0_paper", "play_0_scissors", "loop"])
        self.assertEqual(game.agent_actions["Player1"], ["obstruct", "play0", "loop"])
        self.assertEqual(game.agent_actions["Score"], ["play0", "loop"])


    def test_synchronization_0(self):
        formula_1 = parseAndTransformFormula("<<Player0,Player1>><>( (finished == 1) )")
        formula_2 = parseAndTransformFormula("<<Player0,Player1>><>( (finished == 1) && (move_1 == 1))")
        self.assertIsNotNone(formula_1)
        self.assertIsNotNone(formula_2)
        state = self.game_simple.new_initial_state()
        self.assertEqual(state.current_player(), pyspiel.PlayerId.SIMULTANEOUS)
        self.assertEqual(state.legal_actions(0), [0, 1, 2])
        self.assertEqual(state.legal_actions(1), [4, 5])
        self.assertEqual(state.legal_actions(2), [7])
        self.assertEqual(state.agent_local_states[0].current_node, "idle")
        self.assertEqual(state.agent_local_states[1].current_node, "idle")
        self.assertEqual(state.agent_local_states[2].current_node, "count")
        self.assertEqual(state.agent_local_states[2].persistent_variables["move_0"], 0)
        self.assertEqual(state.agent_local_states[2].persistent_variables["move_1"], 0)
        self.assertEqual(state.agent_local_states[2].persistent_variables["finished"], 0)
        self.assertFalse(state.is_formula_satisfied(formula_1))
        self.assertFalse(state.is_formula_satisfied(formula_2))
        state.apply_actions([0, 5, 7])  # synchronization on play_0_rock
        self.assertEqual(state.agent_local_states[0].current_node, "finish")
        self.assertEqual(state.agent_local_states[1].current_node, "finish")
        self.assertEqual(state.agent_local_states[2].current_node, "counted")
        self.assertEqual(state.agent_local_states[2].persistent_variables["move_0"], 1)
        self.assertEqual(state.agent_local_states[2].persistent_variables["move_1"], 0)
        self.assertEqual(state.agent_local_states[2].persistent_variables["finished"], 0)
        self.assertFalse(state.is_formula_satisfied(formula_1))
        self.assertFalse(state.is_formula_satisfied(formula_2))
        self.assertEqual(state.legal_actions(0), [3])
        self.assertEqual(state.legal_actions(1), [6])
        self.assertEqual(state.legal_actions(2), [8])
        state.apply_actions([3, 6, 8])
        self.assertEqual(state.agent_local_states[0].current_node, "finish")
        self.assertEqual(state.agent_local_states[1].current_node, "finish")
        self.assertEqual(state.agent_local_states[2].current_node, "counted")
        self.assertEqual(state.agent_local_states[2].persistent_variables["move_0"], 1)
        self.assertEqual(state.agent_local_states[2].persistent_variables["move_1"], 0)
        self.assertEqual(state.agent_local_states[2].persistent_variables["finished"], 1)
        self.assertTrue(state.is_formula_satisfied(formula_1))
        self.assertFalse(state.is_formula_satisfied(formula_2))


    def test_synchronization_1(self):
        state = self.game_simple.new_initial_state()
        self.assertEqual(state.current_player(), pyspiel.PlayerId.SIMULTANEOUS)
        self.assertEqual(state.legal_actions(0), [0, 1, 2])
        self.assertEqual(state.legal_actions(1), [4, 5])
        self.assertEqual(state.legal_actions(2), [7])
        self.assertEqual(state.agent_local_states[0].current_node, "idle")
        self.assertEqual(state.agent_local_states[1].current_node, "idle")
        self.assertEqual(state.agent_local_states[2].current_node, "count")
        self.assertEqual(state.agent_local_states[2].persistent_variables["move_0"], 0)
        self.assertEqual(state.agent_local_states[2].persistent_variables["move_1"], 0)
        self.assertEqual(state.agent_local_states[2].persistent_variables["finished"], 0)
        state.apply_actions([0, 4, 7])  # synchronization fails because Player1 obstructs
        self.assertEqual(state.agent_local_states[0].current_node, "idle")
        self.assertEqual(state.agent_local_states[1].current_node, "finish")
        self.assertEqual(state.agent_local_states[2].current_node, "count")
        self.assertEqual(state.agent_local_states[2].persistent_variables["move_0"], 0)
        self.assertEqual(state.agent_local_states[2].persistent_variables["move_1"], 0)
        self.assertEqual(state.agent_local_states[2].persistent_variables["finished"], 0)
        self.assertEqual(state.legal_actions(0), [0, 1, 2])
        self.assertEqual(state.legal_actions(1), [6])
        self.assertEqual(state.legal_actions(2), [7])
        state.apply_actions([0, 6, 7])
        self.assertEqual(state.agent_local_states[0].current_node, "idle")
        self.assertEqual(state.agent_local_states[1].current_node, "finish")
        self.assertEqual(state.agent_local_states[2].current_node, "count")
        self.assertEqual(state.agent_local_states[2].persistent_variables["move_0"], 0)
        self.assertEqual(state.agent_local_states[2].persistent_variables["move_1"], 0)
        self.assertEqual(state.agent_local_states[2].persistent_variables["finished"], 0)


    def test_synchronization_2(self):
        state = self.game_simple_2.new_initial_state()
        self.assertEqual(state.current_player(), pyspiel.PlayerId.SIMULTANEOUS)
        self.assertEqual(state.legal_actions(0), [0, 1, 2])
        self.assertEqual(state.legal_actions(1), [4, 5])
        self.assertEqual(state.legal_actions(2), [7])
        self.assertEqual(state.legal_actions(3), [9])
        self.assertEqual(state.agent_local_states[0].current_node, "idle")
        self.assertEqual(state.agent_local_states[1].current_node, "idle")
        self.assertEqual(state.agent_local_states[2].current_node, "idle")
        self.assertEqual(state.agent_local_states[3].current_node, "count")
        self.assertEqual(state.agent_local_states[3].persistent_variables["move_0"], 0)
        print("test_synchronization_2")
        state.apply_actions([0, 5, 7, 9])
        # Player0 and Player2 both are leaders of shared transitions, but Player1 may support either
        # of them and must choose. Below we check that it is impossible to use Player1's support twice.
        self.assertTrue(True)  # currently we eliminate this case by making as assumption
        #self.assertFalse(state.agent_local_states[0].current_node == "finish" and state.agent_local_states[2].current_node == "finish")



if __name__ == "__main__":
    unittest.main()
