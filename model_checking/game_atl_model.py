from pathlib import Path
import re

from atl_model_game import AtlModelGame, AtlModelState
from game_mnk import GameInterface
import pyspiel
from stv.parsers.parser_stv_v2 import Stv2Parser
from stv.parsers.stv_specification import *





class GameInterfaceAtlModel(GameInterface):
    def __init__(self, atl_spec_path):
        parser = Stv2Parser()
        with Path(atl_spec_path).open() as f:
            text = f.read()
        self.stv_spec, self.formula = parser(text)
        self.coalition = set()
        for i, a in enumerate(self.stv_spec.agents):
            if a.name in self.formula.coalition:
                self.coalition.add(i)
        GameInterface.__init__(self, players={"cross": 0, "nought": 1})

    def get_name(self):
        return "atl_model"

    def load_game(self) -> pyspiel.Game:
        params = {"spec": self.stv_spec, "formula": self.formula}
        return AtlModelGame(params)

    def load_game_as_turn_game(self) -> pyspiel.Game:
        game = self.load_game()
        return pyspiel.convert_to_turn_based(game)

    def formal_subproblem_description(self, game_state: AtlModelState, history, formulae_to_check: str = None, is_in_turn_wrapper=True) -> str:
        # game_state is assumed here to be a simultanous AtlModelState in a turn wrapper; if not wrapped, it will lead to errors.
        game_state = game_state.simultaneous_game_state() if is_in_turn_wrapper else game_state
        replacements = {}
        for a in game_state.agent_local_states:
            replacements[a.name] = (a.current_node, a.persistent_variables)
        return generate_stv2_encoding(self.stv_spec, self.formula, replacements=replacements)

    def formal_subproblem_description_game_tree(self, game_tree, history, formulae_to_check: str = None) -> str:
        """Generates a formal description of a subproblem resulting from removing actions not included in the
        game tree. History is used to generate the initial state."""
        raise Exception("Generation of subproblem description from game tree not supported!")

    def termination_condition(self, history: str):
        """Determines when the branching of the game search space will conclude."""
        # Method currently not used, instead state handles termination conditions
        return False

    def get_moves_from_history_str(self, history: str) -> list[str]:
        """Converts a single history string to a list of successive actions."""
        return re.findall(r'[xo]\(\d+,\d+\)', history)

    @classmethod
    def get_default_formula_and_coalition(cls):
        raise Exception("Default formula not supported for this interface. Use .formula attribute instead.")