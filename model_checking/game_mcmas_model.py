from pathlib import Path
import re

from mcmas_model_game import McmasModelGame, McmasModelState
from game_mnk import GameInterface
import pyspiel

from model_checking.mcmas.parsers.ispl_parser import ISPLParser, StrategicFormula


class GameInterfaceMcmasModel(GameInterface):
    def __init__(self, model_path: str):
        parser = ISPLParser()
        with Path(model_path).open("r", encoding="utf-8") as file:
            self.model_text = file.read()
        self.model = parser.parse(self.model_text)
        self.formula: StrategicFormula = self.model.formulae.formulas[0]
        self.coalition = self.model.groups.find_group_members(self.formula.agent)
        GameInterface.__init__(self, players={a.name: i for i, a in enumerate(self.model.agents)})

    def get_name(self):
        return "mcmas_model"

    def load_game(self) -> pyspiel.Game:
        params = {"spec": self.model, "formula": self.formula}
        return McmasModelGame(params)

    def load_game_as_turn_game(self) -> pyspiel.Game:
        game = self.load_game()
        return pyspiel.convert_to_turn_based(game)

    def formal_subproblem_description(self, game_state: McmasModelState, history, formulae_to_check: str = None, is_in_turn_wrapper=True) -> str:
        # The idea: rules of the game remain the same, only values of variables are changed.
        game_state = game_state.simultaneous_game_state() if is_in_turn_wrapper else game_state
        formula = formulae_to_check if formulae_to_check is not None else self.formula
        if isinstance(formula, StrategicFormula):
            formula_text = formula.get_text() + ";"
        else:
            formula_text = str(formula)
        init_text = " and ".join([f"Environment.{k} = {v}" for k, v in game_state.env_variables.items()])
        spec = re.sub(r"InitStates.*?end InitStates", f"InitStates\n{init_text};\nend InitStates", self.model_text, flags=re.DOTALL)
        spec = re.sub(r"Formulae.*?end Formulae", f"Formulae\n{formula_text}\nend Formulae", spec, flags=re.DOTALL)
        return spec

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