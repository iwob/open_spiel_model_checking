import re
import pyspiel
from game_mnk import GameInterface


class GameKuhnPoker(GameInterface):
    def __init__(self):
        GameInterface.__init__(self, players={"player0": 0, "player1": 1})

    def get_name(self):
        return "kuhn_poker"

    def load_game(self):
        # In combinatorial game theory, a misère game is one played according to the "misère play condition"; that is,
        # a player unable to move wins. This is in contrast to the "normal play condition" in which a player
        # unable to move loses.
        return pyspiel.load_game("kuhn_poker")

    def formal_subproblem_description(self, game_state, history, formulae_to_check: str = None) -> str:
        return "Placeholder"

    def termination_condition(self, history: str):
        """Determines when the branching of the game search space will conclude."""
        pass

    def get_moves_from_history_str(self, history: str) -> list[str]:
        """Converts a single history string to a list of successive actions."""
        if history == "":
            return []
        else:
            return history.split(',')  # E.g. input to process: "Deal:0,Deal:1,Pass,Bet,Pass"

    @classmethod
    def get_default_formula_and_coalition(cls):
        return "<player0> F player0wins;", {0}

