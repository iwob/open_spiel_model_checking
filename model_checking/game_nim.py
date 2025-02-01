import re
from textwrap import dedent, indent
import pyspiel
from game_mnk import GameInterface


MAX_PILE_VALUE = 10

def generate_player_protocol(piles, player_no):
    conditions = []
    for i in range(0, len(piles)):
        for j in range(1, piles[i]+1):
            conditions.append(f"Environment.turn = player{player_no} and Environment.pile{i+1} >= {j}: {{ pile{i+1}_take{j} }};")
    return conditions

def generate_evaluation_conditions_win(piles, player):
    text = f"Environment.turn = player{1-player} and "
    text += " and ".join([f"Environment.pile{i+1} = 0" for i in range(len(piles))])
    text += ";"
    return text

def generate_piles_init_conditions(piles, player_to_move, history, add_comment=True):
    text = " and ".join([f"Environment.pile{i+1} = {piles[i]}" for i in range(len(piles))])
    # for i in range(len(piles)):
    #     text += f"Environment.pile{i+1} = {piles[i]}"
    # text = " and ".join(conditions)[:-1]

    if history and add_comment:
        comment  = f"--  History: {history}\n"
        comment += f"--  Game state:\n"
        comment += f"--  ({player_to_move}): {' '.join([str(x) for x in piles])}\n"
        return comment + text
    else:
        return text

def generate_actions(piles: list):
    actions = []
    for i in range(len(piles)):  # pile index
        for j in range(1, piles[i]+1):  # number of objects possible to take off the pile
            actions.append(f"pile{i+1}_take{j}")
    actions.append("none")
    return actions

def generate_piles_evolution(piles: list):
    conditions = []
    for i in range(len(piles)):  # pile index
        for j in range(1, piles[i]+1):  # number of objects possible to take off the pile
            condition = f"pile{i+1} = pile{i+1} - {j} if pile{i+1} >= {j} and (Player0.Action = pile{i+1}_take{j} or Player1.Action = pile{i+1}_take{j});"
            conditions.append(condition)
    return conditions

def get_env_str(piles: list):
    obsvars = "\n".join([f"pile{i} : 0 .. {MAX_PILE_VALUE};" for i in range(1, len(piles)+1)])
    piles_evolution = "\n".join(generate_piles_evolution(piles))
    return f"""\
Agent Environment
    Obsvars:
        turn : {{player0, player1}};
{indent(obsvars, " "*8)}
    end Obsvars
    Actions = {{ }}; 
    Protocol: end Protocol
    Evolution:
        turn = player0 if turn = player1 and (! Player1.Action = none);
        turn = player1 if turn = player0 and (! Player0.Action = none);
{indent(piles_evolution, " "*8)}
    end Evolution
end Agent"""

def get_agent_str(agent_name, actions_xo, protocol_xo):
    return f"""\
Agent {agent_name}
    Vars:
        null : boolean; -- for syntax reasons only
    end Vars
    Actions = {{{actions_xo}}};
    Protocol:
{indent(protocol_xo, " "*8)}
        Other : {{ none }}; -- technicality
    end Protocol
    Evolution:
        null=true if null=true;
    end Evolution
end Agent"""

def make_nim_specification(piles: list, history, player_to_move: int, formulae=None) -> str:
    if formulae is None:
        formulae = dedent("""\
            <player0> F player0wins;
            <player1> F player1wins;""")
    player_actions = ", ".join(generate_actions(piles))
    player_protocol_0 = "\n".join(generate_player_protocol(piles, 0))  # conditions on actions, the same for both players
    player_protocol_1 = "\n".join(generate_player_protocol(piles, 1))  # conditions on actions, the same for both players
    evaluation_conditions_0 = generate_evaluation_conditions_win(piles, 0)
    evaluation_conditions_1 = generate_evaluation_conditions_win(piles, 1)
    piles_init_conditions = generate_piles_init_conditions(piles, player_to_move, history)
    env_turn = "player0" if player_to_move == 0 else "player1"
    return f"""\
Semantics=SingleAssignment;

{get_env_str(piles)}

{get_agent_str("Player0", player_actions, player_protocol_0)}

{get_agent_str("Player1", player_actions, player_protocol_1)}

Evaluation
    player0wins if
{indent(evaluation_conditions_0, " "*4)}
    player1wins if
{indent(evaluation_conditions_1, " "*4)}
end Evaluation

InitStates
{indent(piles_init_conditions, " "*4)}
    and Environment.turn = {env_turn}
    and Player0.null = true and Player1.null = true;
end InitStates

Groups
    player0 = {{Player0}}; player1 = {{Player1}};
end Groups

Formulae
{indent(formulae, " "*4)}
end Formulae"""



class GameNim(GameInterface):
    def __init__(self, pile_sizes_str: str):
        """
        :param num_piles: Number of piles.
        :param piles: A string representing number of objects in a pile, in the format as in the example: "1;3;5;7".
        """
        self.pile_sizes_str = pile_sizes_str
        # self.pile_sizes = [int(x) for x in pile_sizes_str.split(';')]
        GameInterface.__init__(self, players={"player0": 0, "player1": 1})

    def get_name(self):
        return "nim"

    def load_game(self):
        # In combinatorial game theory, a misère game is one played according to the "misère play condition"; that is,
        # a player unable to move wins. This is in contrast to the "normal play condition" in which a player
        # unable to move loses.
        return pyspiel.load_game("nim", {"pile_sizes": self.pile_sizes_str, "is_misere": False})

    def formal_subproblem_description(self, game_state, history, formulae_to_check: str = None) -> str:
        game_state_desc = str(game_state)  # e.g.: '(0): 2 4 1'
        piles = [int(x) for x in game_state_desc.split(': ')[1].split(' ')]
        player_to_move = int(re.findall(r"\(\d+\)", str(game_state))[0][1])
        return make_nim_specification(piles, history, player_to_move, formulae_to_check)

    def termination_condition(self, history: str):
        """Determines when the branching of the game search space will conclude."""
        pass

    def get_moves_from_history(self, history: str) -> list[str]:
        """Converts a single history string to a list of successive actions."""
        if history == "":
            return []
        else:
            moves = history.split(';,')  # E.g. input to process: "pile:2, take:1;,pile:3, take:1;"
            for i, _ in enumerate(moves):
                if moves[i][-1] != ';':
                    moves[i] += ';'
            return moves

    def get_num_actions(self, history):
        return history.count(';')

    def get_default_formula_and_coalition(self):
        return "<player0> F player0wins;", {0}



def is_position_winning(piles: list) -> bool:
    pass