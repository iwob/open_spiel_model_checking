import re
from textwrap import dedent, indent
import pyspiel
from game_mnk import GameInterface


def get_nim_sum(piles: list[int]) -> int:
    res = 0
    for pile in piles:
        res = res ^ pile
    return res


def is_position_winning(piles: list[int]) -> bool:
    # To win the game of Nim, your goal is to always leave your opponent with a "balanced" state (a Nim-sum of zero).
    # You can guarantee a win by removing objects so that the exclusive OR (XOR) sum of all the pile sizes equals zero.
    return get_nim_sum(piles) != 0


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

    if add_comment:
        comment  = f"--  History: {history}\n"
        comment += f"--  Game state:\n"
        comment += f"--  ({player_to_move}): {' '.join([str(x) for x in piles])}\n"
        comment += f"--  Is a winning position: {is_position_winning(piles)}\n"
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
    obsvars = "\n".join([f"pile{i+1} : 0 .. {pile};" for i, pile in enumerate(piles)])
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

def make_tourality_specification(num_players: int, board: list, history, player_to_move: int, formulae: str) -> str:
    env_obsvars_turn_vals = "{" + ", ".join([f"turn_{i}" for i in range(num_players)]) + "}"
    env_obsvars_points_vars = ", ".join([f"points_{i}" for i in range(num_players)]) + "}"
    num_rewards = sum([row.count(2) for row in board])
    player_actions = ", ".join(generate_actions(piles))
    player_protocol_0 = "\n".join(generate_player_protocol(piles, 0))  # conditions on actions, the same for both players
    player_protocol_1 = "\n".join(generate_player_protocol(piles, 1))  # conditions on actions, the same for both players
    evaluation_conditions_0 = generate_evaluation_conditions_win(piles, 0)
    evaluation_conditions_1 = generate_evaluation_conditions_win(piles, 1)
    piles_init_conditions = generate_piles_init_conditions(piles, player_to_move, history)
    env_turn = "player0" if player_to_move == 0 else "player1"
    return f"""\
Semantics=SingleAssignment;

Agent Environment
Obsvars:
    turn : {env_obsvars_turn_vals}; xred, yred, xblu, yblu : 1..8;
    reward[1..{num_rewards}] : {{avail, taken}};
    {env_obsvars_points_vars}: 0..{num_rewards};
    constant b[1..8][1..8] : {empty, block}; -- the board
    constant xreward[1..5], yreward[5]: [1..5]; -- positions of the rewards
end Obsvars
Evolution:
    -- turn switches between every two moves
    turn=red if turn=blu; turn=blu if turn=red;
    -- positions are updated according to the move
    yred=yred-1 if turn = red & Red.Action=up;
    yred=yred+1 if turn = red & Red.Action=dn;
    -- and similar for other actions and player Blu;
    -- board and points are updated according to the moves of the players:
    for 1=1..5: reward[i] = taken if reward[i] = avail &
    (turn = blu & xred = xreward[i] & yred = yreward[i] |
    turn = red & xblu = xreward[i] & yblu = yreward[i]);
    for 1=1..5: points_red=points_red+1 if reward[i] = avail &
    turn = blu & xred = xreward[i] & yred = yreward[i];
    for 1=1..5: points_blu=points_blu+1 if reward[i] = avail &
    turn = red & xblu = xreward[i] & yblu = yreward[i];
end Evolution
end Agent

Agent Red
Actions = { up, dn, lt, rt };
Protocol:
    -- if it is red’s turn and at position i,j and target field is not blocked
    -- and target field is not occupied, then the movement action is available
    for some x=1..8: for some y=1..7:
    xred=x & yred=y & b[x,y+1]=empty & !(xblu=x & yblu=y) : { dn };
    for some x=1..8: for some y=2..8:
    xred=x & yred=y & b[x,y-1]=empty & !(xblu=x & yblu=y) : { up };
    for some x=1..7: for some y=1..8:
    xred=x & yred=y & b[x+1,y]=empty & !(xblu=x & yblu=y) : { rt };
    for some x=2..8: for some y=1..8:
    xred=x & yred=y & b[x-1,y]=empty & !(xblu=x & yblu=y) : { rt };
end Protocol
end Agent

Agent Blu
-- similar
end Agent

InitStates
    b = [[empty, empty, block, empty, block, empty, empty, empty], ...] &
    xreward=[3,2,5,4,3], yreward=[2,5,7,1,4] &
    xred = 1 & yred = 1 & xblu = 8 & yblu = 8 & turn = red &
    for i=1..5: reward[i] = avail & points_red = 0 & points_blu = 0;
end InitStates
Formulae
    <Red> F (points_red >= 3); -- has Red a winning strategy?
end Formulae

"""



class GameTourality(GameInterface):
    def __init__(self, board: list):
        """
        :param board: A 2D array describing an initial state of the board. Convention:
        - 0: empty field
        - 1: wall
        - 2: token to be collected
        - 1x: initial position of the player x
        """
        # Board convention:
        #
        self.pile_sizes_str = pile_sizes_str
        # self.pile_sizes = [int(x) for x in pile_sizes_str.split(';')]
        GameInterface.__init__(self, players={"player0": 0, "player1": 1})

    def get_name(self):
        return "tourality"

    def load_game(self):
        # In combinatorial game theory, a misère game is one played according to the "misère play condition"; that is,
        # a player unable to move wins. This is in contrast to the "normal play condition" in which a player
        # unable to move loses.
        return pyspiel.load_game("tourality", {"pile_sizes": self.pile_sizes_str, "is_misere": False})

    def formal_subproblem_description(self, game_state, history, formulae_to_check: str = None) -> str:
        if formulae_to_check is None:
            formulae_to_check, _ = self.get_default_formula_and_coalition()
        if isinstance(history, list):
            history = ",".join(history)
        game_state_desc = str(game_state)  # e.g.: '(0): 2 4 1'
        piles = [int(x) for x in game_state_desc.split(': ')[1].split(' ')]
        player_to_move = int(re.findall(r"\(\d+\)", str(game_state))[0][1])
        return make_nim_specification(piles, history, player_to_move, formulae_to_check)

    def termination_condition(self, history: str):
        """Determines when the branching of the game search space will conclude."""
        pass

    def get_moves_from_history_str(self, history: str) -> list[str]:
        """Converts a single history string to a list of successive actions."""
        if history == "":
            return []
        else:
            moves = history.split(';,')  # E.g. input to process: "pile:2, take:1;,pile:3, take:1;"
            for i, _ in enumerate(moves):
                if moves[i][-1] != ';':
                    moves[i] += ';'
            return moves

    @classmethod
    def get_default_formula_and_coalition(cls):
        return "<player0> F player0wins;", {0}




if __name__ == "__main__":
    board = [
        # 1  2  3  4  5  6  7  8
        [10, 0, 0, 0, 0, 0, 2, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 1, 0, 0, 2, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 1, 2, 0],
        [ 0, 2, 0, 0, 0, 1, 0, 0],
        [ 0, 0, 1, 0, 0, 2, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 11],
    ]
    thr = sum([row.count(2) for row in board]) / 2 + 1
    make_tourality_specification(board, None, player_to_move=0, formulae=f"<Player0> F (points_0 >= {thr});")