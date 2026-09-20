import re
from textwrap import dedent, indent
import pyspiel
from game_mnk import GameInterface

INDENT_SIZE = 6


def clean_nl(text):
    if text[-1] == "\n":
        return text[:-1]
    else:
        return text

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


def get_env_evolution(board: list, num_players: int, num_rewards: int, can_players_overlap: bool):
    size_x = len(board[0])
    size_y = len(board)

    turn_switcher = " ".join([f"turn=turn_p{i} if turn=turn_p{(i-1) % num_players};" for i in range(num_players)])
    player_position_update = ""
    for i in range(num_players):
        player_position_update += f"y_p{i} = y_p{i} - 1 if turn = turn_p{i} and Player{i}.Action = up;\n"
        player_position_update += f"y_p{i} = y_p{i} + 1 if turn = turn_p{i} and Player{i}.Action = down;\n"
        player_position_update += f"x_p{i} = x_p{i} - 1 if turn = turn_p{i} and Player{i}.Action = left;\n"
        player_position_update += f"x_p{i} = x_p{i} + 1 if turn = turn_p{i} and Player{i}.Action = right;\n"

    def create_entry(bx, by, x, y, action):
        text = ""
        for p in range(num_players):
            text += f"b_{by}_{bx} = block if y_p{p} = {y} and x_p{p} = {x} and turn = turn_p{p} and Player{p}.Action = {action};\n"
            text += f"b_{y}_{x} = empty if y_p{p} = {y} and x_p{p} = {x} and turn = turn_p{p} and Player{p}.Action = {action};\n"
        return text
    board_update = ""
    if not can_players_overlap:
        for i in range(size_y):
            for j in range(size_x):
                if board[i][j] == 1:
                    continue  # because agent cannot ever be in a field with a wall
                if i > 0:
                    board_update += create_entry(j, i, j, i - 1, action="down")
                if i < size_y - 1:
                    board_update += create_entry(j, i, j, i + 1, action="up")
                if j > 0:
                    board_update += create_entry(j, i, j - 1, i, action="right")
                if j < size_x - 1:
                    board_update += create_entry(j, i, j + 1, i, action="left")

    rewards_deactivation = ""
    for i in range(num_rewards):
        rewards_deactivation += f"reward_{i} = taken if reward_{i} = avail and ("
        rewards_deactivation += " or ".join([f"(turn = turn_p{(j+1) %  num_players} and x_p{j} = xreward_{i} and y_p{j} = yreward_{i})" for j in range(num_players)])
        rewards_deactivation += ");\n"

    points_update = ""
    for i in range(num_rewards):
        for j in range(num_players):
            points_update += f"points_p{j} = points_p{j} + 1 if reward_{i} = avail and " +\
             f"turn = turn_p{(j+1) % num_players} and x_p{j} = xreward_{i} and y_p{j} = yreward_{i};\n"

    return f"""-- turn switching
{turn_switcher}
-- board update (if agents cannot overlap)
{board_update}
-- positions are updated according to the move
{player_position_update}
-- board and points are updated according to the moves of the players:
{rewards_deactivation}
{clean_nl(points_update)}
"""


def get_agent_spec(num: int, board: list, can_players_overlap: bool = False):
    size_x = len(board[0])
    size_y = len(board)

    def create_entry(bx, by, x, y, action):
        if board[y][x] == 1:
            return ""
        elif can_players_overlap:
            return f"Environment.y_p{num}={y} and Environment.x_p{num}={x}: {{ {action} }};\n"
        else:
            return f"Environment.b_{by}_{bx} = empty and Environment.y_p{num}={y} and Environment.x_p{num}={x}: {{ {action} }};\n"
    agent_moves = ""
    for i in range(size_y):
        for j in range(size_x):
            if board[i][j] == 1:
                continue  # because agent cannot ever be in a field with a wall
            if i > 0:
                agent_moves += create_entry(j, i, j, i-1, action="down")
            if i < size_y - 1:
                agent_moves += create_entry(j, i, j, i+1, action="up")
            if j > 0:
                agent_moves += create_entry(j, i, j-1, i, action="right")
            if j < size_x - 1:
                agent_moves += create_entry(j, i, j+1, i, action="left")
    agent_moves += "Other : { pass };"
    return f"""Agent Player{num}
Vars:
{indent("null : boolean; -- for syntax reasons only", " " * INDENT_SIZE)}
end Vars
Actions = {{ up, down, left, right, pass }};
Protocol:
{indent(clean_nl(agent_moves), " " * INDENT_SIZE)}
end Protocol
Evolution:
{indent("null=true if null=true;", " " * INDENT_SIZE)}
end Evolution
end Agent\n"""


def get_init_state(board: list, num_players: int, player_to_move: int):
    def encode_raw(x):
        return x
    def encode_visual(x):
        if x == 0:
            return "."
        elif x == 1:
            return "W"
        elif x == 2:
            return "*"
        elif x >= 10:
            return x-10
        else:
            raise Exception(f"Unknown board element (value={x})")

    comment = "-- Game state:\n"
    for row in board:
        comment += "-- "
        for cell in row:
            comment += str(encode_visual(cell)) + " "
        comment += "\n"

    init_text = ""
    reward_id = 0
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if j > 0:
                init_text += " and "
            if cell < 10:
                init_text += f"Environment.b_{i}_{j} = empty"
                if cell == 2:  # reward fields
                    init_text += f" and Environment.xreward_{reward_id} = {j} and Environment.yreward_{reward_id} = {i} and Environment.reward_{reward_id} = avail"
                    reward_id += 1
            else:
                init_text += f"Environment.b_{i}_{j} = block and Environment.x_p{cell - 10} = {j} and Environment.y_p{cell - 10} = {i}"
        if i < len(board) - 1:
            init_text += " and "
        init_text += "\n"
    init_text += " and " + " and ".join([f"Environment.points_p{i} = 0" for i in range(num_players)]) + "\n"
    init_text += f" and Environment.turn = turn_p{player_to_move};"
    return comment + init_text


def get_evaluation(num_players: int, num_rewards: int, additional_evaluations: str):
    thr = num_rewards / 2 + 1
    text = ""
    for i in range(num_players):
        text += f"player{i}wins if Environment.points_p{i} >= {int(thr)};\n"
    return text + additional_evaluations

def make_tourality_specification(board: list, history, player_to_move: int, formulae: str,
                                 can_players_overlap: bool=False, additional_evaluations: str = "") -> str:
    size_x = len(board[0])
    size_y = len(board)
    num_rewards = sum([row.count(2) for row in board])
    num_players = sum([cell >= 10 for row in board for cell in row])
    env_evolution = get_env_evolution(board, num_players, num_rewards, can_players_overlap=can_players_overlap)
    init_state = get_init_state(board, num_players, player_to_move)

    env_vars = "turn: {" + ", ".join([f"turn_p{i}" for i in range(num_players)]) + "};\n"
    env_vars += " ".join([f"x_p{i} : 0..{size_x-1};" for i in range(num_players)]) + "\n"
    env_vars += " ".join([f"y_p{i} : 0..{size_y-1};" for i in range(num_players)]) + "\n"
    env_vars += " ".join([f"points_p{i} : 0..{num_rewards};" for i in range(num_players)]) + "\n"
    for i in range(size_y):
        env_vars += " ".join([f"b_{i}_{j} : {{empty, block}};" for j in range(size_x)]) + "\n"
    for i in range(num_rewards):
        env_vars += f"reward_{i} : {{ avail, taken }}; "
        env_vars += f"xreward_{i} : 0..{size_x-1}; "
        env_vars += f"yreward_{i} : 0..{size_y-1};"
        if i < num_rewards - 1:
            env_vars += "\n"

    evaluation = clean_nl(get_evaluation(num_players, num_rewards, additional_evaluations=additional_evaluations))


    groups = " ".join([f"Player{i} = {{Player{i}}};" for i in range(num_players)])
    groups += "\nAll = {" + ", ".join([f"Player{i}" for i in range(num_players)]) + "};"

    agents = ""
    for i in range(num_players):
        agents += get_agent_spec(i, board, can_players_overlap=can_players_overlap)
        if i < num_players - 1:
            agents += "\n"

    return f"""\
Semantics=SingleAssignment;

Agent Environment
Obsvars:
{indent(env_vars, " " * INDENT_SIZE)}
end Obsvars
Actions = {{ }};
Protocol: end Protocol
Evolution:
{indent(clean_nl(env_evolution), " " * INDENT_SIZE)}
end Evolution
end Agent

{agents}

Evaluation
{indent(evaluation, " " * INDENT_SIZE)}
end Evaluation

InitStates
{indent(init_state, " " * INDENT_SIZE)}
end InitStates

Groups
{indent(groups, " " * INDENT_SIZE)}
end Groups

Formulae
{indent(formulae, " " * INDENT_SIZE)}
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
    with open("example_specifications/tourality/schlingloff_1.ispl", "w") as f:
        f.write(make_tourality_specification(board, history=None, player_to_move=0, formulae=f"<Player0> F (player0wins);"))
    with open("example_specifications/tourality/schlingloff_1_overlap.ispl", "w") as f:
        f.write(make_tourality_specification(board, history=None, player_to_move=0, can_players_overlap=True, formulae=f"<Player0> F (player0wins);"))

    board = [
        # 1  2  3  4  5  6  7  8
        [10, 0, 0, 0, 2, 0, 0, 0],
        [ 0, 2, 0, 1, 1, 0, 0, 0],
        [ 1, 0, 0, 0, 1, 0, 0, 2],
        [ 2, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 2, 0, 0, 0, 1, 1],
        [ 1, 2, 1, 0, 2, 0, 0, 0],
        [ 0, 0, 1, 2, 0, 0, 2, 1],
        [ 0, 0, 0, 0, 0, 1, 0, 11],
    ]
    with open("example_specifications/tourality/schlingloff_2.ispl", "w") as f:
        f.write(make_tourality_specification(board, history=None, player_to_move=0, formulae=f"<Player0> F (player0wins);"))
    with open("example_specifications/tourality/schlingloff_2_overlap.ispl", "w") as f:
        f.write(make_tourality_specification(board, history=None, player_to_move=0, can_players_overlap=True, formulae=f"<Player0> F (player0wins);"))

    board = [
        # 1  2  3  4  5  6  7  8
        [10, 1, 0, 0, 2, 1, 1, 2],
        [ 0, 1, 0, 1, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 1, 1, 1, 0],
        [ 1, 1, 1, 0, 0, 0, 0, 2],
        [ 1, 1, 0, 2, 0, 1, 1, 1],
        [ 1, 0, 0, 1, 0, 0, 1, 1],
        [ 1, 0, 1, 1, 1, 0, 0, 1],
        [ 2, 0, 1, 1, 1, 1, 0, 11],
    ]
    with open("example_specifications/tourality/schlingloff_3.ispl", "w") as f:
        f.write(make_tourality_specification(board, history=None, player_to_move=0, formulae=f"<Player0> F (player0wins);"))
    with open("example_specifications/tourality/schlingloff_3_overlap.ispl", "w") as f:
        f.write(make_tourality_specification(board, history=None, player_to_move=0, can_players_overlap=True, formulae=f"<Player0> F (player0wins);"))

    board = [
        [2, 0, 0, 10, 11, 0, 2, 2],
    ]
    with open("example_specifications/tourality/simple_01.ispl", "w") as f:
        f.write(make_tourality_specification(board, history=None, player_to_move=0, formulae=f"<Player0> F (player0wins);\n<Player1> F (player1wins);\n<All> F (player0wins);"))

    board = [
        [2, 0, 0, 11, 10, 0, 2, 2],
    ]
    with open("example_specifications/tourality/simple_02.ispl", "w") as f:
        f.write(make_tourality_specification(board, history=None, player_to_move=0, formulae=f"<Player0> F (player0wins);\n<Player1> F (player1wins);\n<All> F (player0wins);"))


    board = [
        [2, 0, 0, 10, 11, 0, 2, 2],
        [1, 1, 1, 1, 0, 0, 0, 0 ]
    ]
    with open("example_specifications/tourality/simple_03.ispl", "w") as f:
        f.write(make_tourality_specification(board, history=None, player_to_move=0, formulae=f"<Player0> F (player0wins);\n<Player1> F (player1wins);\n<All> F (player0wins);"))

    board = [
        [2, 11, 10]
    ]
    with open("example_specifications/tourality/degenerate_01.ispl", "w") as f:
        f.write(make_tourality_specification(board, history=None, player_to_move=0, formulae=f"<Player0> F (player0wins);\n<Player1> F (player1wins);\n<All> F (player0wins);"))

    board = [
        [2, 10, 1, 11]
    ]
    with open("example_specifications/tourality/degenerate_02.ispl", "w") as f:
        f.write(make_tourality_specification(board, history=None, player_to_move=0, formulae=f"<Player0> F (player0wins);\n<Player1> F (player1wins);\n<All> F (player0wins);"))

    board = [
        [2, 10, 1, 11, 0]
    ]
    with open("example_specifications/tourality/degenerate_03.ispl", "w") as f:
        f.write(make_tourality_specification(board, history=None, player_to_move=0, formulae=f"<Player0> F (player0wins);\n<Player1> F (player1wins);\n<All> F (player0wins);"))

    board = [
        [11, 10, 2, 12, 0]
    ]
    with open("example_specifications/tourality/degenerate_04.ispl", "w") as f:
        f.write(make_tourality_specification(board, history=None, player_to_move=0, formulae=f"<Player0> F (player0wins);\n<Player1> F (player1wins);\n<All> F (player0wins);"))


    board = [
        [11],
        [2],
        [10],
        [0],
    ]
    with open("example_specifications/tourality/degenerate_05.ispl", "w") as f:
        f.write(make_tourality_specification(board, history=None, player_to_move=0, formulae=f"<Player0> F (player0wins);\n<Player1> F (player1wins);\n<All> F (player1wins);"))
