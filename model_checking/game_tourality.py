import dataclasses
import re
from queue import Queue
from pathlib import Path
from textwrap import dedent, indent
import pyspiel
from game_mnk import GameInterface
from game_mcmas_model import GameInterfaceMcmasModel
from mcmas.parsers.ispl_parser import ISPLParser, StrategicFormula, ISPLModel, BooleanNot, Comparison
from mcmas_model_game import McmasModelGame, McmasModelState
from mcmas.parsers.ispl_parser import BooleanBinary

INDENT_SIZE = 6
FIELD_EMPTY = 0
FIELD_WALL = 1
FIELD_REWARD = 2

def clean_nl(text):
    if text[-1] == "\n":
        return text[:-1]
    else:
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
-- positions are updated according to the move
{player_position_update}
-- board and points are updated according to the moves of the players:
{rewards_deactivation}
{clean_nl(points_update)}
"""


def get_agent_spec(num: int, board: list, num_players: int, can_players_overlap: bool = False):
    size_x = len(board[0])
    size_y = len(board)

    def create_entry(bx, by, x, y, action):
        if board[y][x] == FIELD_WALL:
            return ""
        elif can_players_overlap:
            return f"Environment.turn=turn_p{num} and Environment.y_p{num}={y} and Environment.x_p{num}={x}: {{ {action} }};\n"
        else:
            player_collisions = []
            for i in range(num_players):
                if i != num:
                    player_collisions.append(f"(! (Environment.y_p{i}={by} and Environment.x_p{i}={bx}))")
            player_collisions_text = " and ".join(player_collisions)
            return f"Environment.turn=turn_p{num} and {player_collisions_text} and Environment.y_p{num}={y} and Environment.x_p{num}={x}: {{ {action} }};\n"
    agent_moves = ""
    for i in range(size_y):
        for j in range(size_x):
            if board[i][j] == FIELD_WALL:
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

def encode_raw(x):
    return x

def encode_visual(x: int):
    if x == FIELD_EMPTY:
        return "."
    elif x == FIELD_WALL:
        return "W"
    elif x == FIELD_REWARD:
        return "*"
    elif x >= 10:
        return x-10
    else:
        raise Exception(f"Unknown board element (value={x})")


def visualize_board(board: list, comment_markers: str = ""):
    text = ""
    for row in board:
        text += comment_markers
        for cell in row:
            text += str(encode_visual(cell)) + " "
        text += "\n"
    return text


def get_init_state(board: list, num_players: int, player_to_move: int):
    comment = "-- Game state:\n"
    comment += visualize_board(board, comment_markers="-- ")

    init_text = ""
    reward_id = 0
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if cell == FIELD_REWARD:
                init_text += f" Environment.xreward_{reward_id} = {j} and Environment.yreward_{reward_id} = {i} and Environment.reward_{reward_id} = avail and"
                reward_id += 1
            elif cell >= 10:
                init_text += f" Environment.x_p{cell - 10} = {j} and Environment.y_p{cell - 10} = {i} and"
    init_text += "\n"
    init_text += " and ".join([f"Environment.points_p{i} = 0" for i in range(num_players)]) + "\n"
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
        agents += get_agent_spec(i, board, num_players, can_players_overlap=can_players_overlap)
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

@dataclasses.dataclass
class TouralityLogicState:
    board: list
    player_positions: dict[int, tuple[int, int]]
    player_points: dict[int, int]
    turn: int
    num_players: int
    rewards_status: list

    def execute_actions(self, actions: list[str], env_variables):
        for p_id, a_name in enumerate(actions):
            if a_name == "pass":
                continue
            else:
                y, x = self.player_positions[p_id]
                self.board[y][x] = FIELD_EMPTY  # player leaves that spot

                if a_name == "up":
                    new_spot = y-1, x
                elif a_name == "down":
                    new_spot = y+1, x
                elif a_name == "left":
                    new_spot = y, x - 1
                elif a_name == "right":
                    new_spot = y, x + 1
                else:
                    raise Exception(f"Unknown action {a_name}")

                env_variables[f"y_p{p_id}"] = new_spot[0]
                env_variables[f"x_p{p_id}"] = new_spot[1]

                if self.board[new_spot[0]][new_spot[1]] == FIELD_REWARD:
                    self.player_points[p_id] += 1
                    env_variables[f"points_p{p_id}"] = self.player_points[p_id]
                    for i, (r_id, r_x, r_y) in enumerate(self.rewards_status):
                        if r_y == new_spot[0] and r_x == new_spot[1]:
                            print("Reward taken")
                            env_variables[f"reward_{r_id}"] = "taken"
                            del self.rewards_status[i]
                            break

                    # env_variables[f"reward_p{p_id}"] = self.player_points[p_id]
                self.board[new_spot[0]][new_spot[1]] = 10 + p_id

                self.turn = (self.turn + 1) % self.num_players
                env_variables["turn"] = f"turn_p{self.turn}"

    def _legal_actions(self, player):
        """Returns a list of legal actions, sorted in ascending order. In simultaneous games
         possible actions for each player are generated using function."""
        super._legal_actions(player)



class TouralityGame(McmasModelGame):
    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return TouralityState(game=self,
                              model=self.spec,
                              formula=self.formula,
                              silent=self.silent)


class TouralityState(McmasModelState):
    COUNTER = 0
    def __init__(self, game: TouralityGame, model: ISPLModel, formula:StrategicFormula, seed=None, silent=True):
        super().__init__(game, model, formula, seed=seed, silent=silent)
        self.logic = self.reconstruct_board()
        # print("Board initialized")
        # print(str(self))
        # TouralityState.COUNTER += 1
        # print("Counter: ", TouralityState.COUNTER)

    def __str__(self):
        text = "; ".join([f"{a.name}: {self.logic.player_points[i]}" for i, a in enumerate(self.model.agents)]) + "\n"
        text += visualize_board(self.logic.board)
        return text

    def reconstruct_board(self) -> TouralityLogicState:
        player_points = {}
        turn = None
        vars_players_x = []
        vars_players_y = []
        vars_rewards_status = []
        vars_rewards_x = []
        vars_rewards_y = []
        vars_points = []
        for v, value in self.env_variables.items():
            if v.startswith("reward"):
                vars_rewards_status.append((v, value))
            elif v.startswith("xreward"):
                vars_rewards_x.append((v, value))
            elif v.startswith("yreward"):
                vars_rewards_y.append((v, value))
            elif v.startswith("x_p"):
                vars_players_x.append((v, value))
            elif v.startswith("y_p"):
                vars_players_y.append((v, value))
            elif v == "turn":
                # value = turn_p0
                turn = int(value[6:])
            else:
                # v = points_p[...]
                vars_points.append((v, value))

        num_rows = next(ov for ov in self.model.environment.observable_vars if ov.name == "y_p0").upper + 1
        num_cols = next(ov for ov in self.model.environment.observable_vars if ov.name == "x_p0").upper + 1
        board = [[FIELD_WALL for j in range(num_cols)] for i in range(num_rows)]

        # Extraction of free fields via player possible positions checked for in the protocol function.
        # This needs to be done, because for efficiency reasons the board is not represented explicitly.
        for rule in self.game.spec.agents[0].protocol.rules:
            q = Queue()
            q.put(rule.condition.left)
            q.put(rule.condition.right)
            while not q.empty():
                item = q.get()
                if isinstance(item, BooleanBinary):
                    if not isinstance(item.left, Comparison):
                        q.put(item.left)
                    if not isinstance(item.right, Comparison):
                        q.put(item.right)
                elif isinstance(item, BooleanNot):
                    xy1 = (item.operand.left.left.name, int(item.operand.left.right.value))
                    xy2 = (item.operand.right.left.name, int(item.operand.right.right.value))
                    x, y = None, None
                    if xy1[0].startswith("x_"):
                        x = xy1[1]
                    else:
                        y = xy1[1]
                    if xy2[0].startswith("x_"):
                        x = xy2[1]
                    else:
                        y = xy2[1]
                    # A field with coordinates (x,y) is definitely free
                    board[y][x] = FIELD_EMPTY
                    break  # We have found our negation in the formula, now we can move on to processing new rules

        num_players = len(vars_players_x)

        # Adding rewards to the board
        for x, y in zip(sorted(vars_rewards_x), sorted(vars_rewards_y)):
            board[y[1]][x[1]] = FIELD_REWARD

        # Setting numbers of points for player
        for p_name, value in vars_points:
            # points_p1
            p_id = int(p_name[8:])
            player_points[p_id] = value

        # Putting players on board
        player_positions = {}
        for x, y in zip(sorted(vars_players_x), sorted(vars_players_y)):
            p_id = int(x[0][3:])
            board[y[1]][x[1]] = 10 + p_id
            player_positions[p_id] = (y[1], x[1])

        # Rewards status
        rewards_status = []
        for r_status, r_y, r_x in zip(sorted(vars_rewards_status), sorted(vars_rewards_y), sorted(vars_rewards_x)):
            if r_status[1] == "avail":
                r_id = int(r_status[0][7:])  # e.g.: reward_1
                rewards_status.append((r_id, r_y[1], r_x[1]))
        return TouralityLogicState(board, player_positions, player_points, turn, num_players, rewards_status)


    def _execute_agent_actions(self, actions):
        agent_actions = [self.get_action_name(a) for a in actions]
        self.logic.execute_actions(agent_actions, self.env_variables)
        return False






class GameTourality(GameInterfaceMcmasModel):
    def __init__(self, model_path: str):
        """
        :param board: A 2D array describing an initial state of the board. Convention:
        - 0: empty field
        - 1: wall
        - 2: token to be collected
        - 1x: initial position of the player x
        """
        GameInterfaceMcmasModel.__init__(self, model_path)

    def get_name(self):
        return "tourality"

    def load_game(self) -> pyspiel.Game:
        params = {"spec": self.model, "formula": self.formula}
        return TouralityGame(params)

    @classmethod
    def get_default_formula_and_coalition(cls):
        return "<Player0> F player0wins;", {0}



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
