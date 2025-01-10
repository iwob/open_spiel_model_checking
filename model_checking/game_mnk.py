import re
from textwrap import dedent, indent
import pyspiel

def generate_table(m, n):
    table = []
    for i in range(1, n+ 1):
        row = []
        for j in range(1, m + 1):
            row.append(f"b{i}{j} : {{x, o, b}}")
        table.append(row)
    return table


def generate_conditions(m, n):
    conditions = []
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            condition_o = f"b{i}{j} = o if turn = nought and Nought.Action = a{i}{j};"
            condition_x = f"b{i}{j} = x if turn = cross  and Cross.Action  = a{i}{j};"
            conditions.append(condition_o)
            conditions.append(condition_x)
    return conditions


def generate_actions(m, n):
    actions = []

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            actions.append(f"a{i}{j}")

    actions.append("none")

    return actions


def generate_environment_conditions(m, n):
    conditions = []

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            condition = f"Environment.b{i}{j}=b:{{a{i}{j}}};"
            conditions.append(condition)

    return conditions


def generate_evaluation_conditions_win(m, n, k, player):
    row_conditions = []
    col_conditions = []
    diag_conditions = []

    for row in range(1, n + 1):
        for col in range(1, m - k + 2):
            row_conditions.append(" and ".join(f"Environment.b{row}{col + i} = {player}" for i in range(k)) + "\n")

    for col in range(1, m + 1):
        for row in range(1, n - k + 2):
            col_conditions.append(" and ".join(f"Environment.b{row + i}{col} = {player}" for i in range(k)) + "\n")

    for row in range(1, n - k + 2):
        for col in range(1, m - k + 2):
            diag_conditions.append(" and ".join(f"Environment.b{row + i}{col + i} = {player}" for i in range(k)) + "\n")

    for row in range(k, n + 1):
        for col in range(1, m - k + 2):
            diag_conditions.append(" and ".join(f"Environment.b{row - i}{col + i} = {player}" for i in range(k)) + "\n")

    all_conditions = row_conditions + col_conditions + diag_conditions
    return " or ".join(all_conditions)


def generate_board_condition(m, n, value, history):
    conditions = []
    pattern = r'[xo]\(\d+,\d+\)'
    if history:
        moves_list = re.findall(pattern, history)
    board = [[value for _ in range(m)] for _ in range(n)]

    if history:
        for move in moves_list:
            symbol = move[0]
            coords = move[2:-1].split(',')

            row, col = int(coords[0]), int(coords[1])
            board[row][col] = symbol

    for row in range(1, n + 1):
        for col in range(1, m + 1):
            conditions.append(f"Environment.b{row}{col} = {board[row - 1][col - 1]}")
        conditions[-1] += "\n"

    return " and ".join(conditions)


def get_env_str(m, n):
    board_obsvars = "\n".join(["; ".join(r) + ";" for r in generate_table(m, n)])
    board_move_conditions = "\n".join(generate_conditions(m, n))
    return dedent(f"""\
Agent Environment
    Obsvars:
        turn : {{nought, cross}};
{indent(board_obsvars, " "*8)}
    end Obsvars
    Actions = {{ }}; 
    Protocol: end Protocol
    Evolution:
        turn=nought if turn=cross; turn=cross if turn=nought;
{indent(board_move_conditions, " "*8)}
    end Evolution
end Agent""")

def get_agent_str(agent_name, actions_xo, protocol_xo):
    return dedent(f"""\
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
end Agent""")

def make_whole_board(m, n, k, history, formulae=None) -> str:
    if formulae is None:
        formulae = dedent("""\
            <cross> F (crosswins and ! noughtwins); -- TRUE
            <nought> F (noughtwins and ! crosswins); -- FALSE""")
    move = (history.count('o') + history.count('x')) % 2
    actions_xo = ", ".join(generate_actions(m, n))
    protocol_xo = "\n".join(generate_environment_conditions(m, n))  # conditions on actions, the same for both players
    evaluation_conditions_o = generate_evaluation_conditions_win(m, n, k, "o")
    evaluation_conditions_x = generate_evaluation_conditions_win(m, n, k, "x")
    board_init_conditions = generate_board_condition(m, n, "b", history)
    env_turn = "cross" if move == 0 else "nought"
    return f"""\
Semantics=SingleAssignment;

{get_env_str(m, n)}

{get_agent_str("Nought", actions_xo, protocol_xo)}

{get_agent_str("Cross", actions_xo, protocol_xo)}

Evaluation
    noughtwins if
{indent(evaluation_conditions_o, " "*4)};
    crosswins if
{indent(evaluation_conditions_x, " "*4)};
end Evaluation

InitStates
{indent(board_init_conditions, " "*4)}
    and Environment.turn = {env_turn}
    and Nought.null = true and Cross.null = true;
end InitStates

Groups
    nought = {{Nought}}; cross = {{Cross}};
end Groups

Formulae
{indent(formulae, " "*4)}
end Formulae"""



class GameInterface:
    def load_game(self):
        """Loads OpenSpiel game implementing the game."""
        pass

    def formal_subproblem_description(self, history, formulae_to_check=None) -> str:
        """Generates a formal description of a subproblem based on the current history of the game."""
        pass


class GameMnk(GameInterface):
    def __init__(self, m, n, k):
        self.m = m
        self.n = n
        self.k = k

    def load_game(self):
        return pyspiel.load_game("mnk", {"m": self.m, "n": self.n, "k": self.k})

    def formal_subproblem_description(self, history, formulae_to_check=None) -> str:
        return make_whole_board(self.m, self.n, self.k, history, formulae_to_check)

