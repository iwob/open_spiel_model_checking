import re
from textwrap import indent
import pyspiel

GAME_TREE_LEAF_ID = "SUBMODEL"
GAME_TREE_CUR_PLAYER_ID = "CUR_PLAYER"
RESERVED_TREE_IDS = {GAME_TREE_LEAF_ID, GAME_TREE_CUR_PLAYER_ID}


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


def generate_player_protocol(m, n):
    conditions = []
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            condition = f"Environment.b{i}{j}=b: {{a{i}{j}}};"
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
    return " or ".join(all_conditions)[:-1]


def generate_board_condition(m, n, value, history, add_comment=True):
    conditions = []
    pattern = r'[xo]\(\d+,\d+\)'
    if history is not None and isinstance(history, str):
        moves_list = re.findall(pattern, history)
    else:
        moves_list = history
    board = [[value for _ in range(m)] for _ in range(n)]

    if history is not None:
        for move in moves_list:
            symbol = move[0]
            coords = move[2:-1].split(',')
            row, col = int(coords[0]), int(coords[1])
            board[row][col] = symbol

    for row in range(1, n + 1):
        for col in range(1, m + 1):
            conditions.append(f"Environment.b{row}{col} = {board[row - 1][col - 1]}")
        conditions[-1] += "\n"

    text = " and ".join(conditions)[:-1]
    if history is not None and add_comment:
        comment  = f"--  History: {history}\n"
        comment += f"--  Game state:\n"
        comment += indent("\n".join(["".join(r) for r in board])+"\n", "--  ").replace('b', '.')
        return comment + text
    else:
        return text


def get_env_str(m, n):
    board_obsvars = "\n".join(["; ".join(r) + ";" for r in generate_table(m, n)])
    board_move_conditions = "\n".join(generate_conditions(m, n))
    return f"""\
Agent Environment
    Obsvars:
        turn : {{nought, cross}};
{indent(board_obsvars, " "*8)}
    end Obsvars
    Actions = {{ }}; 
    Protocol: end Protocol
    Evolution:
        turn=nought if turn=cross and (! Cross.Action = none);
        turn=cross if turn=nought and (! Nought.Action = none);
{indent(board_move_conditions, " "*8)}
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

def make_whole_board(m, n, k, history, formulae) -> str:
    move = (history.count('o') + history.count('x')) % 2
    actions_xo = ", ".join(generate_actions(m, n))
    protocol_xo = "\n".join(generate_player_protocol(m, n))  # conditions on actions, the same for both players
    evaluation_conditions_o = generate_evaluation_conditions_win(m, n, k, "o")
    evaluation_conditions_x = generate_evaluation_conditions_win(m, n, k, "x")
    board_init_conditions = generate_board_condition(m, n, "b", history)
    env_turn = "cross" if move == 0 else "nought"
    return f"""\
Semantics=SingleAssignment;

{get_env_str(m, n)}

{get_agent_str("Cross", actions_xo, protocol_xo)}

{get_agent_str("Nought", actions_xo, protocol_xo)}

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


def translate_action(name):
    """Translates action name from that used by OpenSpiel to the one used in the specification."""
    # Example name as used by OpenSpiel: (0,1)
    x = name[2:-1].split(',')
    return f"a{int(x[0])+1}{int(x[1])+1}"

def translate_player(player_id, capitalize_first=False):
    if player_id == 0:
        return "Cross" if capitalize_first else "cross"
    else:
        return "Nought" if capitalize_first else "nought"

def get_additional_env_evolution_items(tree, conditions_rhs_list, results):
    if GAME_TREE_LEAF_ID in tree:
        return
    else:
        for a in tree:
            if a not in RESERVED_TREE_IDS:
                move_no = len(conditions_rhs_list) + 1
                act = translate_action(a)
                p1 = translate_player(tree[GAME_TREE_CUR_PLAYER_ID])
                p1c = translate_player(tree[GAME_TREE_CUR_PLAYER_ID], capitalize_first=True)
                lhs = f"move_{move_no} = {act}"
                rhs = f"turn = {p1} and {p1c}.Action = {act} and move_{move_no} = none"
                if len(conditions_rhs_list) > 0:
                    rhs += f" and {' and '.join(conditions_rhs_list)}"
                item = f"{lhs} if {rhs};"
                results.append(item)

                if GAME_TREE_LEAF_ID in tree[a]:
                    item2 = f"moves_frozen = false if {rhs};"
                    results.append(item2)
                conditions_rhs_list.append(lhs)
                get_additional_env_evolution_items(tree[a], conditions_rhs_list, results)
                del conditions_rhs_list[-1]


def get_move_variables_text(depth, player_actions):
    text = ""
    for i in range(1, depth):  # |actions| = depth - 1
        text += f"move_{i} : {{{player_actions}}};\n"
    return text[:-1]

def get_env_str_game_tree(m, n, depth, player_actions, game_tree):
    evo_items = []
    get_additional_env_evolution_items(game_tree, [], evo_items)
    additional_items = "\n".join(evo_items)
    board_obsvars = "\n".join(["; ".join(r) + ";" for r in generate_table(m, n)])
    board_move_conditions = "\n".join(generate_conditions(m, n))
    return f"""\
Agent Environment
    Obsvars:
        moves_frozen : boolean;
{indent(get_move_variables_text(depth, player_actions), " "*8)}
        turn : {{nought, cross}};
{indent(board_obsvars, " "*8)}
    end Obsvars
    Actions = {{ }}; 
    Protocol: end Protocol
    Evolution:
        turn=nought if turn=cross and (! Cross.Action = none);
        turn=cross if turn=nought and (! Nought.Action = none);
{indent(additional_items, " "*8)}
{indent(board_move_conditions, " "*8)}
    end Evolution
end Agent"""

def compute_tree_depth(game_tree, depth):
    if GAME_TREE_LEAF_ID in game_tree:
        return depth
    else:
        return max([compute_tree_depth(game_tree[a], depth+1) for a in game_tree if a not in RESERVED_TREE_IDS])

def generate_player_protocol_game_tree(m, n, player_id, game_tree):
    conditions = []
    # Forcing player moves as defined in the tree
    def traverse_tree(tree, conditions_lhs_list):
        move_no = len(conditions_lhs_list)
        if GAME_TREE_LEAF_ID in tree:
            return
        else:
            if tree[GAME_TREE_CUR_PLAYER_ID] == player_id:
                lhs = " and ".join(conditions_lhs_list) + f" and Environment.move_{move_no} = none"
                rhs = "{" + ",".join([translate_action(a) for a in tree if a not in RESERVED_TREE_IDS]) + "}"
                conditions.append(f"{lhs}: {rhs};")
            for a in tree:
                if a not in RESERVED_TREE_IDS:
                    conditions_lhs_list.append(f"Environment.move_{move_no} = {translate_action(a)}")
                    traverse_tree(tree[a], conditions_lhs_list)
                    del conditions_lhs_list[-1]

    traverse_tree(game_tree, ["Environment.moves_frozen = true"])

    # Unrestricted actions according to the rules
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if game_tree is None:
                condition = f"Environment.b{i}{j}=b: {{a{i}{j}}};"
            else:
                condition = f"Environment.moves_frozen = false and Environment.b{i}{j}=b: {{a{i}{j}}};"
            conditions.append(condition)
    return conditions


def make_whole_board_game_tree(m, n, k, game_tree, history, formulae) -> str:
    depth = compute_tree_depth(game_tree, 1)
    move = (history.count('o') + history.count('x')) % 2
    actions_xo = ", ".join(generate_actions(m, n))
    protocol_x = "\n".join(generate_player_protocol_game_tree(m, n, 0, game_tree))
    protocol_o = "\n".join(generate_player_protocol_game_tree(m, n, 1, game_tree))
    evaluation_conditions_o = generate_evaluation_conditions_win(m, n, k, "o")
    evaluation_conditions_x = generate_evaluation_conditions_win(m, n, k, "x")
    board_init_conditions = generate_board_condition(m, n, "b", history)
    env_turn = "cross" if move == 0 else "nought"
    return f"""\
Semantics=SingleAssignment;

{get_env_str_game_tree(m, n, depth, actions_xo, game_tree)}

{get_agent_str("Cross", actions_xo, protocol_x)}

{get_agent_str("Nought", actions_xo, protocol_o)}

Evaluation
    noughtwins if
{indent(evaluation_conditions_o, " "*4)};
    crosswins if
{indent(evaluation_conditions_x, " "*4)};
end Evaluation

InitStates
{indent(board_init_conditions, " "*4)}
    and Environment.turn = {env_turn}
    and Nought.null = true and Cross.null = true
    and {' and '.join([f'Environment.move_{i} = none' for i in range(1, depth)])}
    and Environment.moves_frozen = true;
end InitStates

Groups
    nought = {{Nought}}; cross = {{Cross}};
end Groups

Formulae
{indent(formulae, " "*4)}
end Formulae"""




class GameInterface:
    """The primary goal of GameInterface is to connect the world of OpenSpiel to the world of symbolic solvers
    (e.g., MCMAS). This task includes:
    - generation of a symbolic board specification in the solver's domain specific language (DSL),
    - creation of a game object given parameters,
    - parsing strings of actions representing history of moves in the game,
    - provision of the default formula and coalition.

    Certain assumptions are good to keep when implementing new game interfaces:
    - each action available in the game should correspond to appropriately named action in the game specification. """

    def __init__(self, players):
        self.players = players

    def get_name(self):
        """Returns a name of the game."""
        raise Exception("Not implemented!")

    def load_game(self) -> pyspiel.Game:
        """Loads OpenSpiel game implementing the game."""
        raise Exception("Not implemented!")

    def load_game_as_turn_game(self) -> pyspiel.Game:
        """Loads a game and converts it to a turn game if it was not."""
        return self.load_game()

    def formal_subproblem_description(self, game_state, history, formulae_to_check: str = None) -> str:
        """Generates a formal description of a subproblem based on the current history of the game."""
        raise Exception("Not implemented!")

    def formal_subproblem_description_game_tree(self, game_tree, history, formulae_to_check: str = None) -> str:
        """Generates a formal description of a subproblem resulting from removing actions not included in the
        game tree."""
        raise Exception("Not implemented!")

    def termination_condition(self, c):
        """Determines when the branching of the game search space will conclude."""
        raise Exception("Not implemented!")

    def get_moves_from_history_str(self, history: str) -> list[str]:
        """Converts a single history string to a list of successive actions."""
        raise Exception("Not implemented!")

    def add_move_to_history_str(self, history: str, move: str) -> str:
        """Adds a move to the history string."""
        if history is None or history == "":
            return move
        else:
            return history + "," + move

    def get_player_id(self, player_name: str):
        return self.players[player_name]

    def get_player_name(self, player_id: int):
        for k, v in self.players.items():
            if v == player_id:
                return k
        raise Exception("Unrecognized player name!")

    @classmethod
    def get_default_formula_and_coalition(cls):
        raise Exception("Not implemented!")



class GameMnk(GameInterface):
    def __init__(self, m, n, k, max_num_actions=10):
        self.m = m
        self.n = n
        self.k = k
        self.max_num_actions = max_num_actions
        GameInterface.__init__(self, players={"cross": 0, "nought": 1})

    def get_name(self):
        return "mnk"

    def load_game(self) -> pyspiel.Game:
        return pyspiel.load_game("mnk", {"m": self.m, "n": self.n, "k": self.k})

    def formal_subproblem_description(self, game_state, history, formulae_to_check: str = None) -> str:
        if formulae_to_check is None:
            formulae_to_check, _ = self.get_default_formula_and_coalition()
        if isinstance(history, list):
            history = ",".join(history)
        return make_whole_board(self.m, self.n, self.k, history, formulae_to_check)

    def formal_subproblem_description_game_tree(self, game_tree, history, formulae_to_check: str = None) -> str:
        """Generates a formal description of a subproblem resulting from removing actions not included in the
        game tree. History is used to generate the initial state."""
        if formulae_to_check is None:
            formulae_to_check, _ = self.get_default_formula_and_coalition()
        if isinstance(history, list):
            history = ",".join(history)
        return make_whole_board_game_tree(self.m, self.n, self.k, game_tree, history, formulae_to_check)

    def termination_condition(self, history: str):
        """Determines when the branching of the game search space will conclude."""
        # Method currently not used, instead state handles termination conditions
        if history.count('x') + history.count('o') >= self.max_num_actions:
            return True
        return False

    def get_moves_from_history_str(self, history: str) -> list[str]:
        """Converts a single history string to a list of successive actions."""
        return re.findall(r'[xo]\(\d+,\d+\)', history)

    @classmethod
    def get_default_formula_and_coalition(cls):
        return "<cross> F (crosswins and ! noughtwins);", {0}



if __name__ == "__main__":
    # --  History: x(2,2),o(1,2),x(1,1),o(0,0),x(3,1)
    # --  Game state:
    # --  .....
    # --  .xo..
    # --  .xo..
    # --  .xo..
    # --  .....
    # print(make_whole_board(5, 5, 4, "x(1,1),o(1,2),x(2,1),o(2,2),x(3,1),o(3,2)"))

    from absl import app
    from absl import flags

    flags.DEFINE_integer("m", None, required=True, help="(Game: mnk) Width of the board (i.e., number of columns).")
    flags.DEFINE_integer("n", None, required=True, help="(Game: mnk) Height of the board (i.e., number of rows).")
    flags.DEFINE_integer("k", None, required=True, help="(Game: mnk) Number of elements forming a line to win.")
    flags.DEFINE_string("initial_moves", "", required=False, help="Initial actions to be specified in the game-specific format.")
    flags.DEFINE_string("output_file", None, required=False, help="Path to the directory in which the results of this run will be stored.")
    FLAGS = flags.FLAGS

    def main(argv):
        formula, _ = GameMnk.get_default_formula_and_coalition()
        text = make_whole_board(FLAGS.m, FLAGS.n, FLAGS.k, FLAGS.initial_moves, formula)
        if FLAGS.output_file is None:
            print(text)
        else:
            with open(FLAGS.output_file, "w") as f:
                f.write(text)

    app.run(main)
