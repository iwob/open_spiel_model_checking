import re
def generate_table(m, n):
    table = []
    for i in range(1, n+1):
        row = []
        for j in range(1, m+1):
            row.append(f"b{i}{j} : {{x, o, b}}")
        table.append(row)
    return table

def generate_conditions(m, n):
    conditions = []
    for i in range(1, n+1):
        for j in range(1, m+1):
            condition_o = f"b{i}{j} = o if turn = nought and Nought.Action = a{i}{j};"
            condition_x = f"b{i}{j} = x if turn = cross  and Cross.Action  = a{i}{j};"
            conditions.append(condition_o)
            conditions.append(condition_x)
    return conditions

def generate_actions(m, n):
    actions = []
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            actions.append(f"a{i}{j}")
    
    actions.append("none")
    
    return actions
def generate_environment_conditions(m, n):
    conditions = []
    
    for i in range(1, n+1):
        for j in range(1, m+1):
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


def generate_board_condition(m, n, value, moves):
    conditions = []
    pattern = r'[xo]\(\d+,\d+\)'
    if moves:
      moves_list =  re.findall(pattern, moves)
    board = [[value for _ in range(m)] for _ in range(n)]
    
    if moves:
        for move in moves_list:
            symbol = move[0]
            coords = move[2:-1].split(',')
            
            row, col = int(coords[0]), int(coords[1])
            board[row ][col ] = symbol
    
    for row in range(1, n + 1):
        for col in range(1, m + 1):
            conditions.append(f"Environment.b{row}{col} = {board[row - 1][col - 1]}")
        conditions[-1] += "\n"
    
    return " and ".join(conditions)

def make_whole_board(m,n,k, im):
    move = im.count('o') + im.count('x')
    move = move %2
    table = generate_table(m, n)
    conditions = generate_conditions(m, n)
    actions = generate_actions(m, n)
    actions_str = ", ".join(actions)
    environment_conditions = generate_environment_conditions(m, n)
    evaluation_conditions_o = generate_evaluation_conditions_win(m, n, k, "o")
    evaluation_conditions_x = generate_evaluation_conditions_win(m, n, k, "x")
    print("""Semantics=SingleAssignment;
    Agent Environment
      Obsvars:
      turn : {nought, cross};""")
    for row in table:
        print("; ".join(row) + ";")
    print("""  end Obsvars
      Actions = { }; 
      Protocol: end Protocol
      Evolution:
        turn=nought if turn=cross; turn=cross if turn=nought;""")
    for condition in conditions:
        print(condition)
    print("""  end Evolution
    end Agent

    Agent Nought
      Vars:
        null : boolean; -- for syntax reasons only
      end Vars""")
    print(f"Actions = {{{actions_str}}};")
    print("Protocol:\n")
    for condition in environment_conditions:
        print(condition)

    print("""Other : { none }; -- technicality
      end Protocol
      Evolution:
        null=true if null=true;
      end Evolution
    end Agent\n""")

    print("""  
    Agent Cross
      Vars:
        null : boolean; -- for syntax reasons only
      end Vars""")
    print(f"Actions = {{{actions_str}}};")
    print("Protocol:\n")
    for condition in environment_conditions:
        print(condition)
    print("""Other : { none }; -- technicality
      end Protocol
      Evolution:
        null=true if null=true;
      end Evolution
    end Agent\n""")
    print("""Evaluation
      noughtwins if""")
    print(evaluation_conditions_o + ";")
    print("  crosswins if ")
    print(evaluation_conditions_x + ";")
    print("end Evaluation")
    print("InitStates")
    print(generate_board_condition(m, n, "b", im))
    if move ==0:
      print("""  and Environment.turn = cross
        and Nought.null = true and Cross.null = true;
      end InitStates

      Groups
        nought = {Nought}; cross = {Cross};
      end Groups

      Formulae
        <cross> F (crosswins and ! noughtwins); -- TRUE
        <nought> F (noughtwins and ! crosswins); -- FALSE
      end Formulae""")
    else:
        print("""  and Environment.turn = nought
        and Nought.null = true and Cross.null = true;
      end InitStates

      Groups
        nought = {Nought}; cross = {Cross};
      end Groups

      Formulae
        <cross> F (crosswins and ! noughtwins); -- TRUE
        <nought> F (noughtwins and ! crosswins); -- FALSE
      end Formulae""")
        
make_whole_board(5,5,5,"x(0,1),o(3,1),x(0,3),o(1,1),x(0,4),o(1,0),x(0,2)")