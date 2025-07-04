import sys
from pathlib import Path
from lark import Lark, Transformer, v_args
from lark.visitors import merge_transformers
from .stv_specification import StvSpecification, AgentLocalModelSpec, Transition, State

_stv2_parser_lark_spec = (Path(__file__).parent / "grammar_stv_v2.lark").open().read()
_formula_parser_lark_spec = (Path(__file__).parent / "formula.lark").open().read()


#Some links:
# - Lark tutorial for JSON: https://lark-parser.readthedocs.io/en/latest/json_tutorial.html
# - Lark documentation: https://lark-parser.readthedocs.io/en/latest/grammar.html

class ExprNode:
    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.is_constant = False
        self.is_variable = False

    def __str__(self):
        if len(self.args) == 1:
            return f"({self.name} {self.args[0]})"
        elif len(self.args) == 2:
            return f"({self.args[0]} {self.name} {self.args[1]})"
        else:
            return f"{self.name}({', '.join([str(a) for a in self.args])})"

    def __repr__(self):
        return str(self)


class ConstantNode(ExprNode):
    def __init__(self, value):
        super().__init__(value, [])
        self.is_constant = True

    def __str__(self):
        return str(self.name)


class VariableNode(ExprNode):
    def __init__(self, value):
        super().__init__(value, [])
        self.is_variable = True

    def __str__(self):
        return self.name


class ModalExprNode:
    def __init__(self, modal_op, coalition, formula: ExprNode):
        self.modal_op = modal_op
        self.coalition = coalition
        self.formula = formula

    def __str__(self):
        return f"<<{', '.join(self.coalition)}>>{self.modal_op}({self.formula})"



class FormulaTransformer(Transformer):
    def __init__(self, silent=True):
        super().__init__()
        self.silent = silent

    def start(self, args):
        return args[0]

    def formula_expr(self, args):
        if len(args) == 2:
            return ExprNode("!", [args[1]])
        elif len(args) == 3:
            if str(args[0]) == "(":
                return args[1]
            op = str(args[1])
            if op in {"&&", "||"}:
                return ExprNode(op, [args[0], args[2]])
            else:
                return ExprNode(op, [args[0], args[2]])
        else:
            # "<<" formula_agent_list ">>" "[]" "(" formula_expr ")"
            coalition = args[1]
            op = str(args[3])
            formula = args[5]
            return ModalExprNode(op, coalition, formula)

    def constant(self, args):
        return ConstantNode(int(args[0]))

    def variable(self, args):
        return VariableNode(str(args[0]))

    def formula_agent_list(self, args):
        return [str(a) for a in args]



class Stv2Transformer(Transformer):
    def __init__(self, silent):
        super().__init__()
        self.silent = silent

    def get_state_param_value(self, param_string):
        if str.isnumeric(param_string):
            return int(param_string)
        else:
            return param_string

    def start(self, args):
        agent_models = []
        formula = None
        for a in args:
            if isinstance(a, AgentLocalModelSpec):
                agent_models.append(a)
            elif isinstance(a, tuple) and a[0] == "FORMULA:":
                formula = a[1]
        model = StvSpecification(agents=agent_models)
        return model, formula

    def decl_formula(self, args):
        return "FORMULA:", args[0]

    def agent(self, args):
        name, num = args[0]
        metadata = args[1]
        if "init" not in metadata:
            raise Exception(f"Initial state is not defined for agent {name}")
        init_state = metadata["init"]
        local_variables = metadata.get("LOCAL:", [])
        local_variables_init_values = metadata.get("INITIAL:", {})  #TODO: should we set all uninitialized variables to 0 here?
        persistent_variables = metadata.get("PERSISTENT:", [])
        transitions = args[2]
        return AgentLocalModelSpec(name, num, init_state, local_variables=local_variables, local_variables_init_values=local_variables_init_values, persistent_variables=persistent_variables, transitions=transitions)

    def agent_header(self, args):
        name = str(args[0])
        if len(args) == 1:
            return name, None
        else:
            return name, int(args[1])

    def agent_meta(self, args):
        d = {str(label): value for label, value in args}
        return d

    def agent_meta_el(self, args):
        name = str(args[0])
        if name == "init":
            return name, str(args[1])
        elif len(args) >= 4:
            return name, args[2]
        elif name == "INITIAL:":  # case of empty initial
            return name, {}
        else:  # case of []
            return name, []

    def local_vars_list(self, args):
        return [str(a) for a in args]

    def local_vars_list_initial(self, args):
        return {k: v for k, v in args}

    def local_vars_list_initial_el(self, args):
        return str(args[0]), self.get_state_param_value(str(args[1]))

    def agent_body(self, args):
        return args

    def init_state(self, args):
        return str(args[0])

    @v_args(inline=True)  # inline is roughly equivalent to *args
    def transition(self, shared, global_name, local_name, state0_name, state0_vars, state1_name, state1_vars):
        is_shared, shared_num = (False, None) if shared is None else (True, int(shared[1]))
        state0 = State(str(state0_name), state0_vars)
        state1 = State(str(state1_name), state1_vars)
        if local_name is not None:
            local_name = str(local_name)
        # print("Created state: ", str(state0))
        # print("Created state: ", str(state1))
        t = Transition(str(global_name), state0, state1, is_shared=is_shared, shared_num=shared_num, local_name=local_name)
        if not self.silent:
            print("Created transition: ", t)
        return t

    def shared(self, args):
        if len(args) == 0:
            return True, None
        else:
            return True, args[0]

    def var_equality(self, args):
        return {k: v for k, v in args}

    def var_assign(self, args):
        return {k: v for k, v in args}

    def var_equality_in(self, args):
        return str(args[0]), self.get_state_param_value(str(args[1]))

    def var_assign_in(self, args):
        return str(args[0]), self.get_state_param_value(str(args[1]))

    def list(self, items):
        return list(items)
    def pair(self, key_value):
        k, v = key_value
        return k, v
    def dict(self, items):
        return dict(items)


def getParser():
    return Lark(_stv2_parser_lark_spec, start='start')  #, parser='lalr'

def getFormulaParser():
    return Lark(_formula_parser_lark_spec, start='start')

def parseAndTransformFormula(formula: str):
    parser = getFormulaParser()
    transformer = FormulaTransformer(silent=True)
    return transformer.transform(parser.parse(formula))

def do_test_parser():
    parser = getParser()
    with (Path(__file__).parent / "benchmarks" / "rps.stv").open() as f:
        text = f.read()
    tree = parser.parse(text)
    print("TEST2:\n", tree.pretty())


class Stv2Parser:
    """PDDL domain parser class."""

    def __init__(self, silent=True):
        """Initialize."""
        # About merging transformers:
        # - https://github.com/lark-parser/lark/blob/master/examples/composition/storage.lark
        # - https://lark-parser.readthedocs.io/en/latest/examples/composition/main.html#sphx-glr-examples-composition-main-py
        self._transformer = merge_transformers(Stv2Transformer(silent=silent), formula=FormulaTransformer(silent=silent))
        self._parser = Lark(
            _stv2_parser_lark_spec, parser="earley", import_paths=[str(Path(__file__).parent)]
        )  # earley supports rule priority, lalr supports terminal priority

    def __call__(self, text):
        """Call."""
        sys.tracebacklimit = 0  # noqa
        tree = self._parser.parse(text)
        sys.tracebacklimit = None  # noqa
        stv_spec, formula = self._transformer.transform(tree)
        return stv_spec, formula


def do_test_STV_parser():
    parser = Stv2Parser()
    # with (Path(__file__).parent / "benchmarks" / "rps.stv").open() as f:
    with (Path(__file__).parent.parent.parent / "example_specifications" / "complex" / "Voters5Candidates3.txt").open() as f:
        text = f.read()
    model, formula = parser(text)
    print("TEST2\nagent0: ", model.agents[0])
    print("formula: ", formula)


def get_coalition_from_formula(formula):
    # Example formula: <<Train1>>[](Train1_pos==1 || Train1_pos==2 || Train1_pos==3)
    a = formula.find("<<")
    b = formula.find(">>", a + 2)
    return formula[a + 2:b].split(",")


if __name__ == "__main__":
    # do_test_parser()
    do_test_STV_parser()