# from lark import Lark
#
# with open("grammar_ispl_v2.lark", encoding="utf-8") as f:
#     grammar = f.read()
#
# parser = Lark(
#     grammar,
#     parser="earley",
#     lexer="dynamic",
#     start="start",
# )
#
# with open("mnk_3,3,3_initial_U.ispl", encoding="utf-8") as f:
#     tree = parser.parse(f.read())
#
# print(tree.pretty())
from pathlib import Path

from typing import Optional

from ispl_parser import parse_file, ISPLModel, BooleanBinary, Comparison, Reference, IntLiteral, BoolLiteral, Name
from model_checking.stv.parsers.parser_stv_v2 import ModalExprNode
from model_checking.stv.parsers.stv_specification import StvSpecification, AgentLocalModelSpec

# model = parse_file(Path(__file__).parent / "benchmarks" / "mnk_3,3,3_initial_U.ispl")
model = parse_file(Path(__file__).parent / "benchmarks" / "nim_2;3;4.ispl")

# print(model)

print(model.semantics)

for agent in model.agents:
    print(agent.name)

print(model.environment.observable_vars)
print(model.formulae.formulas)


def get_init_value_assignment(ref, value, agent_name) -> Optional[tuple[str, object]]:
    if ref.owner == agent_name:
        if isinstance(value, IntLiteral) or isinstance(value, BoolLiteral):
            return ref.name, value.value
        elif isinstance(value, Name):
            return ref.name, value.name
        else:
            raise Exception(f"Unknown initial value type for {ref}: {value}")
    else:
        return None


def get_agent_initial_values(condition, agent_name):
    if isinstance(condition, BooleanBinary):
        if condition.operator != "and":
            raise Exception("Nondeterministic initial states are not currently supported")
        L = get_agent_initial_values(condition.left, agent_name)
        R = get_agent_initial_values(condition.right, agent_name)
        return L | R
    elif isinstance(condition, Comparison):
        if isinstance(condition.left, Reference) and not isinstance(condition.right, Reference):
            pair = get_init_value_assignment(condition.left, condition.right, agent_name)
        elif isinstance(condition.right, Reference) and not isinstance(condition.left, Reference):
            pair = get_init_value_assignment(condition.right, condition.left, agent_name)
        else:
            raise Exception("Only assignment of constants to variables are currently supported")

        if pair is None:
            return {}
        else:
            return {pair[0]: pair[1]}
    else:
        raise Exception("Initial state must be represented as assignments (=) to variables")



def convert_to_stv_spec(model: ISPLModel) -> (StvSpecification, ModalExprNode):
    agents = []
    # Environment
    local_variables = [vd.name for vd in model.environment.observable_vars]
    persistent_variables = local_variables
    transitions = []

    initial_vals = get_agent_initial_values(model.initial_states.condition, "Environment")
    # local_variables_init_values =
    env = AgentLocalModelSpec("Environment",
                              num_instances=1,
                              init_state="Q",
                              local_variables=local_variables,
                              persistent_variables=persistent_variables,
                              local_variables_init_values=initial_vals,
                              transitions=[])
    agents.append(env)


    stv_spec = StvSpecification(agents)
    return stv_spec, None
    pass


convert_to_stv_spec(model)