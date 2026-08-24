from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lark import Lark, Transformer


# ============================================================
# AST - Model
# ============================================================

@dataclass
class ISPLModel:
    semantics: Optional[str] = None
    environment: Optional["Environment"] = None
    agents: list["Agent"] = field(default_factory=list)
    evaluation: Optional["Evaluation"] = None
    initial_states: Optional["InitialStates"] = None
    groups: Optional["Groups"] = None
    fairness: Optional["Fairness"] = None
    formulae: Optional["Formulae"] = None


# ============================================================
# AST - Variables
# ============================================================

@dataclass
class VariableDef:
    name: str
    type: str
    lower: Optional[int] = None
    upper: Optional[int] = None
    values: list[str] = field(default_factory=list)


# ============================================================
# AST - Values
# ============================================================

@dataclass
class IntLiteral:
    value: int


@dataclass
class BoolLiteral:
    value: bool


@dataclass
class Name:
    name: str


@dataclass
class Reference:
    owner: str
    name: str


# ============================================================
# AST - Expressions
# ============================================================

@dataclass
class UnaryExpr:
    operator: str
    operand: object


@dataclass
class BinaryExpr:
    operator: str
    left: object
    right: object


# ============================================================
# AST - Boolean conditions
# ============================================================

@dataclass
class BooleanNot:
    operator: str
    operand: object


@dataclass
class BooleanBinary:
    operator: str
    left: object
    right: object


@dataclass
class Comparison:
    operator: str
    left: object
    right: object


@dataclass
class ActionTest:
    scope: str
    agent: Optional[str]
    action: str


# ============================================================
# AST - Evolution
# ============================================================

@dataclass
class EvolutionAssignment:
    target: str
    value: object


@dataclass
class EvolutionAnd:
    left: object
    right: object


@dataclass
class EvolutionRule:
    result: object
    condition: object


# ============================================================
# AST - Protocol
# ============================================================

@dataclass
class ProtocolRule:
    condition: object
    actions: list[str]


@dataclass
class OtherBranch:
    actions: list[str]


@dataclass
class Protocol:
    rules: list[ProtocolRule] = field(default_factory=list)
    other: Optional[OtherBranch] = None


# ============================================================
# AST - Environment
# ============================================================

@dataclass
class Environment:
    observable_vars: list[VariableDef] = field(default_factory=list)
    vars: list[VariableDef] = field(default_factory=list)
    red_states: Optional[object] = None
    actions: list[str] = field(default_factory=list)
    protocol: Optional[Protocol] = None
    evolution: list[EvolutionRule] = field(default_factory=list)


# ============================================================
# AST - Agent
# ============================================================

@dataclass
class Agent:
    name: str
    local_observable_vars: list[str] = field(default_factory=list)
    vars: list[VariableDef] = field(default_factory=list)
    red_states: Optional[object] = None
    actions: list[str] = field(default_factory=list)
    protocol: Optional[Protocol] = None
    evolution: list[EvolutionRule] = field(default_factory=list)


# ============================================================
# AST - Evaluation
# ============================================================

@dataclass
class EvaluationRule:
    name: str
    condition: object


@dataclass
class Evaluation:
    rules: list[EvaluationRule] = field(default_factory=list)


# ============================================================
# AST - Initial states
# ============================================================

@dataclass
class InitialStates:
    condition: object


# ============================================================
# AST - Groups
# ============================================================

@dataclass
class Group:
    name: str
    members: list[str]


@dataclass
class Groups:
    groups: list[Group] = field(default_factory=list)


# ============================================================
# AST - Temporal / epistemic formulae
# ============================================================

@dataclass
class Atom:
    name: str


@dataclass
class UnaryFormula:
    operator: str
    operand: object


@dataclass
class BinaryFormula:
    operator: str
    left: object
    right: object


@dataclass
class TemporalFormula:
    operator: str
    operand: object


@dataclass
class QuantifiedUntilFormula:
    quantifier: str
    left: object
    right: object


@dataclass
class KnowledgeFormula:
    operator: str
    agent: str
    operand: object


@dataclass
class StrategicFormula:
    agent: str
    operator: str
    operand: object


@dataclass
class StateLabel:
    agent: str
    state: str


@dataclass
class EnvironmentStateLabel:
    state: str


@dataclass
class LTLFormula:
    formula: object


@dataclass
class CTLStarFormula:
    formula: object


@dataclass
class Fairness:
    formulas: list[object] = field(default_factory=list)


@dataclass
class Formulae:
    formulas: list[object] = field(default_factory=list)


# ============================================================
# Transformer
# ============================================================

class ISPLTransformer(Transformer):

    # --------------------------------------------------------
    # Primitive tokens
    # --------------------------------------------------------

    def ID(self, token):
        return str(token)

    def NUMBER(self, token):
        return int(token)

    # --------------------------------------------------------
    # Semantics
    # --------------------------------------------------------

    def single_assignment(self, _):
        return "SingleAssignment"

    def multi_assignment(self, _):
        return "MultiAssignment"

    def semantics(self, items):
        return items[0]

    # --------------------------------------------------------
    # Integer
    # --------------------------------------------------------

    def positive_integer(self, items):
        return IntLiteral(items[0])

    def negative_integer(self, items):
        return IntLiteral(-items[0])

    def integer_value(self, items):
        return items[0]

    # --------------------------------------------------------
    # Boolean values
    # --------------------------------------------------------

    def true_value(self, _):
        return BoolLiteral(True)

    def false_value(self, _):
        return BoolLiteral(False)

    # --------------------------------------------------------
    # Lists
    # --------------------------------------------------------

    def enum_list(self, items):
        return [str(item) for item in items]

    def action_list(self, items):
        return [str(item) for item in items]

    def action_list_nonempty(self, items):
        return [str(item) for item in items]

    # --------------------------------------------------------
    # Variable definitions
    # --------------------------------------------------------

    def boolean_var(self, items):
        return VariableDef(
            name=str(items[0]),
            type="boolean",
        )

    def integer_var(self, items):
        return VariableDef(
            name=str(items[0]),
            type="integer",
            lower=items[1].value,
            upper=items[2].value,
        )

    def enum_var(self, items):
        return VariableDef(
            name=str(items[0]),
            type="enum",
            values=items[1],
        )

    def observable_vars(self, items):
        return list(items)

    def env_vars(self, items):
        return list(items)

    def agent_vars(self, items):
        return list(items)

    # --------------------------------------------------------
    # Red states
    # --------------------------------------------------------

    def red_state_condition(self, items):
        return items[0]

    def red_states(self, items):
        return items[0] if items else None

    # --------------------------------------------------------
    # Actions
    # --------------------------------------------------------

    def env_actions(self, items):
        return items[0] if items else []

    def agent_actions(self, items):
        return items[0]

    def local_obsvars(self, items):
        return items[0] if items else []

    # --------------------------------------------------------
    # Protocol
    # --------------------------------------------------------

    def env_protocol_line(self, items):
        return ProtocolRule(
            condition=items[0],
            actions=items[1],
        )

    def agent_protocol_line(self, items):
        return ProtocolRule(
            condition=items[0],
            actions=items[1],
        )

    def otherbranch(self, items):
        return OtherBranch(
            actions=items[0],
        )

    def env_protocol(self, items):
        rules = []
        other = None

        for item in items:
            if isinstance(item, ProtocolRule):
                rules.append(item)

            elif isinstance(item, OtherBranch):
                other = item

        return Protocol(
            rules=rules,
            other=other,
        )

    def agent_protocol(self, items):
        rules = []
        other = None

        for item in items:
            if isinstance(item, ProtocolRule):
                rules.append(item)

            elif isinstance(item, OtherBranch):
                other = item

        return Protocol(
            rules=rules,
            other=other,
        )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    def environment(self, items):
        observable_vars = []
        vars_ = []
        red_states = None
        actions = []
        protocol = None
        evolution = []

        for item in items:
            if isinstance(item, list):
                if not item:
                    continue

                if all(isinstance(x, VariableDef) for x in item):
                    if not observable_vars:
                        observable_vars = item
                    else:
                        vars_ = item

                elif all(isinstance(x, str) for x in item):
                    actions = item

                elif all(isinstance(x, EvolutionRule) for x in item):
                    evolution = item

            elif isinstance(item, Protocol):
                protocol = item

            elif isinstance(
                item,
                (
                    BooleanNot,
                    BooleanBinary,
                    Comparison,
                    ActionTest,
                ),
            ):
                red_states = item

        return Environment(
            observable_vars=observable_vars,
            vars=vars_,
            red_states=red_states,
            actions=actions,
            protocol=protocol,
            evolution=evolution,
        )

    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    def agent(self, items):
        name = str(items[0])

        local_observable_vars = []
        vars_ = []
        red_states = None
        actions = []
        protocol = None
        evolution = []

        for item in items[1:]:
            if isinstance(item, Protocol):
                protocol = item

            elif isinstance(item, list):
                if not item:
                    continue

                if all(isinstance(x, VariableDef) for x in item):
                    vars_ = item

                elif all(isinstance(x, str) for x in item):
                    if not local_observable_vars:
                        local_observable_vars = item
                    else:
                        actions = item

                elif all(isinstance(x, EvolutionRule) for x in item):
                    evolution = item

            elif isinstance(
                item,
                (
                    BooleanNot,
                    BooleanBinary,
                    Comparison,
                    ActionTest,
                ),
            ):
                red_states = item

        return Agent(
            name=name,
            local_observable_vars=local_observable_vars,
            vars=vars_,
            red_states=red_states,
            actions=actions,
            protocol=protocol,
            evolution=evolution,
        )

    # --------------------------------------------------------
    # Evolution
    # --------------------------------------------------------

    def result_assignment(self, items):
        return EvolutionAssignment(
            target=str(items[0]),
            value=items[1],
        )

    def result_and(self, items):
        return EvolutionAnd(
            left=items[0],
            right=items[1],
        )

    def result_paren(self, items):
        return items[0]

    def evolution_line(self, items):
        return EvolutionRule(
            result=items[0],
            condition=items[1],
        )

    def env_evolution(self, items):
        return list(items)

    def agent_evolution(self, items):
        return list(items)

    # --------------------------------------------------------
    # References / names
    # --------------------------------------------------------

    def name_value(self, items):
        return Name(str(items[0]))

    def environment_reference(self, items):
        return Reference(
            owner="Environment",
            name=str(items[0]),
        )

    def agent_reference(self, items):
        return Reference(
            owner=str(items[0]),
            name=str(items[1]),
        )

    # --------------------------------------------------------
    # Expressions
    # --------------------------------------------------------

    def expression_paren(self, items):
        return items[0]

    def binary_add(self, items):
        return BinaryExpr(
            operator="+",
            left=items[0],
            right=items[2],
        )

    def binary_sub(self, items):
        return BinaryExpr(
            operator="-",
            left=items[0],
            right=items[2],
        )

    def binary_mul(self, items):
        return BinaryExpr(
            operator="*",
            left=items[0],
            right=items[2],
        )

    def binary_div(self, items):
        return BinaryExpr(
            operator="/",
            left=items[0],
            right=items[2],
        )

    def binary_bit_and(self, items):
        return BinaryExpr(
            operator="&",
            left=items[0],
            right=items[2],
        )

    def binary_bit_xor(self, items):
        return BinaryExpr(
            operator="^",
            left=items[0],
            right=items[2],
        )

    def binary_bit_or(self, items):
        return BinaryExpr(
            operator="|",
            left=items[0],
            right=items[2],
        )

    def unary_bit_not(self, items):
        return UnaryExpr(
            operator="~",
            operand=items[1],
        )

    # --------------------------------------------------------
    # Logic operators
    # --------------------------------------------------------

    def op_lt(self, _):
        return "<"

    def op_le(self, _):
        return "<="

    def op_gt(self, _):
        return ">"

    def op_ge(self, _):
        return ">="

    def op_eq(self, _):
        return "="

    def op_ne(self, _):
        return "!="

    # --------------------------------------------------------
    # Boolean conditions
    # --------------------------------------------------------

    def bool_paren(self, items):
        return items[0]

    def bool_and(self, items):
        return BooleanBinary(
            operator="and",
            left=items[0],
            right=items[2],
        )

    def bool_or(self, items):
        return BooleanBinary(
            operator="or",
            left=items[0],
            right=items[2],
        )

    def bool_not(self, items):
        return BooleanNot(
            operator="not",
            operand=items[1],
        )

    def bool_comparison(self, items):
        return Comparison(
            operator=items[1],
            left=items[0],
            right=items[2],
        )

    # --------------------------------------------------------
    # Action tests
    # --------------------------------------------------------

    def local_action_test(self, items):
        return ActionTest(
            scope="local",
            agent=None,
            action=str(items[0]),
        )

    def agent_action_test(self, items):
        return ActionTest(
            scope="agent",
            agent=str(items[0]),
            action=str(items[1]),
        )

    def environment_action_test(self, items):
        return ActionTest(
            scope="environment",
            agent=None,
            action=str(items[0]),
        )

    def bool_action_test(self, items):
        return items[0]

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    def evaluation_rule(self, items):
        return EvaluationRule(
            name=str(items[0]),
            condition=items[1],
        )

    def evaluation(self, items):
        return Evaluation(
            rules=list(items),
        )

    # --------------------------------------------------------
    # Initial states
    # --------------------------------------------------------

    def init_states(self, items):
        return InitialStates(
            condition=items[0],
        )

    # --------------------------------------------------------
    # Groups
    # --------------------------------------------------------

    def environment_name(self, _):
        return "Environment"

    def named_agent(self, items):
        return str(items[0])

    def group_members(self, items):
        return list(items)

    def group_declaration(self, items):
        return Group(
            name=str(items[0]),
            members=items[1],
        )

    def group_line(self, items):
        return list(items)

    def groups(self, items):
        declarations = []

        for item in items:
            if isinstance(item, list):
                declarations.extend(item)
            elif isinstance(item, Group):
                declarations.append(item)

        return Groups(groups=declarations)

    # --------------------------------------------------------
    # Formula helpers
    # --------------------------------------------------------

    def formula_paren(self, items):
        return items[0]

    def formula_atom(self, items):
        return Atom(str(items[0]))

    def formula_and(self, items):
        return BinaryFormula(
            operator="and",
            left=items[0],
            right=items[1],
        )

    def formula_or(self, items):
        return BinaryFormula(
            operator="or",
            left=items[0],
            right=items[1],
        )

    def formula_not(self, items):
        return UnaryFormula(
            operator="!",
            operand=items[0],
        )

    def formula_implies(self, items):
        return BinaryFormula(
            operator="->",
            left=items[0],
            right=items[1],
        )

    # --------------------------------------------------------
    # CTL temporal operators
    # --------------------------------------------------------

    def formula_ag(self, items):
        return TemporalFormula("AG", items[0])

    def formula_eg(self, items):
        return TemporalFormula("EG", items[0])

    def formula_ax(self, items):
        return TemporalFormula("AX", items[0])

    def formula_ex(self, items):
        return TemporalFormula("EX", items[0])

    def formula_af(self, items):
        return TemporalFormula("AF", items[0])

    def formula_ef(self, items):
        return TemporalFormula("EF", items[0])

    # --------------------------------------------------------
    # CTL Until
    # --------------------------------------------------------

    def formula_au(self, items):
        return QuantifiedUntilFormula(
            quantifier="A",
            left=items[0],
            right=items[1],
        )

    def formula_eu(self, items):
        return QuantifiedUntilFormula(
            quantifier="E",
            left=items[0],
            right=items[1],
        )

    # --------------------------------------------------------
    # Knowledge
    # --------------------------------------------------------

    def formula_k(self, items):
        return KnowledgeFormula(
            operator="K",
            agent=str(items[0]),
            operand=items[1],
        )

    def formula_k_environment(self, items):
        return KnowledgeFormula(
            operator="K",
            agent="Environment",
            operand=items[0],
        )

    def formula_gk(self, items):
        return KnowledgeFormula(
            operator="GK",
            agent=str(items[0]),
            operand=items[1],
        )

    def formula_gck(self, items):
        return KnowledgeFormula(
            operator="GCK",
            agent=str(items[0]),
            operand=items[1],
        )

    def formula_o(self, items):
        return KnowledgeFormula(
            operator="O",
            agent=str(items[0]),
            operand=items[1],
        )

    def formula_o_environment(self, items):
        return KnowledgeFormula(
            operator="O",
            agent="Environment",
            operand=items[0],
        )

    def formula_dk(self, items):
        return KnowledgeFormula(
            operator="DK",
            agent=str(items[0]),
            operand=items[1],
        )

    # --------------------------------------------------------
    # Strategic operators
    # --------------------------------------------------------

    def formula_strategic_x(self, items):
        return StrategicFormula(
            agent=str(items[0]),
            operator="X",
            operand=items[1],
        )

    def formula_strategic_f(self, items):
        return StrategicFormula(
            agent=str(items[0]),
            operator="F",
            operand=items[1],
        )

    def formula_strategic_g(self, items):
        return StrategicFormula(
            agent=str(items[0]),
            operator="G",
            operand=items[1],
        )

    def formula_strategic_until(self, items):
        return StrategicFormula(
            agent=str(items[0]),
            operator="U",
            operand=BinaryFormula(
                operator="U",
                left=items[1],
                right=items[2],
            ),
        )

    # --------------------------------------------------------
    # Green/Red states
    # --------------------------------------------------------

    def formula_green_agent(self, items):
        return StateLabel(
            agent=str(items[0]),
            state="GreenStates",
        )

    def formula_red_agent(self, items):
        return StateLabel(
            agent=str(items[0]),
            state="RedStates",
        )

    def formula_green_environment(self, _):
        return EnvironmentStateLabel(
            state="GreenStates",
        )

    def formula_red_environment(self, _):
        return EnvironmentStateLabel(
            state="RedStates",
        )

    # --------------------------------------------------------
    # LTL wrapper
    # --------------------------------------------------------

    def formula_ltl(self, items):
        return LTLFormula(
            formula=items[0],
        )

    # --------------------------------------------------------
    # CTL* wrapper
    # --------------------------------------------------------

    def formula_ctlstar(self, items):
        return CTLStarFormula(
            formula=items[0],
        )

    # --------------------------------------------------------
    # LTL
    # --------------------------------------------------------

    def ltl_paren(self, items):
        return items[0]

    def ltl_atom(self, items):
        return Atom(str(items[0]))

    def ltl_and(self, items):
        return BinaryFormula("and", items[0], items[1])

    def ltl_or(self, items):
        return BinaryFormula("or", items[0], items[1])

    def ltl_not(self, items):
        return UnaryFormula("!", items[0])

    def ltl_implies(self, items):
        return BinaryFormula("->", items[0], items[1])

    def ltl_g(self, items):
        return TemporalFormula("G", items[0])

    def ltl_f(self, items):
        return TemporalFormula("F", items[0])

    def ltl_x(self, items):
        return TemporalFormula("X", items[0])

    def ltl_until(self, items):
        return BinaryFormula("U", items[0], items[1])

    def ltl_k(self, items):
        return KnowledgeFormula(
            operator="K",
            agent=str(items[0]),
            operand=items[1],
        )

    def ltl_gk(self, items):
        return KnowledgeFormula(
            operator="GK",
            agent=str(items[0]),
            operand=items[1],
        )

    def ltl_gck(self, items):
        return KnowledgeFormula(
            operator="GCK",
            agent=str(items[0]),
            operand=items[1],
        )

    def ltl_dk(self, items):
        return KnowledgeFormula(
            operator="DK",
            agent=str(items[0]),
            operand=items[1],
        )

    # --------------------------------------------------------
    # CTL*
    # --------------------------------------------------------

    def ctl_paren(self, items):
        return items[0]

    def ctl_atom(self, items):
        return Atom(str(items[0]))

    def ctl_and(self, items):
        return BinaryFormula("and", items[0], items[1])

    def ctl_or(self, items):
        return BinaryFormula("or", items[0], items[1])

    def ctl_not(self, items):
        return UnaryFormula("!", items[0])

    def ctl_implies(self, items):
        return BinaryFormula("->", items[0], items[1])

    def ctl_a(self, items):
        return TemporalFormula("A", items[0])

    def ctl_e(self, items):
        return TemporalFormula("E", items[0])

    def ctl_k(self, items):
        return KnowledgeFormula(
            operator="K",
            agent=str(items[0]),
            operand=items[1],
        )

    def ctl_gk(self, items):
        return KnowledgeFormula(
            operator="GK",
            agent=str(items[0]),
            operand=items[1],
        )

    def ctl_gck(self, items):
        return KnowledgeFormula(
            operator="GCK",
            agent=str(items[0]),
            operand=items[1],
        )

    def ctl_dk(self, items):
        return KnowledgeFormula(
            operator="DK",
            agent=str(items[0]),
            operand=items[1],
        )

    def ctl_path_state(self, items):
        return items[0]

    def ctl_path_g(self, items):
        return TemporalFormula("G", items[0])

    def ctl_path_f(self, items):
        return TemporalFormula("F", items[0])

    def ctl_path_x(self, items):
        return TemporalFormula("X", items[0])

    def ctl_path_until(self, items):
        return BinaryFormula("U", items[0], items[1])

    # --------------------------------------------------------
    # Fairness
    # --------------------------------------------------------

    def fairness(self, items):
        return Fairness(
            formulas=list(items),
        )

    # --------------------------------------------------------
    # Formulae
    # --------------------------------------------------------

    def formula_line(self, items):
        return items[0]

    def formulae(self, items):
        return Formulae(
            formulas=list(items),
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    def interpreted_system(self, items):
        model = ISPLModel()

        if isinstance(items[0], str):
            model.semantics = items[0]
        else:
            model.semantics = "MultiAssignment"  # The default value

        for item in items:
            if isinstance(item, Environment):
                model.environment = item

            elif isinstance(item, Agent):
                model.agents.append(item)

            elif isinstance(item, Evaluation):
                model.evaluation = item

            elif isinstance(item, InitialStates):
                model.initial_states = item

            elif isinstance(item, Groups):
                model.groups = item

            elif isinstance(item, Fairness):
                model.fairness = item

            elif isinstance(item, Formulae):
                model.formulae = item

        return model


# ============================================================
# Parser wrapper
# ============================================================

class ISPLParser:
    """
    High-level ISPL parser.

    Example:

        parser = ISPLParser()

        model = parser.parse(text)

        model = parser.parse_file("model.ispl")
    """

    def __init__(
        self,
        grammar_path: str | Path | None = None,
    ):
        if grammar_path is None:
            grammar_path = (
                Path(__file__).resolve().with_name("ispl.lark")
            )

        grammar_path = Path(grammar_path)

        self._parser = Lark.open(
            grammar_path,
            parser="earley",
            lexer="dynamic",
            ambiguity="resolve",
            start="start",
        )

    def parse(self, text: str) -> ISPLModel:
        tree = self._parser.parse(text)
        return ISPLTransformer().transform(tree)

    def parse_file(
        self,
        path: str | Path,
        encoding: str = "utf-8",
    ) -> ISPLModel:
        path = Path(path)

        with path.open(
            "r",
            encoding=encoding,
        ) as file:
            return self.parse(file.read())


# ============================================================
# Convenience API
# ============================================================

_default_parser: ISPLParser | None = None


def get_parser() -> ISPLParser:
    global _default_parser

    if _default_parser is None:
        _default_parser = ISPLParser()

    return _default_parser


def parse(text: str) -> ISPLModel:
    return get_parser().parse(text)


def parse_file(
    path: str | Path,
    encoding: str = "utf-8",
) -> ISPLModel:
    return get_parser().parse_file(
        path,
        encoding=encoding,
    )


# ============================================================
# Optional CLI
# ============================================================

def main() -> None:
    import argparse
    from pprint import pprint

    argument_parser = argparse.ArgumentParser(
        description="Parse an ISPL file into an AST."
    )

    argument_parser.add_argument(
        "file",
        help="Path to an ISPL file.",
    )

    args = argument_parser.parse_args()

    model = parse_file(args.file)

    pprint(model)


if __name__ == "__main__":
    main()
