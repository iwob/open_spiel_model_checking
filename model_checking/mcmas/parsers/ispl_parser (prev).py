from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from lark import Lark, Transformer
from lark.exceptions import UnexpectedInput


# ============================================================
# AST - podstawowe typy
# ============================================================

@dataclass
class Model:
    semantics: Optional[str]
    environment: Optional["Environment"]
    agents: list["Agent"]
    evaluation: Optional["Evaluation"]
    initial_states: Optional["InitialStates"]
    groups: Optional["Groups"]
    fairness: Optional["Fairness"]
    formulae: Optional["Formulae"]


# ============================================================
# Variables
# ============================================================

@dataclass
class VariableDef:
    name: str
    type: str
    lower: Optional[int] = None
    upper: Optional[int] = None
    values: list[str] = field(default_factory=list)


# ============================================================
# Expressions
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
class Ref:
    owner: str
    name: str


@dataclass
class Unary:
    op: str
    operand: object


@dataclass
class Binary:
    op: str
    left: object
    right: object


@dataclass
class Assignment:
    target: object
    value: object


# ============================================================
# Protocol
# ============================================================

@dataclass
class ProtocolRule:
    condition: object
    actions: list[str]


@dataclass
class OtherRule:
    actions: list[str]


@dataclass
class Protocol:
    rules: list[ProtocolRule] = field(default_factory=list)
    other: Optional[OtherRule] = None


# ============================================================
# Evolution
# ============================================================

@dataclass
class EvolutionRule:
    result: object
    condition: object


# ============================================================
# Environment
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
# Agents
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
# Evaluation
# ============================================================

@dataclass
class EvaluationRule:
    name: str
    condition: object


@dataclass
class Evaluation:
    rules: list[EvaluationRule] = field(default_factory=list)


# ============================================================
# Initial states
# ============================================================

@dataclass
class InitialStates:
    condition: object


# ============================================================
# Groups
# ============================================================

@dataclass
class Group:
    name: str
    members: list[str]


@dataclass
class Groups:
    groups: list[Group] = field(default_factory=list)


# ============================================================
# Temporal / epistemic formulae
# ============================================================

@dataclass
class Atom:
    name: str


@dataclass
class UnaryFormula:
    op: str
    operand: object


@dataclass
class BinaryFormula:
    op: str
    left: object
    right: object


@dataclass
class TemporalFormula:
    quantifier: str
    operator: str
    operand: object


@dataclass
class UntilFormula:
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
    """
    Converts the Lark parse tree into the dataclass-based AST.
    """

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

    def semantics_type(self, items):
        return str(items[0])

    def semantics(self, items):
        return str(items[0])

    # --------------------------------------------------------
    # Integer / bool
    # --------------------------------------------------------

    def integer(self, items):
        if len(items) == 1:
            return IntLiteral(items[0])

        return IntLiteral(-items[0])

    def boolvalue(self, items):
        return BoolLiteral(str(items[0]) == "true")

    # --------------------------------------------------------
    # Lists
    # --------------------------------------------------------

    def id_list(self, items):
        return [str(item) for item in items]

    def id_list_nonempty(self, items):
        return [str(item) for item in items]

    def enumlist(self, items):
        return [str(item) for item in items]

    def agentname_list(self, items):
        return [str(item) for item in items]

    # --------------------------------------------------------
    # Variable definitions
    # --------------------------------------------------------

    def onevardef(self, items):
        name = str(items[0])

        if len(items) == 2:
            # boolean
            return VariableDef(
                name=name,
                type="boolean",
            )

        if len(items) == 4:
            # integer .. integer
            lower = items[1]
            upper = items[2]

            if isinstance(lower, IntLiteral):
                lower = lower.value

            if isinstance(upper, IntLiteral):
                upper = upper.value

            return VariableDef(
                name=name,
                type="integer",
                lower=lower,
                upper=upper,
            )

        # enum
        return VariableDef(
            name=name,
            type="enum",
            values=items[-1],
        )

    def obsvardef(self, items):
        return list(items)

    def envvardef(self, items):
        return list(items)

    def vardef(self, items):
        return list(items)

    # --------------------------------------------------------
    # Observable local variables
    # --------------------------------------------------------

    def lobsvardef(self, items):
        if not items:
            return []

        return items[0]

    # --------------------------------------------------------
    # Environment actions
    # --------------------------------------------------------

    def envactiondef(self, items):
        if not items:
            return []

        return items[0]

    def actiondef(self, items):
        if not items:
            return []

        return items[0]

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
                # We distinguish variable lists from action lists
                # based on their contents.
                if item and isinstance(item[0], VariableDef):
                    if not observable_vars:
                        observable_vars = item
                    else:
                        vars_ = item

                elif item and isinstance(item[0], str):
                    actions = item

                elif not item:
                    continue

            elif isinstance(item, Protocol):
                protocol = item

            elif isinstance(item, list):
                evolution = item

            elif isinstance(item, object):
                if isinstance(item, list):
                    evolution = item

        # More reliable extraction by type.
        for item in items:
            if isinstance(item, Protocol):
                protocol = item

            elif isinstance(item, list):
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

        return Environment(
            observable_vars=observable_vars,
            vars=vars_,
            red_states=red_states,
            actions=actions,
            protocol=protocol,
            evolution=evolution,
        )

    # --------------------------------------------------------
    # Red states
    # --------------------------------------------------------

    def enreddef(self, items):
        if not items:
            return None
        return items[0]

    def reddef(self, items):
        if not items:
            return None
        return items[0]

    # --------------------------------------------------------
    # Protocol
    # --------------------------------------------------------

    def enprotdeflist(self, items):
        return list(items)

    def protdeflist(self, items):
        return list(items)

    def enprotline(self, items):
        return ProtocolRule(
            condition=items[0],
            actions=items[1],
        )

    def protline(self, items):
        return ProtocolRule(
            condition=items[0],
            actions=items[1],
        )

    def otherbranch(self, items):
        return OtherRule(
            actions=items[0],
        )

    def envprotdef(self, items):
        rules = []
        other = None

        for item in items:
            if isinstance(item, list):
                rules.extend(item)

            elif isinstance(item, ProtocolRule):
                rules.append(item)

            elif isinstance(item, OtherRule):
                other = item

        return Protocol(
            rules=rules,
            other=other,
        )

    def protdef(self, items):
        rules = []
        other = None

        for item in items:
            if isinstance(item, list):
                rules.extend(item)

            elif isinstance(item, ProtocolRule):
                rules.append(item)

            elif isinstance(item, OtherRule):
                other = item

        return Protocol(
            rules=rules,
            other=other,
        )

    # --------------------------------------------------------
    # Evolution
    # --------------------------------------------------------

    def envevline(self, items):
        return EvolutionRule(
            result=items[0],
            condition=items[1],
        )

    def evline(self, items):
        return EvolutionRule(
            result=items[0],
            condition=items[1],
        )

    def envevdef(self, items):
        return list(items)

    def evdef(self, items):
        return list(items)

    # --------------------------------------------------------
    # Agents
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
                    if not vars_:
                        vars_ = item
                    else:
                        local_observable_vars = item

                elif all(isinstance(x, str) for x in item):
                    if not local_observable_vars:
                        actions = item

                elif all(isinstance(x, EvolutionRule) for x in item):
                    evolution = item

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
    # Evaluation
    # --------------------------------------------------------

    def evaline(self, items):
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

    def istates(self, items):
        return InitialStates(
            condition=items[0],
        )

    # --------------------------------------------------------
    # Groups
    # --------------------------------------------------------

    def groupline(self, items):
        return Group(
            name=str(items[0]),
            members=items[1],
        )

    def groups(self, items):
        return Groups(
            groups=list(items),
        )

    # --------------------------------------------------------
    # References / values
    # --------------------------------------------------------

    def environment_variable(self, items):
        return Ref(
            owner="Environment",
            name=str(items[0]),
        )

    def dotted_id(self, items):
        return Ref(
            owner=str(items[0]),
            name=str(items[1]),
        )

    def varvalue1(self, items):
        return items[0]

    def varvalue2(self, items):
        return items[0]

    def varvalue3(self, items):
        return items[0]

    def varvalue4(self, items):
        return items[0]

    # --------------------------------------------------------
    # Boolean / arithmetic expressions
    # --------------------------------------------------------

    def _binary(self, items):
        if len(items) == 1:
            return items[0]

        return Binary(
            op=str(items[1]),
            left=items[0],
            right=items[2],
        )

    def expr1(self, items):
        return self._binary(items)

    def term1(self, items):
        return self._binary(items)

    def expr2(self, items):
        return self._binary(items)

    def term2(self, items):
        return self._binary(items)

    def expr3(self, items):
        return self._binary(items)

    def term3(self, items):
        return self._binary(items)

    def expr4(self, items):
        return self._binary(items)

    def term4(self, items):
        return self._binary(items)

    def expr5(self, items):
        return self._binary(items)

    def term5(self, items):
        return self._binary(items)

    def expr6(self, items):
        return self._binary(items)

    def term6(self, items):
        return self._binary(items)

    def factor4(self, items):
        if len(items) == 1:
            return items[0]

        return Unary(
            op="~",
            operand=items[0],
        )

    def factor5(self, items):
        if len(items) == 1:
            return items[0]

        return Unary(
            op="~",
            operand=items[0],
        )

    def factor6(self, items):
        if len(items) == 1:
            return items[0]

        return Unary(
            op="~",
            operand=items[0],
        )

    def element4(self, items):
        return items[0]

    def element5(self, items):
        return items[0]

    def element6(self, items):
        return items[0]

    def element1(self, items):
        return items[0]

    def element2(self, items):
        return items[0]

    def element3(self, items):
        return items[0]

    # --------------------------------------------------------
    # Logical conditions
    # --------------------------------------------------------

    def enlboolcond(self, items):
        return self._logical(items)

    def lboolcond(self, items):
        return self._logical(items)

    def eboolcond(self, items):
        return self._logical(items)

    def gboolcond(self, items):
        return self._logical(items)

    def evaboolcond(self, items):
        return self._logical(items)

    def isboolcond(self, items):
        return self._logical(items)

    def _logical(self, items):
        if len(items) == 1:
            return items[0]

        if len(items) == 2:
            return Unary(
                op="!",
                operand=items[1],
            )

        return Binary(
            op=str(items[1]),
            left=items[0],
            right=items[2],
        )

    # --------------------------------------------------------
    # Assignments in evolution
    # --------------------------------------------------------

    def boolresult(self, items):
        if len(items) == 1:
            return items[0]

        if len(items) == 2:
            return items[0]

        return Binary(
            op=str(items[1]),
            left=items[0],
            right=items[2],
        )

    def boolresult1(self, items):
        if len(items) == 1:
            return items[0]

        if len(items) == 2:
            return items[0]

        return Binary(
            op=str(items[1]),
            left=items[0],
            right=items[2],
        )

    # --------------------------------------------------------
    # Atoms
    # --------------------------------------------------------

    def formula(self, items):
        return self._formula(items)

    def fformula(self, items):
        return self._formula(items)

    def ltlformula(self, items):
        return self._formula(items)

    def ctls_state_formula(self, items):
        return self._formula(items)

    def ctls_path_formula(self, items):
        return self._formula(items)

    def _formula(self, items):
        if len(items) == 1:
            item = items[0]

            if isinstance(item, str):
                return Atom(item)

            return item

        if len(items) == 2:
            first = str(items[0])

            # Unary temporal operators
            if first in {
                "AG", "EG", "AX", "EX",
                "AF", "EF",
                "G", "F", "X",
            }:
                return TemporalFormula(
                    quantifier="",
                    operator=first,
                    operand=items[1],
                )

            return UnaryFormula(
                op=first,
                operand=items[1],
            )

        # Binary formulas
        if len(items) == 3:
            op = str(items[1])

            if op in {"and", "or", "->", "U"}:
                return BinaryFormula(
                    op=op,
                    left=items[0],
                    right=items[2],
                )

        return items[0]

    # --------------------------------------------------------
    # Formula containers
    # --------------------------------------------------------

    def form_list(self, items):
        return list(items)

    def formlist(self, items):
        return list(items)

    def formulae(self, items):
        return Formulae(
            formulas=list(items),
        )

    def fairformulae(self, items):
        return Fairness(
            formulas=list(items),
        )

    # --------------------------------------------------------
    # LTL / CTL*
    # --------------------------------------------------------

    def ltlformula_wrapper(self, items):
        return LTLFormula(items[0])

    def ctlsformula(self, items):
        return CTLStarFormula(items[0])

    # --------------------------------------------------------
    # Knowledge
    # --------------------------------------------------------

    def knowledge(self, items):
        return KnowledgeFormula(
            operator=str(items[0]),
            agent=str(items[1]),
            operand=items[2],
        )

    # --------------------------------------------------------
    # Generic start/model
    # --------------------------------------------------------

    def interpreted_system(self, items):
        semantics = None
        environment = None
        agents = []
        evaluation = None
        initial_states = None
        groups = None
        fairness = None
        formulae = None

        for item in items:
            if isinstance(item, str):
                if item in {"SingleAssignment", "MultiAssignment"}:
                    semantics = item

            elif isinstance(item, Environment):
                environment = item

            elif isinstance(item, Agent):
                agents.append(item)

            elif isinstance(item, Evaluation):
                evaluation = item

            elif isinstance(item, InitialStates):
                initial_states = item

            elif isinstance(item, Groups):
                groups = item

            elif isinstance(item, Fairness):
                fairness = item

            elif isinstance(item, Formulae):
                formulae = item

        return Model(
            semantics=semantics,
            environment=environment,
            agents=agents,
            evaluation=evaluation,
            initial_states=initial_states,
            groups=groups,
            fairness=fairness,
            formulae=formulae,
        )


# ============================================================
# Parser
# ============================================================

class ISPLParser:
    """
    High-level ISPL parser.

    Usage:

        parser = ISPLParser()

        model = parser.parse(text)

        model = parser.parse_file("model.ispl")
    """

    def __init__(self, grammar_path: Optional[str | Path] = None):
        if grammar_path is None:
            grammar_path = Path(__file__).with_name("ispl.lark")

        grammar_path = Path(grammar_path)

        self._parser = Lark.open(
            grammar_path,
            parser="earley",
            lexer="dynamic",
            ambiguity="resolve",
            start="start",
        )

    def parse(self, text: str) -> Model:
        tree = self._parser.parse(text)

        return ISPLTransformer().transform(tree)

    def parse_file(
        self,
        path: str | Path,
        encoding: str = "utf-8",
    ) -> Model:
        path = Path(path)

        with path.open("r", encoding=encoding) as f:
            return self.parse(f.read())


# ============================================================
# Convenience functions
# ============================================================

_default_parser: Optional[ISPLParser] = None


def get_parser() -> ISPLParser:
    global _default_parser

    if _default_parser is None:
        _default_parser = ISPLParser()

    return _default_parser


def parse(text: str) -> Model:
    return get_parser().parse(text)


def parse_file(
    path: str | Path,
    encoding: str = "utf-8",
) -> Model:
    return get_parser().parse_file(path, encoding=encoding)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    import argparse
    import pprint

    parser = argparse.ArgumentParser(
        description="Parse an ISPL file and print its AST."
    )

    parser.add_argument(
        "file",
        help="Path to the ISPL file",
    )

    args = parser.parse_args()

    model = parse_file(args.file)

    pprint.pp(model)


if __name__ == "__main__":
    main()