

class State:
    def __init__(self, name: str, parameters: dict = None):
        """

        :param name: Name of a state.
        :param parameters: A dictionary assigning to each parameter a value it assumes in this state.
        """
        self.name = name
        self.parameters = parameters if parameters is not None else {}
        self._hash = hash((self.name, frozenset(self.parameters.items())))

    def __str__(self):
        if len(self.parameters) > 0:
            lst = [f'{k}={self.parameters[k]}' for k in self.parameters]
            return f"{self.name} ({', '.join(lst)})"
        else:
            return f"{self.name}"

    def __repr__(self):
        return "State({}, {})".format(self.name, self.parameters)

    def __eq__(self, other):
        if isinstance(other, tuple):
            other = State(other[0], other[1])
        return self.name == other.name and self.parameters == other.parameters

    def __hash__(self):
        return self._hash


class Transition:
    def __init__(self, name: str, q0: State, q1: State, is_shared=False, shared_num=None, local_name=None):
        self.name = name
        self.local_name = local_name
        self.q0 = q0
        self.q1 = q1
        self.is_shared = is_shared
        self.shared_num = shared_num

    def __str__(self):
        return self.stv2_string()

    def __repr__(self):
        name_str = f"{self.name}[{self.local_name}]" if self.is_shared else self.name
        return f"Transition({name_str}, {self.q0}, {self.q1}, {self.is_shared}, {self.shared_num})"

    def stv2_string(self):
        """Returns encoding of this transition in the STV-2 format."""
        text_shared = ""
        text_q0_params = ""
        text_q1_params = ""
        if self.is_shared:
            text_shared = f"shared " if self.shared_num is None else f"shared[{self.shared_num}] "

        if len(self.q0.parameters) > 0:
            text_q0_params = " [" + ", ".join([f"{k}=={v}" for k, v in self.q0.parameters.items()]) + "]"
        text_q0 = f"{self.q0.name}{text_q0_params}"

        if len(self.q1.parameters) > 0:
            text_q1_params = " [" + ", ".join([f"{k}:={v}" for k, v in self.q1.parameters.items()]) + "]"
        text_q1 = f"{self.q1.name}{text_q1_params}"

        if self.is_shared:
            return f"{text_shared}{self.name}[{self.local_name}]: {text_q0} -> {text_q1}"
        else:
            return f"{text_shared}{self.name}: {text_q0} -> {text_q1}"



class AgentLocalModelSpec:
    def __init__(self, name: str, num_instances: int, init_state: str, local_variables: dict,
                 persistent_variables: dict, local_variables_init_values: dict, transitions: list[Transition]):
        """

        :param local_variables: A dictionary containing local variables of agent's model and their initial values.
        :param transitions: a list of transitions.
        """
        self.name = name
        self.init_state = init_state
        self.num_instances = num_instances
        self.local_variables = local_variables
        self.persistent_variables = persistent_variables
        self.local_variables_init_values = local_variables_init_values
        self.transitions = set(transitions)

    def get_out_transitions(self, state):
        return [t for t in self.transitions if t.q0 == state]

    def get_in_transitions(self, state):
        return [t for t in self.transitions if t.q1 == state]

    def states(self) -> set[State]:
        s = set()
        for t in self.transitions:
            s.add(t.q0)
            s.add(t.q1)
        return s

    def nonterminal_states(self) -> set[State]:
        s = set()
        for t in self.transitions:
            s.add(t.q0)
            # s.add(t.q1)  # q1 is not added, because if it is nonterminal it will at least once happen to be q0 in some transition
        return s

    def cut_transitions(self, strategy: dict):
        """Removes from the local model transitions that do not adhere to the provided strategy."""
        for state, strategy_trans in strategy.items():
            print(f"Processing state: {state}")
            out_trans = self.get_out_transitions(state)
            for t in out_trans:
                if t != strategy_trans:
                    print(f"\tRemoving transition: {t}  (strategy transition: {strategy_trans})")
                    self.transitions.remove(t)



class StvSpecification:
    def __init__(self, agents: list[AgentLocalModelSpec]):
        self.agents = agents

    def __iter__(self):
        return self.agents

    def has_agent(self, name: str):
        for a in self.agents:
            if name == a.name:
                return True
        return False

    def get_agent_model(self, name: str):
        for a in self.agents:
            if a.name == name:
                return a
        return None

    def strategy_cut(self, strategy):
        """For a partial strategy represented as dict[<agent_name>, dict[<state object>, <transition object>]] removes
         from the model all transitions that are excluded by the strategy."""
        for a, strat in strategy.items():
            agent_model = self.get_agent_model(a)
            if agent_model is None:
                raise Exception("Trying to access nonexistent agent!")
            agent_model.cut_transitions(strat)

    def get_stv2_encoding(self, formula):
        return generate_stv2_encoding(self, formula)



def generate_stv2_encoding(model: StvSpecification, formula: str):
    text = ""
    for i, a in enumerate(model.agents):
        if i > 0:
            text += "\n"
        text_name = a.name if a.num_instances is None else f"{a.name}[{a.num_instances}]"
        text_local_vars = ",".join(a.local_variables)
        text_local_vars_init = ",".join([f"{k}:={v}" for k, v in a.local_variables_init_values.items()])
        text +=\
f"""Agent {text_name}:
LOCAL: [{text_local_vars}]
INITIAL: [{text_local_vars_init}]
init {a.init_state}\n"""

        for t in a.transitions:
            text += t.stv2_string() + "\n"

    text += f"\nFORMULA: {formula}\n"
    return text