import random
import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel
from stv.parsers.parser_stv_v2 import ExprNode, ModalExprNode
from stv.parsers.stv_specification import *

_DEFAULT_PARAMS = {
    "spec": None,
    "formula": None
}
_NUM_PLAYERS = 2
_NUM_ROWS = 3
_NUM_COLS = 3
_NUM_CELLS = _NUM_ROWS * _NUM_COLS
_GAME_TYPE = pyspiel.GameType(
    short_name="atl_model",
    long_name="atl_model",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,  # SEQUENTIAL
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,  # REWARDS | TERMINAL
    max_num_players=1000,
    min_num_players=1,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=False,
    provides_observation_tensor=False,
    parameter_specification=_DEFAULT_PARAMS)



class AtlModelGame(pyspiel.Game):
    """A Python version of the Tic-Tac-Toe game."""

    def __init__(self, params, silent=True):
        assert "spec" in params, "Model specification not provided to the Game constructor!"
        assert "formula" in params, "Formula not provided to the Game constructor!"
        self.spec = params["spec"]
        self.formula = params["formula"]
        # self.spec = spec
        # self.formula = formula
        self.silent = silent
        self.agent_actions, self.action_name_to_id_dict = self.get_agent_actions_dict(self.spec)
        self.possible_actions = self.get_possible_actions(self.spec, self.agent_actions)
        self._GAME_INFO = pyspiel.GameInfo(
            num_distinct_actions=len(self.possible_actions),
            max_chance_outcomes=0,
            num_players=len(self.spec.agents),  # potentially +1 because of environment
            min_utility=-1.0,
            max_utility=1.0,
            utility_sum=0.0,
            max_game_length=100)
        # Persistent variables and states in the observation vector will be in the alphabetical order
        self.persistent_variables_ordered = {self.get_player_index(a.name): sorted(a.local_variables_init_values.keys()) for a in self.spec.agents}
        self.persistent_variables_index_per_player = {k: {n: i for i, n in enumerate(sorted_vars)} for k, sorted_vars in self.persistent_variables_ordered.items()}
        self.nodes_ordered = {self.get_player_index(a.name): sorted(a.state_names()) for a in self.spec.agents}
        self.nodes_index_per_player = {k: {n: i for i, n in enumerate(sorted_nodes)} for k, sorted_nodes in self.nodes_ordered.items()}
        super().__init__(_GAME_TYPE, self._GAME_INFO, {})

    def get_player_name(self, player_index):
        """Converts a player's number ID in Open Spiel to an identifier used for actions."""
        return self.spec.agents[player_index].name

    def get_player_index(self, player_name):
        """Converts a player's name to a number ID used in Open Spiel."""
        for i, a in enumerate(self.spec.agents):
            if a.name == player_name:
                return i
        raise Exception(f"Player '{player_name}' was not found.")

    def num_players(self):
        return self._GAME_INFO.num_players

    def get_max_tensor_size(self):
        max_total_size = None
        for k in self.persistent_variables_ordered:
            Q = len(self.nodes_ordered[k])
            P = len(self.persistent_variables_ordered[k])
            total_size = Q + P
            if max_total_size is None or total_size > max_total_size:
                max_total_size = total_size
        return max_total_size

    @staticmethod
    def from_spec(spec: StvSpecification, formula: ModalExprNode, params=None, silent=True):
        if params is None:
            params = {}
        params["spec"] = spec
        params["formula"] = formula
        return AtlModelGame(params, silent=silent)

    @staticmethod
    def get_agent_actions_dict(spec):
        ACTION_ID = 0
        agent_actions = {}
        action_name_to_id_dict = {}
        for i, a in enumerate(spec.agents):
            agent_actions[a.name] = []
            action_name_to_id_dict[a.name] = {}
            for t in a.transitions:
                if t.is_shared and t.local_name not in agent_actions[a.name]:
                    agent_actions[a.name].append(t.local_name)
                    action_name_to_id_dict[a.name][t.local_name] = ACTION_ID
                    ACTION_ID += 1
                elif not t.is_shared and t.name not in agent_actions[a.name]:
                    # Action with the same name can be used in multiple states
                    agent_actions[a.name].append(t.name)
                    action_name_to_id_dict[a.name][t.name] = ACTION_ID
                    ACTION_ID += 1
        return agent_actions, action_name_to_id_dict

    @staticmethod
    def get_possible_actions(spec, agent_actions):
        actions = []
        for a in spec.agents:
            actions.extend(agent_actions[a.name])
        return actions

    # -----------------------------------------------------------------------------------
    # Below is the implementation of Game's API in Open Spiel

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return AtlModelState(game=self,
                             spec=self.spec,
                             formula=self.formula,
                             agent_actions=self.agent_actions,
                             action_name_to_id_dict=self.action_name_to_id_dict,
                             possible_actions=self.possible_actions,
                             silent=self.silent)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        if ((iig_obs_type is None) or
                (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
            if params is None:
                params = {}
            params["game"] = self
            return AtlModelStateObserver(params)
        else:
            return IIGObserverForPublicInfoGame(iig_obs_type, params)




class AgentLocalState:
    def __init__(self,
                 agent_id: int,
                 agent_spec: AgentLocalModelSpec,
                 game: AtlModelGame,
                 max_tensor_size: int|None = None):
        self.agent_spec = agent_spec
        self.id = agent_id
        self.name = agent_spec.name
        self.current_node: str = agent_spec.init_state
        self.persistent_variables = agent_spec.local_variables_init_values.copy()
        self.persistent_variables_ordered = game.persistent_variables_ordered[self.id].copy()
        self.persistent_variables_index = game.persistent_variables_index_per_player[self.id].copy()
        self.nodes_ordered = game.nodes_ordered[self.id].copy()
        self.nodes_index = game.nodes_index_per_player[self.id].copy()
        self.max_tensor_size = max_tensor_size
        # TODO: local non-persistent variables are currently not handled


    def get_num_nodes(self):
        return len(self.nodes_ordered)

    def get_num_variables(self):
        return len(self.persistent_variables_ordered)

    def execute_action(self, name):
        """Executes an action and changes agent's local state. Name can be either a name (private actions) or
         local name (synchronized actions)."""
        # TODO: this probably should be updated by an external function for the shared actions
        for t in self.agent_spec.transitions:
            if t.name == name and not t.is_shared:
                self.current_node = t.q1.name
                for k, v in t.q1.parameters.items():
                    self.persistent_variables[k] = v
                break
            elif t.local_name == name and t.is_shared:
                # TODO: local name can correspond to a set of actions taken by other players
                self.current_node = t.q1.name
                for k, v in t.q1.parameters.items():
                    self.persistent_variables[k] = v
                break

    def execute_transition(self, transition: Transition):
        self.current_node = transition.q1.name
        # Change persistent variables (if applicable)
        for k, v in transition.q1.parameters.items():
            self.persistent_variables[k] = v

    def is_precondition_satisified(self, trans: Transition):
        """Checks, if a precondition of a transition is satisfied."""
        for k, v in trans.q0.parameters.items():
            if self.persistent_variables[k] != v:
                return False
        return True

    def get_available_transitions(self) -> list[Transition]:
        private_trans = []
        shared_trans = []
        shared_trans_dict = {}
        for t in self.agent_spec.transitions:
            # t_name = t.local_name if t.is_shared else t.name
            if self.current_node == t.q0.name and self.is_precondition_satisified(t):
                if t.is_shared:
                    shared_trans_dict.setdefault(t.local_name, [])
                    shared_trans_dict[t.local_name].append(t)
                else:
                    private_trans.append(t)

        for _, transitions in shared_trans_dict.items():
            shared_trans.append(SharedTransition.from_transition_set(transitions))
        return private_trans + shared_trans

    def get_transition_for_action(self, action_name) -> Transition | SharedTransition | None:
        """Returns action's transition object based on a current node. Since actions can be executed in multiple
         nodes, a corresponding transition object cannot be determined without the context of the current node."""
        for t in self.get_available_transitions():
            if t.is_shared and t.local_name == action_name:
                assert t.is_abstract
                return t  # this is an abstract transition
            elif not t.is_shared and t.name == action_name:
                return t
        return None

    def __str__(self):
        return f"{self.name} (#{self.id}): {self.current_node} [{','.join([f'{k}={v}' for k, v in self.persistent_variables.items()])}]"

    def information_state_tensor(self):
        """Returns information state tensor of this agent."""
        Q = self.get_num_nodes()
        P = self.get_num_variables()
        if self.max_tensor_size is not None:
            tensor = np.zeros(self.max_tensor_size, np.float32)
        else:
            tensor = np.zeros(Q+P, np.float32)
        t_nodes = tensor[0:Q]
        t_variables = tensor[Q:Q + P]

        node_index = self.nodes_index[self.current_node]
        t_nodes[node_index] = 1.0
        for var_name, i in self.persistent_variables_index.items():
            t_variables[i] = self.persistent_variables[var_name]
        return tensor

    def information_state_string(self):
        return str(self)


class AtlModelState(pyspiel.State):
    """A state of the planning game. It is modified in place after each action."""

    def __init__(self, game: AtlModelGame, spec: StvSpecification, formula: ModalExprNode, agent_actions=None, action_name_to_id_dict=None,
                 possible_actions=None, seed=None, silent=True):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        if seed is not None:
            random.seed(seed)
        if agent_actions is None or action_name_to_id_dict is None:
            agent_actions, action_name_to_id_dict = AtlModelGame.get_agent_actions_dict(spec)
        if possible_actions is None:
            possible_actions = AtlModelGame.get_possible_actions(spec, agent_actions)
        self.agent_actions = agent_actions
        self.action_name_to_id_dict = action_name_to_id_dict
        self.possible_actions = possible_actions
        self.ARTIFICIAL_PLAYER_ID = len(spec.agents)
        self._silent = silent
        self._cur_player = pyspiel.PlayerId.SIMULTANEOUS
        self._cur_obs = None
        self._cur_executable_steps_dict = None
        self._cur_num_steps = 0
        self._is_terminal = False

        # self.game = game
        self.spec = spec
        self.formula = formula
        self.agent_local_states = [AgentLocalState(i, a, game, max_tensor_size=game.get_max_tensor_size()) for i, a in enumerate(self.spec.agents)]
        self.previous_global_state = self.get_global_state()

    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every perfect-information sequential-move game.

    def get_global_state(self):
        agent_states = [a.current_node for a in self.agent_local_states]
        global_variables = self.combine_agents_variables()
        return agent_states, global_variables

    def get_player_name(self, player_index):
        """Converts a player's number ID in Open Spiel to an identifier used for actions."""
        return self.spec.agents[player_index].name

    def get_player_index(self, player_name):
        """Converts a player's name to a number ID used in Open Spiel."""
        for i, a in enumerate(self.spec.agents):
            if a.name == player_name:
                return i
        raise Exception(f"Player '{player_name}' was not found.")

    def get_action_id(self, player, action_name):
        pass

    def get_action_name(self, action_id):
        return self.possible_actions[action_id]

    def _select_current_player(self):
        """Selects current player using semantics of ATL as implemented in STV. In order to do that, synchronizations
        are selected based on repertoires.
         """
        pass

    # ----------------------------------------------------------
    # legacy

    def _print(self, text):
        if not self._silent:
            print(text)


    # -----------------------------------------------------------------------------------
    # Below is the implementation of State's API in Open Spiel

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else pyspiel.PlayerId.SIMULTANEOUS

    def _legal_actions(self, player):
        """Returns a list of legal actions, sorted in ascending order. In simultaneous games
         possible actions for each player are generated using function."""
        assert player >= 0
        player_name = self.get_player_name(player)
        actions = []
        for t in self.agent_local_states[player].get_available_transitions():
            t_name = t.local_name if t.is_shared else t.name
            action_idx = self.action_name_to_id_dict[player_name][t_name]
            actions.append(action_idx)
        return actions

    # def _apply_action(self, action):
    #     """Applies the specified action to the state."""
    #     if self._is_terminal:
    #         raise Exception("Trying to execute an action after terminal state was reached!")
    #     raise Exception("Trying to execute a single action in a game with simultaneous moves!")


    def _execute_agent_actions(self, actions):
        shared_transactions = []
        was_action_executed = False  # Can be used to detect deadlock

        # Execute all selected private actions - these agents, under imperfect information, won't get any new
        # information to decide, so we may as well execute them.
        #
        # It is also important to note that agents can choose a shared action even if another agent required
        # for synchronization is not in a state in which it can execute it. At no point agents are aware of the
        # state of other agents.
        for player, action in enumerate(actions):
            action_name = self.possible_actions[action]
            transition = self.agent_local_states[player].get_transition_for_action(action_name)
            if transition.is_shared:
                assert transition.is_abstract
                # At this point we have an abstract shared transition which can correspond to any of the
                # concrete transitions contained in it (e.g., A in "action1[A] ... action2[A]").
                # Agent can only decide on A and isn't aware of the underlying actions.
                shared_transactions.append((player, transition))
            else:
                self.execute_transition(player, transition)
                was_action_executed = True

        # Aggregate instances of shared actions
        shared_trans_support_dict = {}
        for p, at in shared_transactions:
            # p = 1  (player id of the action executioner)
            # at - represents play_0 in the example below
            # at.transition_set = {play_0_rock, play_0_paper, play_0_scissors}
            #   Player0:
            #       shared[3] play_0_rock[play_0_rock]: idle -> finish
            #       shared[3] play_0_paper[play_0_paper]: idle -> finish
            #       shared[3] play_0_scissors[play_0_scissors]: idle -> finish
            #   Player1:
            #       shared[3] play_0_rock[play_0]: idle -> idle2
            #       shared[3] play_0_paper[play_0]: idle -> idle2
            #       shared[3] play_0_scissors[play_0]: idle -> idle2

            # Aggregate over name and collect all support
            assert at.is_abstract
            for t in at.transition_set:
                assert not t.is_abstract
                shared_trans_support_dict.setdefault(t.name, [])
                shared_trans_support_dict[t.name].append((p, t))

        # Check if they can be executed, and if yes then execute them.
        # ASSUMPTION #1: there is always a "leader" initiating synchronization and in that way ensuring that
        # there is always only a single possible choice of action.
        # ASSUMPTION #2: in the specification, x in "shared[x]" always denotes a correct number of synchronizing
        # agents - this is not enforced by the parser.
        enabled_shared = []
        for _, support in shared_trans_support_dict.items():
            # Check if all the required players choose to synchronize.
            if support[0][1].shared_num == len(support):
                for p, t in support:
                    enabled_shared.append((p, t))

        # Execute all enabled shared transitions. We can do that because of our "leader" assumption. In general
        # this won't work, because agents can support multiple concrete transitions, and then we happen to need
        # to decide, which transition gets executed, potentially blocking other from being executed.
        # I think detecting such a situation would necessitate a chance player.
        for p, t in enabled_shared:
            assert not t.is_abstract
            self.execute_transition(p, t)
            was_action_executed = True

        return was_action_executed


    def _is_formula_satisfied_interpreter(self, formula, global_variables):
        if isinstance(formula, ModalExprNode):
            if formula.modal_op != "<>":
                raise Exception("Currently only <> modal operator is supported!")
            return self._is_formula_satisfied_interpreter(formula.formula, global_variables)
        elif len(formula.args) == 0:
            if formula.is_variable:
                return global_variables[formula.name]
            else:
                return formula.name
        elif len(formula.args) == 1:
            X = self._is_formula_satisfied_interpreter(formula.args[0], global_variables)
            if formula.name == "!":
                return not X
            else:
                raise Exception("Incorrect expression node!")
        elif len(formula.args) == 2:
            L = self._is_formula_satisfied_interpreter(formula.args[0], global_variables)
            R = self._is_formula_satisfied_interpreter(formula.args[1], global_variables)
            if formula.name == "==":
                return L == R
            elif formula.name == ">=":
                return L >= R
            elif formula.name == "<=":
                return L <= R
            elif formula.name == "&&":
                return R and L
            elif formula.name == "||":
                return R or L
            else:
                raise Exception("Incorrect expression node!")
        else:
            raise Exception("Incorrect expression node!")


    def combine_agents_variables(self):
        global_variables = {}
        for a in self.agent_local_states:
            for k, v in a.persistent_variables.items():
                global_variables[k] = v
        return global_variables


    def is_formula_satisfied(self, formula: ModalExprNode|None = None):
        if formula is None:
            formula = self.formula
        global_variables = self.combine_agents_variables()
        return self._is_formula_satisfied_interpreter(formula, global_variables)


    def _apply_actions(self, actions):
        """Execute simultaneous actions."""
        if self._is_terminal:
            raise Exception("Trying to execute actions in a finished game!")

        self._execute_agent_actions(actions)

        y = self.is_formula_satisfied(self.formula)
        if self.formula.modal_op == "[]" and not y or\
           self.formula.modal_op == "<>" and y:
            # case []: coalition failed to ensure property
            # case <>: coalition managed to achieve property
            self._is_terminal = True

        new_global_state = self.get_global_state()
        if new_global_state == self.previous_global_state:
            if not self._silent:
                print("GAME ENTERED CYCLE (global state didn't change)")
            self._is_terminal = True
        else:
            self.previous_global_state = new_global_state

    def execute_transition(self, player, transition):
        if not self._silent:
            print(f"*** Executing transition for player {player}: {transition}")
        self.agent_local_states[player].execute_transition(transition)

    def _action_to_string(self, player, action):
        """Action -> string."""
        action_name = self.possible_actions[action]
        return "{}".format(action_name)

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._is_terminal

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        return self.rewards()

    def rewards(self):
        """Returns reward from the most recent state transition (s, a, s') for all players."""
        if not self._is_terminal:
            return [0.0 for _ in self.agent_local_states]
        else:
            if self.formula.modal_op == "[]":
                raise Exception("[] modal operator is currently not handled!")
            else:
                if self.is_formula_satisfied():
                    # Coalition won
                    return [1.0 if a.name in self.formula.coalition else -1.0 for a in self.agent_local_states]
                else:
                    return [-1.0 if a.name in self.formula.coalition else 1.0 for a in self.agent_local_states]

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        text = "\n".join([f"{a.name}: {a.current_node} (vars: {a.persistent_variables})" for a in self.agent_local_states])
        return text

    def information_state_string(self, player):
        return self.agent_local_states[player].information_state_string()

    def information_state_tensor(self, player):
        return self.agent_local_states[player].information_state_tensor()




class AtlModelStateObserver:
    """Observer, conforming to the PyObserver interface (see observation.py).

    Assumptions for the type of ATL models that we consider (asynchronous, imperfect information,
    imperfect recall):
    - Each agent has access only to the persistent values of its private variables and in which state it currently is.
    - Tensor representing observations for the given agent is represented as: Q + P, where Q is the set of all names
     of states, and P are the values of all persistent variables.
    """

    def __init__(self, params):
        """Initializes an empty observation tensor."""
        self.game: AtlModelGame = params.get("game")
        self.persistent_variables_index_per_player = self.game.persistent_variables_index_per_player
        self.nodes_index_per_player = self.game.nodes_index_per_player

        # Build the single flat tensor, as suggested in a github issue (https://github.com/google-deepmind/open_spiel/issues/815):
        # "For different observations / infostates the common way is to make it a constant size equal to the maximum
        # across players and then just store different information in them (see the Sheriff game as an example)."
        max_total_size = None
        for k in self.game.persistent_variables_ordered:
            Q = self._get_num_nodes(k)
            P = self._get_num_variables(k)
            total_size = Q + P
            if max_total_size is None or total_size > max_total_size:
                max_total_size = total_size
        self.tensor = np.zeros(max_total_size, np.float32)

        # The observation should contain a 1-D tensor in `self.tensor` and a
        # dictionary of views onto the tensor, which may be of any shape.
        # Here the observation is indexed `(cell state, row, column)`.
        self.dict = {"observation": self.tensor}
        # self.dict["nodes"] = self.tensor[0:1]  # dummy value, we don't know agent
        # self.dict["variables"] = self.tensor[0:1]  # dummy value, we don't know agent

    def _get_num_nodes(self, player):
        return len(self.game.nodes_ordered[player])

    def _get_num_variables(self, player):
        return len(self.game.persistent_variables_ordered[player])

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        self.tensor.fill(0)
        Q = self._get_num_nodes(player)
        P = self._get_num_variables(player)
        self.dict["nodes"] = self.tensor[0:Q]
        self.dict["variables"] = self.tensor[Q:Q+P]

        node_index = self.nodes_index_per_player[player][state.agent_local_states[player].current_node]
        self.dict["nodes"][node_index] = 1.0
        for var_name, i in self.persistent_variables_index_per_player[player].items():
            self.dict["variables"][i] = state.agent_local_states[player].persistent_variables[var_name]

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        return str(state)





# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, AtlModelGame) # is registering needed?
