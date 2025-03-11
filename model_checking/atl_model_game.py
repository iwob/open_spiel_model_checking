import random
import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel
from stv.parsers.stv_specification import *


_NUM_PLAYERS = 2
_NUM_ROWS = 3
_NUM_COLS = 3
_NUM_CELLS = _NUM_ROWS * _NUM_COLS
_GAME_TYPE = pyspiel.GameType(
    short_name="planning_game",
    long_name="planning_game",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,  # TERMINAL
    max_num_players=1000,
    min_num_players=1,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={})



class AtlModelGame(pyspiel.Game):
    """A Python version of the Tic-Tac-Toe game."""

    def __init__(self, spec: StvSpecification, params=None, silent=True):
        self.spec = spec
        self.silent = silent
        self.agent_actions, self.action_name_to_id_dict = self.get_agent_actions_dict(spec)
        self.possible_actions = self.get_possible_actions(spec, self.agent_actions)
        self._GAME_INFO = pyspiel.GameInfo(
            num_distinct_actions=len(self.possible_actions),
            max_chance_outcomes=0,
            num_players=len(self.spec.agents) + 1,
            min_utility=-1.0,
            max_utility=1.0,
            utility_sum=0.0,
            max_game_length=100000)
        super().__init__(_GAME_TYPE, self._GAME_INFO, params or dict())

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

    def _get_positive_literals(self) -> list:
        literals = []
        # Literals were sorted by a parser, so that they are always in the same order
        for lit in self.space.literals:
            if lit[0] == -1 or lit[0] == "not":
                continue
            else:
                literals.append(lit)
        return literals

    # -----------------------------------------------------------------------------------
    # Below is the implementation of Game's API in Open Spiel

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return AtlModelState(game=self,
                             agent_actions=self.agent_actions,
                             action_name_to_id_dict=self.action_name_to_id_dict,
                             possible_actions=self.possible_actions,
                             spec=self.spec,
                             silent=self.silent)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        if ((iig_obs_type is None) or
                (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
            params["positive_literals"] = self.positive_literals
            params["initial_state"] = self.problem.initial_state
            return PlanningStateObserver(params)
        else:
            return IIGObserverForPublicInfoGame(iig_obs_type, params)




class AgentLocalState:
    def __init__(self, agent_spec: AgentLocalModelSpec):
        self.agent_spec = agent_spec
        self.name = agent_spec.name
        self.current_node = agent_spec.init_state
        self.persistent_variables = agent_spec.local_variables_init_values
        # TODO: local variables are currently not handled

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

    def get_transition_for_action(self, action_name) -> Transition | None:
        """Returns action's transition object based on a current node. Since actions can be executed in multiple
         nodes, a corresponding transition object cannot be determined without the context of the current node."""
        for t in self.get_available_transitions():
            if t.is_shared and t.local_name == action_name:
                assert t.is_abstract
                return t  # this is an abstract transition
            elif not t.is_shared and t.name == action_name:
                return t
        return None



class AtlModelState(pyspiel.State):
    """A state of the planning game. It is modified in place after each action."""

    def __init__(self, game: AtlModelGame, spec: StvSpecification, agent_actions=None, action_name_to_id_dict=None,
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

        self.game = game
        self.spec = spec
        self.agent_local_states = [AgentLocalState(a) for a in self.spec.agents]

        for a in self.agent_local_states:
            print(f"{a.name}:")
            print("\n".join([str(x) for x in a.get_available_transitions()]))


    def get_player_name(self, player_index):
        """Converts a player's number ID in Open Spiel to an identifier used for actions."""
        return self.spec.agents[player_index].name

    def get_player_index(self, player_name):
        """Converts a player's name to a number ID used in Open Spiel."""
        for i, a in enumerate(self.spec.agents):
            if a.name == player_name:
                return i
        raise Exception(f"Player '{player_name}' was not found.")

    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every perfect-information sequential-move game.

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

    def _apply_actions(self, actions):
        """Execute simultaneous actions."""
        shared_trans_buffer = []
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
                # At this point we have an abstract shared transition which can correspond to any of the concrete transitions contained in it
                shared_trans_buffer.append((player, transition))
            else:
                self.execute_transition(player, transition)
                was_action_executed = True

        # Aggregate instances of shared actions
        shared_trans_support_dict = {}
        for p, at in shared_trans_buffer:
            # TODO: it doesn't make sense to aggregate over local names
            # IDEA: instead, aggregate over name and collect there all support
            assert at.is_abstract
            for t in at.transition_set:
                shared_trans_support_dict.setdefault(t.name, (t, p, []))
                shared_trans_support_dict[t.name][2].append([(p,t)])

        # Check if they can be executed, and if yes then execute them
        enabled_shared = []
        for tname, data in shared_trans_support_dict.items():
            t, p, support_trans = data
            # check if all the required players choose to synchronize
            # ASSUMPTION: there is always a "leader" initiating synchronization and in that way ensuring that
            # there is always only a single choice.
            if shared_trans_support_dict[tname][0].shared_num == len(shared_trans_support_dict[tname]):
                enabled_shared.append((p, tname))

        # Execute all enabled shared transitions
        # Because of our "leader" assumption, we simply executre every enabled transition. In general this won't work,
        # because agents can support multiple concrete transition and be absorbed by the first execution
        for p, t in enabled_shared:
            assert not t.is_abstract
            self.execute_transition(p, t)
            was_action_executed = True

        # Detect deadlock
        # TODO: Looping final transitions will need to be also somehow handled here
        if not was_action_executed:
            print("DEADLOCK")
            # self._is_terminal = True

    def execute_transition(self, player, transition):
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
        return self.player_rewards

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        text = "\n".join([f"{a.name}: {a.current_node} (vars: {a.persistent_variables})" for a in self.agent_local_states])
        return text


class PlanningStateObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, params):
        """Initializes an empty observation tensor."""
        self.positive_literals = params.get("positive_literals", None)
        initial_state = params.get("initial_state", None)
        if self.positive_literals is None or initial_state is None:
            raise ValueError(f"Set of positive literals or initial state was not specified")

        # The observation should contain a 1-D tensor in `self.tensor` and a
        # dictionary of views onto the tensor, which may be of any shape.
        # Here the observation is indexed `(cell state, row, column)`.
        self.tensor = np.zeros(len(self.positive_literals), np.float32)
        for i, lit in enumerate(self.positive_literals):
            if lit in initial_state.predicates:
                self.tensor[i] = 1.0
        self.dict = {"observation": self.tensor}

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        # del player
        # We update the observation via the shaped tensor since indexing is more
        # convenient than with the 1-D tensor. Both are views onto the same memory.
        self.tensor = np.zeros(len(self.positive_literals), np.float32)
        for i, lit in enumerate(self.positive_literals):
            if lit in state.state.predicates:
                self.tensor[i] = 1.0
        self.dict = {"observation": self.tensor}
        # obs = self.dict["observation"]
        # obs.fill(0)
        # for row in range(_NUM_ROWS):
        #     for col in range(_NUM_COLS):
        #         cell_state = ".ox".index(state.board[row, col])
        #         obs[cell_state, row, col] = 1

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        # del player
        return str(state)





# Register the game with the OpenSpiel library

# pyspiel.register_game(_GAME_TYPE, PlanningGameGame) # is registering needed?
