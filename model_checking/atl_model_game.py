import random
import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel
from stv.parsers.stv_specification import StvSpecification, AgentLocalModel


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



class AgentLocalState:
    def __init__(self, agent_spec: AgentLocalModel):
        self.agent_spec = agent_spec
        self.name = agent_spec.name
        self.current_node = agent_spec.init_state
        self.persistent_variables = agent_spec.persistent_variables

    def execute_action(self, name):
        """Executes an action and changes agent's local state. Name can be either a name (private actions) or
         local name (synchronized actions)."""
        # TODO: this probably should be updated by an external function
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

    def get_available_actions(self):
        result = []
        for t in self.agent_spec.transitions:
            # if self.current_node == t.q0.name and (not t.is_shared or not any([t.local_name == t2.local_name for t2 in result])):
            if self.current_node == t.q0.name and (not t.is_shared or t.local_name in result):
                for k, v in t.q0.parameters.items():
                    if self.persistent_variables[k] != v:
                        break
                result.append(t)
        return result


class AtlModelGame(pyspiel.Game):
    """A Python version of the Tic-Tac-Toe game."""

    def __init__(self, spec: StvSpecification, params=None, silent=True):
        self.spec = spec
        self.silent = silent
        self._GAME_INFO = pyspiel.GameInfo(
            num_distinct_actions=100,
            max_chance_outcomes=0,
            num_players=len(self.spec.agents) + 1,
            min_utility=-1.0,
            max_utility=1.0,
            utility_sum=0.0,
            max_game_length=100000)
        super().__init__(_GAME_TYPE, self._GAME_INFO, params or dict())


    def _get_positive_literals(self) -> list:
        literals = []
        # Literals were sorted by a parser, so that they are always in the same order
        for lit in self.space.literals:
            if lit[0] == -1 or lit[0] == "not":
                continue
            else:
                literals.append(lit)
        return literals

    def get_player_name(self, player_index):
        """Converts a player's number ID in Open Spiel to an identifier used for actions."""
        return self.agentSpecs[player_index].name

    def get_player_index(self, player_name):
        """Converts a player's name to a number ID used in Open Spiel."""
        for i, a in enumerate(self.agentSpecs):
            if a.name == player_name:
                return i
        raise Exception(f"Player '{player_name}' was not found.")

    # -----------------------------------------------------------------------------------
    # Below is the implementation of Game's API in Open Spiel

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return AtlModelState(game=self, spec=self.spec, silent=self.silent)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        if ((iig_obs_type is None) or
                (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
            params["positive_literals"] = self.positive_literals
            params["initial_state"] = self.problem.initial_state
            return PlanningStateObserver(params)
        else:
            return IIGObserverForPublicInfoGame(iig_obs_type, params)





class AtlModelState(pyspiel.State):
    """A state of the planning game. It is modified in place after each action."""

    def __init__(self, game: AtlModelGame, spec: StvSpecification, seed=None, silent=True):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        if seed is not None:
            random.seed(seed)
        self.ARTIFICIAL_PLAYER_ID = len(spec.agents)
        self._silent = silent
        self._cur_player = None  # value set in agent_env_execution_loop()
        self._cur_obs = None  # value set in agent_env_execution_loop()
        self._cur_executable_steps_dict = None  # value set in agent_env_execution_loop()
        self._cur_num_steps = 0
        self._is_terminal = False

        self.game = game
        self.spec = spec
        self.agent_local_states = [AgentLocalState(a) for a in self.spec.agents]

        for a in self.agent_local_states:
            print(f"{a.name}:")
            print("\n".join([str(x) for x in a.get_available_actions()]))

        # Initialization of the game
        self.agent_env_execution_loop()  # _cur_player is set here


    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every perfect-information sequential-move game.

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
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

    def _legal_actions(self, player):
        """Returns a list of legal actions, sorted in ascending order."""
        player_name = self.game.get_player_name(player)
        return sorted([s.step_id for s in self._cur_executable_steps_dict[player_name]])

    def _apply_action(self, action):
        """Applies the specified action to the state."""
        if self._is_terminal:
            raise Exception("Trying to execute an action after terminal state was reached!")
        step_to_execute = self.game.actionable_steps[action]
        self.execute_step(step_to_execute)

        self._print(f"\nExecuting step: {step_to_execute}")
        self._print(self.text_state_change_diff(self.prev_state, self.state))
        self.print_state_content()

        # Selecting an agent to move and creating an observation vector for it
        self.agent_env_execution_loop()

    def _do_apply_actions(self, actions):
        # Handle simultanoues actions here
        pass

    def _action_to_string(self, player, action):
        """Action -> string."""
        player_name = self.game.get_player_name(player)
        executed_action = self.game.actionable_steps[action]
        return "{}: {}".format(player_name, executed_action)

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._is_terminal

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        return self.player_rewards

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        if self.game.textRenderer is None:
            return str(self.state)
        else:
            return self.game.textRenderer(self.state)


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
