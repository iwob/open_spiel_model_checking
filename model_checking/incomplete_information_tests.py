import numpy as np
import pyspiel
import pickle

from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner
import pyspiel




if __name__ == '__main__':
    # In Kuhn poker, the deck includes only three playing cards, for example, a King, Queen, and Jack.
    # One card is dealt to each player, which may place bets similarly to a standard poker.
    # If both players bet or both players pass, the player with the higher card wins, otherwise,
    # the betting player wins.
    game = pyspiel.load_game("kuhn_poker")

    state = game.new_initial_state()
    player_id = state.current_player()
    print(f"(Player: {player_id}) Legal actions:")
    for action in state.legal_actions():
        print(f"{action}: {state.action_to_string(state.current_player(), action)}")


    # Create the environment
    env = rl_environment.Environment(game)
    num_players = env.num_players
    num_actions = env.action_spec()["num_actions"]


    LOAD_TRAINED_MODELS = False

    if LOAD_TRAINED_MODELS:
        with open("agent0_qlearning.pkl", 'rb') as f:
            agent0 = pickle.load(f)
        with open("agent1_qlearning.pkl", 'rb') as f:
            agent1 = pickle.load(f)
        agents = [agent0, agent1]
    else:
        # Create the agents
        agents = [
            tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
            for idx in range(num_players)
        ]

        # Train the Q-learning agents in self-play.
        for cur_episode in range(10000):  # around 100000 gives 100% on quality tests
            if cur_episode % 1000 == 0:
                print(f"Episodes: {cur_episode}")
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step)
                time_step = env.step([agent_output.action])
            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)
        print("Done!")
        print("\n\n")

        # Saving on disk
        with open("agent0_qlearning.pkl", 'wb') as f:
            pickle.dump(agents[0], f)
        with open("agent1_qlearning.pkl", 'wb') as f:
            pickle.dump(agents[1], f)



    # Evaluate the Q-learning agent against a random agent.
    from open_spiel.python.algorithms import random_agent

    eval_agents = [agents[0], random_agent.RandomAgent(1, num_actions, "Entropy Master 2000")]

    time_step = env.reset()
    # time_step.observations["info_state"]  to get the vector representing a state
    while not time_step.last():
        print("State:")
        print(env.get_state)
        player_id = time_step.observations["current_player"]

        state = env.get_state
        legal_actions = state.legal_actions()
        print(f"(Player: {player_id}) Legal actions:")
        for action in legal_actions:
            print(f"{action}: {state.action_to_string(state.current_player(), action)}")

        # stored as a defaultdict from state (represented as string?) to a defaultdict from actions to probabilities
        q_values = agents[player_id]._q_values
        # observations in time_step contain a separate list per each agent, which I suppose allows to model
        # imperfect information
        obs = time_step.observations["info_state"][player_id]
        obs = str(obs)
        print("obs:", obs)
        if obs not in q_values:
            print("obs not in q_values")
        else:
            print("qvalues:", q_values[obs])

        # Note the evaluation flag. A Q-learner will set epsilon=0 here.
        agent_output = eval_agents[player_id].step(time_step, is_evaluation=True)
        print(f"Agent {player_id} chooses {env.get_state.action_to_string(agent_output.action)}")
        time_step = env.step([agent_output.action])

    print("")
    print(env.get_state)
    print(time_step.rewards)
