from pathlib import Path
from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
import logging

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import random_agent
import pyspiel
from stv.parsers.parser_stv_v2 import Stv2Parser, parseAndTransformFormula
from atl_model_game import AtlModelGame

FLAGS = flags.FLAGS

# Training parameters
flags.DEFINE_string("checkpoint_dir", "/tmp/dqn_test","Directory to save/load the agent models.")
flags.DEFINE_integer("save_every", int(1e4),"Episode frequency at which the DQN agent models are saved.")
flags.DEFINE_integer("num_train_episodes", int(1e6),"Number of training episodes.")
flags.DEFINE_integer("eval_every", 1000, "Episode frequency at which the DQN agents are evaluated.")

# DQN model hyper-parameters
flags.DEFINE_list("hidden_layers_sizes", [64, 64], "Number of hidden units in the Q-Network MLP.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e5), "Size of the replay buffer.")
flags.DEFINE_integer("batch_size", 4, "Number of transitions to sample at each learning step.")  # was 32


def run_DQN(sess, game, load_saved_models=False):
    num_players = game.num_players()
    env = rl_environment.Environment(game)

    def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
        """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
        num_players = len(trained_agents)
        sum_episode_rewards = np.zeros(num_players)
        for player_pos in range(num_players):
            cur_agents = random_agents[:]
            cur_agents[player_pos] = trained_agents[player_pos]
            for _ in range(num_episodes):
                time_step = env.reset()
                episode_rewards = 0
                while not time_step.last():
                    player_id = time_step.observations["current_player"]
                    if env.is_turn_based:
                        agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
                        action_list = [agent_output.action]
                    else:
                        agents_output = [agent.step(time_step, is_evaluation=True) for agent in cur_agents]
                        action_list = [agent_output.action for agent_output in agents_output]
                    time_step = env.step(action_list)
                    episode_rewards += time_step.rewards[player_pos]
                sum_episode_rewards[player_pos] += episode_rewards
        return sum_episode_rewards / num_episodes

    info_state_size = game.get_max_tensor_size()
    num_actions = env.action_spec()["num_actions"]

    # random agents for evaluation
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    # with tf.Session() as sess:
    if True:
        hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
        # pylint: disable=g-complex-comprehension
        agents = [
            dqn.DQN(
                session=sess,
                player_id=idx,
                state_representation_size=info_state_size,
                num_actions=num_actions,
                hidden_layers_sizes=hidden_layers_sizes,
                replay_buffer_capacity=FLAGS.replay_buffer_capacity,
                min_buffer_size_to_learn=4,
                batch_size=FLAGS.batch_size) for idx in range(num_players)
        ]
        sess.run(tf.global_variables_initializer())

        if load_saved_models:
            for a in agents:
                logging.info("-----------------------------")
                a.restore(FLAGS.checkpoint_dir)
            return agents

        for ep in range(FLAGS.num_train_episodes):
            if (ep + 1) % FLAGS.eval_every == 0:
                r_mean = eval_against_random_bots(env, agents, random_agents, 1000)
                logging.info("[%s] Mean episode rewards %s", ep + 1, r_mean)
            if (ep + 1) % FLAGS.save_every == 0:
                for agent in agents:
                    agent.save(FLAGS.checkpoint_dir)

            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if env.is_turn_based:
                    agent_output = agents[player_id].step(time_step)
                    action_list = [agent_output.action]
                else:
                    agents_output = [agent.step(time_step) for agent in agents]
                    action_list = [agent_output.action for agent_output in agents_output]
                time_step = env.step(action_list)

            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)
    return agents


def execute_actions(env, actions, agents):
    time_step = env.reset()
    for a in actions:
        player_id = time_step.observations["current_player"]
        if env.is_turn_based:
            action_list = [a]
        else:
            action_list = a
        time_step = env.step(action_list)\


def test_set(game, agents):
    env = rl_environment.Environment(game)
    time_step = env.reset()
    # execute_actions([])
    agents_output = [agent.step(time_step, is_evaluation=True) for agent in agents]
    action_list = [agent_output.action for agent_output in agents_output]
    print("State:")
    print(str(time_step))
    for i, a in enumerate(agents):
        x = str(agents_output[i].probs).replace('\n', '')
        print(f"{game.spec.agents[i].name}: {x}")
        print(f"{game.spec.agents[i].name}: {game.possible_actions[agents_output[i].action]}")


def main(_):
    parser = Stv2Parser()
    file = Path(__file__).parent / "example_specifications" / "simple" / "simple.stv"
    with file.open() as f:
        text = f.read()
    stv_spec, formula = parser(text)
    game = AtlModelGame.from_spec(stv_spec, formula)

    # CFR (its implementation in OpenSpiel?) requires sequential games, so let's change simultaneous game into sequential one
    # game: pyspiel.Game = pyspiel.convert_to_turn_based(game)
    # pyspiel.register_game(game.get_type(), AtlModelGame)
    print("Registered names:")
    print(pyspiel.registered_names())

    state = game.new_initial_state()
    print("asdasdasdasdasdasdasd")
    print(str(state))
    print("----------------------")
    print(state.information_state_string(0))
    print(state.information_state_string(1))
    print(state.information_state_string(2))
    print("----------------------")
    print(state.information_state_tensor(0))
    print(state.information_state_tensor(1))
    print(state.information_state_tensor(2))
    print("----------------------")

    LOAD_TRAINED_AGENTS = True
    with tf.Session() as sess:
        agents = run_DQN(sess, game, load_saved_models=LOAD_TRAINED_AGENTS)

        test_set(game, agents)

    # cfr_solver = cfr.CFRSolver(game)
    #
    # for i in range(FLAGS.iterations):
    #   cfr_solver.evaluate_and_update_policy()
    #   if i % FLAGS.print_freq == 0:
    #     conv = exploitability.exploitability(game, cfr_solver.average_policy())
    #     print("Iteration {} exploitability {}".format(i, conv))


if __name__ == "__main__":
    app.run(main)
