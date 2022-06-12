import os
import copy

import torch
import numpy

import models
import self_play
from games.gfootball import Game, MuZeroConfig


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = MuZeroConfig()
model = models.MuZeroNetwork(config)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

num_agents = config.num_agents
trained_steps = 0
temperature = config.visit_softmax_temperature_fn(trained_steps)
temperature_threshold = config.temperature_threshold
render = False
opponent = "self"
muzero_player = 0

game = Game()
game_history = self_play.GameHistory(num_agents=num_agents)
observation = game.reset()
init_action = 0 if num_agents == 1 else [0] * num_agents
game_history.action_history.append(init_action)
game_history.observation_history.append(observation)
game_history.reward_history.append(0)
game_history.to_play_history.append(game.to_play())
if num_agents > 1:
    game_history.updater_idx = numpy.random.choice(num_agents, 2)

done = False

with torch.no_grad():
    while (
        not done and len(game_history.action_history) <= config.max_moves
    ):

        assert (
            len(numpy.array(observation).shape) == 3
        ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(observation).shape)} dimensionnal. Got observation of shape: {numpy.array(observation).shape}"
        assert (
            numpy.array(observation).shape == config.observation_shape
        ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {config.observation_shape} but got {numpy.array(observation).shape}."
        stacked_observations = game_history.get_stacked_observations(
            -1,
            config.stacked_observations
        )

        # Choose the action
        if opponent == "self" or muzero_player == game.to_play():
            root, mcts_info = self_play.MCTS(config).run(
                model,
                stacked_observations,
                game.legal_actions(),
                game.to_play(),
                True,
                updater_idx=game_history.updater_idx
            )
            action = self_play.SelfPlay.select_action(
                root,
                temperature
                if not temperature_threshold
                or len(game_history.action_history) < temperature_threshold
                else 0,
                num_agents=num_agents
            )

            if render:
                print(f'Tree depth: {mcts_info["max_tree_depth"]}')
                print(
                    f"Root value for player {game.to_play()}: {root.value():.2f}"
                )
        else:
            pass

        observation, reward, done = game.step(action)

        game_history.store_search_statistics(root, config.action_space)

        # Next batch
        game_history.action_history.append(action)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(reward)
        game_history.to_play_history.append(game.to_play())
