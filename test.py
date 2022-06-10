import torch
import numpy
import self_play
from models import MuZeroNetwork
from games.gfootball import Game, MuZeroConfig


config = MuZeroConfig()
model = MuZeroNetwork(config)
num_agents = config.num_agents

with torch.no_grad():

    game = Game()
    game_history = self_play.GameHistory(num_agents=num_agents)
    observation = game.reset()
    init_action = 0 if num_agents == 1 else [0] * num_agents
    game_history.action_history.append(init_action)
    game_history.observation_history.append(observation)
    game_history.reward_history.append(0)
    game_history.to_play_history.append(game.to_play())

    done = False

    stacked_observations = game_history.get_stacked_observations(-1, config.stacked_observations)

    root, mcts_info = self_play.MCTS(config).run(
        model,
        stacked_observations,
        game.legal_actions(),
        game.to_play(),
        True,
    )
