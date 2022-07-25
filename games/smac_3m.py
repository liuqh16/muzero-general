import os
import datetime
from typing import List, Tuple

import gym
import numpy
import torch
import numpy as np

from .abstract_game import AbstractGame
from smac.env import StarCraft2Env

try:
    import cv2
except ModuleNotFoundError:
    raise ModuleNotFoundError('\nPlease run "pip install gym[atari]"')


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = 1  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.num_agents = 3
        self.observation_shape = (1, 1, 48)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(9))   # Fixed list of all possible actions for single agent. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 8  # Number of previous observations and previous actions to add to the current observation
        self.final_reward_cover = False

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 100  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 1000  # Maximum number of moves if game is not finished before
        self.num_simulations = 100  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "ma-fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Fully Connected Network
        self.encoding_size = 64
        self.fc_representation_layers = [128, 64, 64]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64, 64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [32, 32]  # Define the hidden layers in the reward network
        self.fc_value_layers = [32, 32]  # Define the hidden layers in the value network
        self.fc_policy_layers = [32, 32]  # Define the hidden layers in the policy network



        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = int(1000e3)  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 1024  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 100  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available
        # print("GPU use:", torch.cuda.is_available())

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.05  # Initial learning rate
        self.lr_decay_rate = 0.1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 350e3



        ### Replay Buffer
        self.replay_buffer_size = int(1e5)  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 5  # Number of game moves to keep for every batch element
        self.td_steps = 10  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 1  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 500e3:
            return 1.0
        elif trained_steps < 750e3:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = StarCraft2Env(map_name="3m", seed=seed)
        env_info = self.env.get_env_info()
        self.n_agents = env_info["n_agents"]    # 3
        self.n_actions = env_info["n_actions"]  # 14
        self.info = {}

    def reset(self) -> np.ndarray:
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        self.env.reset()
        self.info = {}
        state = np.expand_dims(self.env.get_state(), axis=(0, 1))  # (1, 1, 48)
        return state

    def step(self, action) -> Tuple[np.ndarray, float, bool]:
        """
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        reward, terminated, self.info = self.env.step(action)
        state = np.expand_dims(self.env.get_state(), axis=(0, 1))  # (1, 1, 48)
        return state, reward, terminated

    def terminate_info(self):
        return self.info

    def to_play(self) -> int:
        """
        Return the current player. (element of MuZeroConfig.players)

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return 0

    def legal_actions(self) -> List[List[int]]:
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        Returns:
            for Multi-Agent settings, return a list of legal action arrays w.r.t. each agent
        """
        legal_actions = []  # type: List[List[int]]
        for agent_id in range(self.n_agents):
            avail_actions = self.env.get_avail_agent_actions(agent_id)
            avail_actions_ind = np.nonzero(avail_actions)[0].tolist()   # type: List[int]
            legal_actions.append(avail_actions_ind)
        return legal_actions

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        # input("Press enter to take a step ")

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()
