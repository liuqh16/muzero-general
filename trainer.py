import copy

import numpy
import torch

import models


class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
        self.model.to(torch.device("cuda" if self.config.train_on_gpu else "cpu"))
        self.model.train()

        self.training_step = initial_checkpoint["training_step"]

        if "cuda" not in str(next(self.model.parameters()).device):
            print("You are not training on GPU.\n")

        # Initialize the optimizer
        if self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.lr_init,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr_init,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"{self.config.optimizer} is not implemented. You can change the optimizer manually in trainer.py."
            )

        if initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )

    def update_weights(self, batch):
        """
        Perform one training step.
        """

        (
            observation_batch,
            action_batch,
            updater_batch,
            target_value,
            target_reward,
            target_policy,
            weight_batch,
            gradient_scale_batch,
        ) = batch

        # Keep values as scalars for calculating the priorities for the prioritized replay
        target_value_scalar = numpy.array(target_value, dtype="float32")
        priorities = numpy.zeros_like(target_value_scalar)

        device = next(self.model.parameters()).device
        if self.config.PER:
            weight_batch = torch.tensor(weight_batch.copy()).float().to(device)
        observation_batch = torch.tensor(numpy.array(observation_batch)).float().to(device)
        action_batch = torch.tensor(action_batch).long().to(device)
        if updater_batch is not None:
            updater_batch = torch.tensor(updater_batch).long().to(device)
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)
        gradient_scale_batch = torch.tensor(gradient_scale_batch).float().to(device)
        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1, num_agents
        # updater_batch: batch, num_unroll_steps+1
        # target_value: batch, num_unroll_steps+1
        # target_reward: batch, num_unroll_steps+1
        # target_policy: batch, num_unroll_steps+1, len(action_space)
        # gradient_scale_batch: batch, num_unroll_steps+1

        target_value = models.scalar_to_support(target_value, self.config.support_size)
        target_reward = models.scalar_to_support(
            target_reward, self.config.support_size
        )
        # target_value: batch, num_unroll_steps+1, 2*support_size+1
        # target_reward: batch, num_unroll_steps+1, 2*support_size+1

        ## Generate predictions
        value, reward, policy_logits, hidden_state = self.model.initial_inference(
            observation_batch
        )
        if updater_batch is not None:
            policy_logits = policy_logits[range(policy_logits.shape[0]), updater_batch[:, 0]]
        predictions = [(value, reward, policy_logits)]
        for i in range(1, action_batch.shape[1]):
            value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                hidden_state, action_batch[:, i]
            )
            # Scale the gradient at the start of the dynamics function (See paper appendix Training)
            hidden_state.register_hook(lambda grad: grad * 0.5)
            if updater_batch is not None:
                policy_logits = policy_logits[range(policy_logits.shape[0]), updater_batch[:, i]]
            predictions.append((value, reward, policy_logits))
        # predictions: num_unroll_steps+1, 3, batch, 2*support_size+1 | 2*support_size+1 | len(action_space) (according to the 2nd dim)

        ## Compute losses
        value_loss, reward_loss, policy_loss = (0, 0, 0)
        value, reward, policy_logits = predictions[0]
        # Ignore reward loss for the first batch step
        current_value_loss, _, current_policy_loss = self.loss_function(
            value.squeeze(-1),
            reward.squeeze(-1),
            policy_logits,
            target_value[:, 0],
            target_reward[:, 0],
            target_policy[:, 0],
        )
        value_loss += current_value_loss
        policy_loss += current_policy_loss
        # Compute priorities for the prioritized replay (See paper appendix Training)
        pred_value_scalar = (
            models.support_to_scalar(value, self.config.support_size)
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )
        priorities[:, 0] = (
            numpy.abs(pred_value_scalar - target_value_scalar[:, 0])
            ** self.config.PER_alpha
        )

        for i in range(1, len(predictions)):
            value, reward, policy_logits = predictions[i]
            (
                current_value_loss,
                current_reward_loss,
                current_policy_loss,
            ) = self.loss_function(
                value.squeeze(-1),
                reward.squeeze(-1),
                policy_logits,
                target_value[:, i],
                target_reward[:, i],
                target_policy[:, i],
            )

            # Scale gradient by the number of unroll steps (See paper appendix Training)
            current_value_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            current_reward_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            current_policy_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )

            value_loss += current_value_loss
            reward_loss += current_reward_loss
            policy_loss += current_policy_loss

            # Compute priorities for the prioritized replay (See paper appendix Training)
            pred_value_scalar = (
                models.support_to_scalar(value, self.config.support_size)
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            priorities[:, i] = (
                numpy.abs(pred_value_scalar - target_value_scalar[:, i])
                ** self.config.PER_alpha
            )

        if self.config.PER:
            # Correct PER bias by using importance-sampling (IS) weights
            value_loss *= weight_batch
            reward_loss *= weight_batch
            policy_loss *= weight_batch

        # Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
        loss = value_loss * self.config.value_loss_weight + reward_loss + policy_loss

        # Mean over batch dimension (pseudocode do a sum)
        loss = loss.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return (
            priorities,
            # For log purpose
            loss.item(),
            value_loss.mean().item(),
            reward_loss.mean().item(),
            policy_loss.mean().item(),
        )

    def update_lr(self):
        """
        Update learning rate
        """
        lr = self.config.lr_init * self.config.lr_decay_rate ** (
            self.training_step / self.config.lr_decay_steps
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    def loss_function(
        value,
        reward,
        policy_logits,
        target_value,
        target_reward,
        target_policy,
    ):
        # Cross-entropy seems to have a better convergence than MSE
        value_loss = (-target_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1)
        reward_loss = (-target_reward * torch.nn.LogSoftmax(dim=1)(reward)).sum(1)
        policy_loss = (-target_policy * torch.nn.LogSoftmax(dim=1)(policy_logits)).sum(
            1
        )
        return value_loss, reward_loss, policy_loss
