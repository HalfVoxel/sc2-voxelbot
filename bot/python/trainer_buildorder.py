import torch
import numpy as np
import torch.nn as nn
from replay_memory import ReplayMemory, Transition
from training_algorithm import TrainingAlgorithm


class PredictBuildOrder(TrainingAlgorithm):
    def __init__(self, model, optimizer, action_loss_weights):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = nn.NLLLoss(weight=None if action_loss_weights is None else torch.tensor(action_loss_weights, dtype=torch.float))

    def train(self, memory: ReplayMemory, batch_size: int):
        self.model.train()
        batch = memory.sample(batch_size)
        loss, _ = self.evaluate_batch(batch)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def test(self, memory):
        self.model.eval()
        batch = memory.get_all()
        loss, _ = self.evaluate_batch(batch)
        return loss.item()

    def eval(self, state):
        self.model.eval()
        _, outputs = self.evaluate_batch([state])
        return outputs.detach().numpy()[0,:]

    def eval_batch(self, states):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(states, dtype=torch.float)).detach().numpy()

    def evaluate_batch(self, batch):
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        transposed_batch = Transition(*zip(*batch))

        batch_inputs = torch.tensor(transposed_batch.state, dtype=torch.float)
        batch_outputs = torch.tensor(transposed_batch.action, dtype=torch.long)

        outputs = self.model(batch_inputs)
        loss = self.loss_fn(outputs, batch_outputs)
        return loss, outputs
