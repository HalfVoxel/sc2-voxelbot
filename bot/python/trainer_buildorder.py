import torch
import numpy as np
import torch.nn as nn
from replay_memory import ReplayMemory, Transition
from training_algorithm import TrainingAlgorithm


class PredictBuildOrder(TrainingAlgorithm):
    def __init__(self, model, optimizer, action_loss_weights, device, sampleClass):
        super().__init__()
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.sampleClass = sampleClass
        self.loss_fn = nn.NLLLoss(weight=None if action_loss_weights is None else torch.tensor(action_loss_weights, dtype=torch.float))

    def train(self, batch):
        self.model.train()
        loss, _ = self.evaluate_batch(batch)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def train2(self, batch):
        self.model.train()
        loss, _ = self.evaluate_batch2(batch)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def test(self, batch):
        self.model.eval()
        loss, _ = self.evaluate_batch(batch)
        return loss.item()

    def test2(self, batch):
        self.model.eval()
        loss, _ = self.evaluate_batch2(batch)
        return loss.item()

    def eval(self, state):
        self.model.eval()
        _, outputs = self.evaluate_batch([state])
        return outputs.detach().numpy()[0, :]

    def eval_batch(self, states):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(states, dtype=torch.float)).detach().numpy()

    def evaluate_batch(self, batch):
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        transposed_batch = self.sampleClass(*zip(*batch))

        # for s in transposed_batch.state:
        #     s.pin_memory()

        # for s in transposed_batch.action:
        #     s.pin_memory()

        batch_inputs = torch.tensor(transposed_batch.state, dtype=torch.float)
        batch_outputs = torch.tensor(transposed_batch.action, dtype=torch.long)
        batch_inputs = batch_inputs.pin_memory()
        batch_outputs = batch_outputs.pin_memory()
        return self.evaluate_batch2(self.sampleClass(state=batch_inputs, action=batch_outputs))

    def evaluate_batch2(self, batch):
        batch_inputs = batch.state.to(device=self.device, non_blocking=True)
        batch_outputs = batch.action.to(device=self.device, non_blocking=True)
        outputs = self.model(batch_inputs)
        loss = self.loss_fn(outputs, batch_outputs)
        return loss, outputs
