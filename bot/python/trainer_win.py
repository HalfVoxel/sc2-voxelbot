import torch
import numpy as np
import torch.nn as nn
from replay_memory import ReplayMemory, Transition
from training_algorithm import TrainingAlgorithm


class TrainerWinPredictor(TrainingAlgorithm):
    def __init__(self, model, optimizer, action_loss_weights):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.time_threshold = 0
        self.loss_fn = nn.NLLLoss(weight=None if action_loss_weights is None else torch.tensor(action_loss_weights, dtype=torch.float), reduction='none')

    def train(self, batch):
        self.model.train()
        loss, losses_by_time = self.evaluate_batch(batch)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item(), losses_by_time

    def test(self, memory):
        self.model.eval()
        batch = memory.get_all()
        loss, losses_by_time = self.evaluate_batch(batch)
        return loss.item(), losses_by_time

    def eval(self, state):
        self.model.eval()
        _, outputs = self.evaluate_batch([state])
        return outputs.detach().numpy()[0,:]

    def eval_batch(self, states):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(states, dtype=torch.float)).detach().numpy()

    def evaluate_batch(self, batch):
        timestep = 0
        total_loss = 0
        total_steps = 0
        losses_by_time = []
        last_probs = torch.tensor(np.ones(len(batch)) * 0.5, dtype=torch.float)
        hidden_states = torch.zeros((len(batch), self.model.gru_hidden_size))
        while True:
            in_progress = np.array([timestep < len(trace.states) for trace in batch])
            progress_indices = np.flatnonzero(in_progress)
            states = [trace.states[timestep] for trace in batch if timestep < len(trace.states)]
            minimap_states = [trace.minimap_states[timestep] for trace in batch if timestep < len(trace.states)]
            labels = [trace.winner for trace in batch if timestep < len(trace.states)]
            lengths = [len(trace.states) for trace in batch if timestep < len(trace.states)]
            passes_threshold = np.array([timestep >= self.time_threshold * len(trace.states) for trace in batch if timestep < len(trace.states)])
            passes_threshold_indices = np.flatnonzero(passes_threshold)

            if len(states) == 0:
                break

            batch_inputs = torch.tensor(states, dtype=torch.float)
            batch_inputs2 = torch.tensor(minimap_states, dtype=torch.float)
            assert batch_inputs.size() == (len(states), *states[0].shape)
            batch_outputs = torch.tensor(labels, dtype=torch.long)

            outputs, hidden_states[progress_indices] = self.model(batch_inputs, batch_inputs2, hidden_states[progress_indices])
            losses = self.loss_fn(outputs, batch_outputs)
            for i in range(len(states)):
                losses_by_time.append((timestep/lengths[i], losses[i].detach().numpy(), batch[progress_indices[i]]))

            new_probs = losses.exp()
            similarity_loss = (last_probs[progress_indices] - new_probs).pow(2).sum()
            last_probs[progress_indices] = new_probs

            total_loss += losses[passes_threshold_indices].sum() + similarity_loss * 0.2
            total_steps += len(passes_threshold_indices)

            timestep += 2

        total_loss /= total_steps

        return total_loss, losses_by_time
