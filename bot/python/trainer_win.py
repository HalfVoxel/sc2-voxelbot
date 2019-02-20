import torch
import numpy as np
import torch.nn as nn
from replay_memory import ReplayMemory, Transition
from training_algorithm import TrainingAlgorithm


class Stepper:
    def __init__(self, model, batch, device, time_threshold, loss_fn):
        self.batch = batch
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.time_threshold = time_threshold
        self.timestep = 0
        self.losses_by_time = []
        self.last_probs = 0.5 * torch.ones(len(batch), dtype=torch.float, device=self.device)
        self.hidden_states = self.model.init_hidden_states(len(batch), device=self.device)

    def detach(self):
        self.losses_by_time = []
        self.hidden_states = self.hidden_states.detach()
        self.last_probs = self.last_probs.detach()

    def step(self):
        in_progress = np.array([self.timestep < len(trace.states) for trace in self.batch])
        progress_indices = torch.tensor(np.flatnonzero(in_progress), device=self.device)
        active_batch = [trace for trace in self.batch if self.timestep < len(trace.states)]
        states = [trace.states[self.timestep] for trace in active_batch]
        # minimap_states = [trace.minimap_states[self.timestep] for trace in active_batch]
        raw_unit_states = [trace.raw_unit_states[self.timestep] for trace in active_batch]
        masks = [trace.masks[self.timestep] for trace in active_batch]
        labels = [trace.winner for trace in active_batch]
        lengths = [len(trace.states) for trace in active_batch]
        passes_threshold = np.array([self.timestep >= self.time_threshold * len(trace.states) for trace in active_batch])
        passes_threshold_indices = np.flatnonzero(passes_threshold)

        if len(states) > 0:
            batch_inputs = torch.tensor(states, dtype=torch.float, device=self.device)
            # batch_inputs2 = torch.tensor(minimap_states, dtype=torch.float, device=self.device)
            batch_inputs3 = torch.tensor(raw_unit_states, dtype=torch.float, device=self.device)
            maskTensors = torch.tensor(masks, dtype=torch.uint8, device=self.device)
            assert batch_inputs.size() == (len(states), *states[0].shape)
            batch_outputs = torch.tensor(labels, dtype=torch.long, device=self.device)

            outputs, self.hidden_states[progress_indices] = self.model(
                inputTensor=batch_inputs, rawUnitTensor=batch_inputs3, stateTensor=self.hidden_states[progress_indices], mask=maskTensors)
            losses = self.loss_fn(outputs, batch_outputs)
            progress_indices_cpu = progress_indices.cpu().numpy()

            for i in range(len(states)):
                self.losses_by_time.append(
                    (self.timestep / lengths[i], losses[i].detach().cpu().numpy(), self.batch[progress_indices_cpu[i]]))

            new_probs = losses.exp()
            similarity_loss = (self.last_probs[progress_indices] - new_probs).pow(2).sum()
            self.last_probs[progress_indices] = new_probs

            loss = losses[passes_threshold_indices].sum() + similarity_loss * 0.05
            steps = len(passes_threshold_indices)
            self.timestep += 2
            return loss, steps
        else:
            return None


class TrainerWinPredictor(TrainingAlgorithm):
    def __init__(self, model, optimizer, action_loss_weights, device):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.time_threshold = 0
        self.device = device
        self.loss_fn = nn.NLLLoss(weight=None if action_loss_weights is None else torch.tensor(
            action_loss_weights, dtype=torch.float), reduction='none')

    def train(self, batch):
        self.model.train()
        total_loss = 0
        total_losses_by_time = []
        steps = 0

        # for loss, losses_by_time in self.evaluate_batch_iter(batch, max_steps_per_update=10):
        for loss, losses_by_time in [self.evaluate_batch(batch)]:
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # for param in policy_net.parameters():
            #     param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            total_loss += loss.item()
            total_losses_by_time += losses_by_time
            steps += 1

        if steps > 0:
            total_loss /= steps
        return total_loss, total_losses_by_time

    def test(self, batch):
        self.model.eval()
        with torch.no_grad():
            loss, losses_by_time = self.evaluate_batch(batch)
            return loss.item(), losses_by_time

    def eval(self, state):
        self.model.eval()
        _, outputs = self.evaluate_batch([state])
        return outputs.detach().numpy()[0, :]

    def eval_batch(self, states):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(states, dtype=torch.float)).detach().numpy()

    def evaluate_batch_iter(self, batch, max_steps_per_update):
        stepper = Stepper(self.model, batch, self.device, self.time_threshold, self.loss_fn)
        counter = 0
        total_loss = torch.zeros(1, requires_grad=False, device=self.device)
        loss_counter = 0
        while True:
            res = stepper.step()
            if res is not None:
                loss, steps = res
                if steps > 0:
                    counter += 1

                total_loss += loss
                loss_counter += steps

            if res is None or counter >= max_steps_per_update:
                if loss_counter > 0:
                    total_loss = total_loss / loss_counter
                    print("Yield", total_loss, loss_counter)
                    yield total_loss, stepper.losses_by_time
                if res is None:
                    return

                stepper.detach()
                total_loss = torch.zeros(1, requires_grad=False, device=self.device)
                loss_counter = 0
                counter = 0

    def evaluate_batch(self, batch):
        stepper = Stepper(self.model, batch, self.device, self.time_threshold, self.loss_fn)
        total_loss = torch.zeros(1, requires_grad=False, device=self.device)
        loss_counter = 0
        while True:
            res = stepper.step()
            if res is not None:
                loss, steps = res
                total_loss += loss
                loss_counter += steps
            else:
                if loss_counter > 0:
                    total_loss = total_loss / loss_counter
                return total_loss, stepper.losses_by_time

        # timestep = 0
        # total_loss = torch.zeros(1, device=self.device)
        # total_steps = 0
        # losses_by_time = []
        # last_probs = 0.5 * torch.ones(len(batch), dtype=torch.float, device=self.device)
        # hidden_states = self.model.init_hidden_states(len(batch), device=self.device)
        # counter = 0
        # while True:
        #     in_progress = np.array([timestep < len(trace.states) for trace in batch])
        #     progress_indices = torch.tensor(np.flatnonzero(in_progress), device=self.device)
        #     active_batch = [trace for trace in batch if timestep < len(trace.states)]
        #     states = [trace.states[timestep] for trace in active_batch]
        #     # minimap_states = [trace.minimap_states[timestep] for trace in active_batch]
        #     raw_unit_states = [trace.raw_unit_states[timestep] for trace in active_batch]
        #     masks = [trace.masks[timestep] for trace in active_batch]
        #     labels = [trace.winner for trace in active_batch]
        #     lengths = [len(trace.states) for trace in active_batch]
        #     passes_threshold = np.array([timestep >= self.time_threshold * len(trace.states) for trace in active_batch])
        #     passes_threshold_indices = np.flatnonzero(passes_threshold)

        #     if len(states) > 0:
        #         batch_inputs = torch.tensor(states, dtype=torch.float, device=self.device)
        #         # batch_inputs2 = torch.tensor(minimap_states, dtype=torch.float, device=self.device)
        #         batch_inputs3 = torch.tensor(raw_unit_states, dtype=torch.float, device=self.device)
        #         maskTensors = torch.tensor(masks, dtype=torch.uint8, device=self.device)
        #         assert batch_inputs.size() == (len(states), *states[0].shape)
        #         batch_outputs = torch.tensor(labels, dtype=torch.long, device=self.device)

        #         outputs, hidden_states[progress_indices] = self.model(
        #             inputTensor=batch_inputs, rawUnitTensor=batch_inputs3, stateTensor=hidden_states[progress_indices], mask=maskTensors)
        #         losses = self.loss_fn(outputs, batch_outputs)
        #         progress_indices_cpu = progress_indices.cpu().numpy()
        #         for i in range(len(states)):
        #             losses_by_time.append(
        #                 (timestep / lengths[i], losses[i].detach().cpu().numpy(), batch[progress_indices_cpu[i]]))

        #         new_probs = losses.exp()
        #         similarity_loss = (last_probs[progress_indices] - new_probs).pow(2).sum()
        #         last_probs[progress_indices] = new_probs

        #         total_loss += losses[passes_threshold_indices].sum() + similarity_loss * 0.05
        #         total_steps += len(passes_threshold_indices)

        #         if len(passes_threshold_indices) > 0:
        #             counter += 1

        #     if len(states) == 0 or (max_steps_per_update is not None and counter >= max_steps_per_update):
        #         if total_steps == 0:
        #             if len(states) == 0:
        #                 # Explicitly return nothing, because we have no differentiable loss
        #                 return
        #             else:
        #                 continue

        #         total_loss /= total_steps

        #         if max_steps_per_update is not None:
        #             yield total_loss, losses_by_time
        #         else:
        #             return total_loss, losses_by_time

        #         counter = 0
        #         total_steps = 0
        #         losses_by_time = []
        #         total_loss = torch.zeros(1, device=self.device)
        #         hidden_states = hidden_states.detach()
        #         last_probs = last_probs.detach()

        #     timestep += 2
