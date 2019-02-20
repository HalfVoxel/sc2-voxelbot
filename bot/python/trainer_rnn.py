import torch
import torch.nn as nn
from training_algorithm import TrainingAlgorithm


class TrainerRNN(TrainingAlgorithm):
    def __init__(self, model, optimizer, action_loss_weights, device, sampleClass, stepperClass, step_size=2):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.time_threshold = 0
        self.step_size = step_size
        self.device = device
        self.sampleClass = sampleClass
        self.stepperClass = stepperClass
        self.loss_fn = nn.NLLLoss(weight=None if action_loss_weights is None else torch.tensor(
            action_loss_weights, dtype=torch.float, device=device), reduction='none')

    def train(self, batch):
        self.model.train()
        total_loss = 0
        steps = 0

        # for loss, losses_by_time in self.evaluate_batch_iter(batch, max_steps_per_update=10):
        for loss in [self.evaluate_batch(batch)]:
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # for param in policy_net.parameters():
            #     param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            total_loss += loss.item()
            steps += 1

        if steps > 0:
            total_loss /= steps
        return total_loss

    def test(self, batch):
        self.model.eval()
        with torch.no_grad():
            loss = self.evaluate_batch(batch)
            return loss.item()

    def eval_batch(self, states):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(states, dtype=torch.float)).detach().numpy()

    def evaluate_batch_iter(self, batch, max_steps_per_update):
        stepper = self.stepperClass(self.model, batch, self.device, self.time_threshold, self.loss_fn, self.step_size)
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
        stepper = self.stepperClass(self.model, batch, self.device, self.time_threshold, self.loss_fn, self.step_size)
        total_loss = torch.zeros(1, requires_grad=False, device=self.device)
        loss_counter = 0
        while True:
            res = stepper.step()
            if res is not None:
                loss, steps = res
                total_loss = total_loss + loss
                loss_counter += steps
            else:
                if loss_counter > 0:
                    total_loss = total_loss / loss_counter
                return total_loss
