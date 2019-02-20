import torch
import numpy as np
import torch.nn as nn


class MovementStepper:
    def __init__(self, model, batch, device, time_threshold, loss_fn, step_size):
        self.batch = batch
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.time_threshold = time_threshold
        self.step_size = step_size
        self.timestep = 0
        self.losses_by_time = []
        self.hidden_states = self.model.init_hidden_states(len(batch.states), device=self.device)
        self.outputs = None

        # MovementTargetTrace = namedtuple('MovementTargetTrace', ['states', 'target_positions', 'unit_type_counts', 'replay_path', 'minimap_states', 'data_path', 'playerID'])
        self.batch = type(self.batch)(
            states=[x.to(device=self.device, non_blocking=True) for x in self.batch.states],
            target_positions=[x.to(device=self.device, non_blocking=True) for x in self.batch.target_positions],
            unit_type_counts=[x.to(device=self.device, non_blocking=True) for x in self.batch.unit_type_counts],
            minimap_states=[x.to(device=self.device, non_blocking=True) for x in self.batch.minimap_states],
            replay_path=self.batch.replay_path,
            data_path=self.batch.data_path,
            playerID=self.batch.playerID,
        )

    def detach(self):
        self.losses_by_time = []
        self.hidden_states = self.hidden_states.detach()

    def step(self):
        in_progress_threshold = 0
        while in_progress_threshold < len(self.batch.states) and self.timestep < len(self.batch.states[in_progress_threshold]):
            in_progress_threshold += 1

        states = self.batch.states[:in_progress_threshold]
        target_positions = self.batch.target_positions[:in_progress_threshold]
        unit_type_counts = self.batch.unit_type_counts[:in_progress_threshold]
        minimap = self.batch.minimap_states[:in_progress_threshold]

        states = [x[self.timestep] for x in states]
        target_positions = [x[self.timestep].to(device=self.device, non_blocking=True) for x in target_positions]
        unit_type_counts = [x[self.timestep].to(device=self.device, non_blocking=True) for x in unit_type_counts]
        minimap = [x[self.timestep] for x in minimap]

        if len(states) > 0:
            batch_outputs = torch.stack(target_positions)
            batch_inputs = torch.stack(states)
            batch_inputs2 = torch.stack(unit_type_counts)
            batch_inputs3 = torch.stack(minimap)

            outputs, self.hidden_states = self.model(
                globalState=batch_inputs,
                unit_type_counts=batch_inputs2,
                hiddenState=self.hidden_states[:in_progress_threshold],
                minimap=batch_inputs3
            )

            self.outputs = outputs
            # NLL loss, NN outputs logsoftmax, and batch_outputs is one-hot, so this is correct
            losses = -(outputs * batch_outputs.view(outputs.size()[0], -1)).sum()
            loss = losses.sum()
            steps = outputs.size()[0]
            self.timestep += self.step_size
            return loss, steps
        else:
            return None
