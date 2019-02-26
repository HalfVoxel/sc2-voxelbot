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
        self.hidden_states = self.model.init_hidden_states(batch.states[0].size()[0], device=self.device)
        self.outputs = None

        # MovementTargetTrace = namedtuple('MovementTargetTrace', ['states', 'target_positions', 'unit_type_counts', 'replay_path', 'minimap_states', 'data_path', 'playerID'])
        self.batch = type(self.batch)(
            states=[x.to(device=self.device, non_blocking=True) for x in self.batch.states],
            target_positions=[x.to(device=self.device, non_blocking=True) for x in self.batch.target_positions],
            unit_type_counts=[x.to(device=self.device, non_blocking=True) for x in self.batch.unit_type_counts],
            minimap_states=[x.to(device=self.device, non_blocking=True) for x in self.batch.minimap_states],
            pathfinding_minimap=self.forward_minimap(batch.pathfinding_minimap),
            replay_path=self.batch.replay_path,
            data_path=self.batch.data_path,
            playerID=self.batch.playerID,
        )

    def forward_minimap(self, pathfinding_minimap):
        tensors = pathfinding_minimap.to(device=self.device, non_blocking=False)
        return self.model.forward_minimap(tensors)

    def detach(self):
        self.losses_by_time = []
        self.hidden_states = self.hidden_states.detach()

    def step(self):
        if self.timestep < len(self.batch.states):
            states = self.batch.states[self.timestep]
            target_positions = self.batch.target_positions[self.timestep]
            unit_type_counts = self.batch.unit_type_counts[self.timestep]
            minimap = self.batch.minimap_states[self.timestep]

            in_progress_threshold = states.size()[0]
            assert in_progress_threshold > 0

            outputs, self.hidden_states = self.model(
                globalState=states,
                unit_type_counts=unit_type_counts,
                hiddenState=self.hidden_states[:in_progress_threshold],
                minimap=minimap,
                pathfinding_minimap=self.batch.pathfinding_minimap[:in_progress_threshold]
            )

            self.outputs = outputs
            # NLL loss, NN outputs logsoftmax, and batch_outputs is one-hot, so this is correct
            losses = -(outputs * target_positions.view(outputs.size()[0], -1)).sum()
            loss = losses.sum()
            steps = outputs.size()[0]
            self.timestep += self.step_size
            return loss, steps
        else:
            return None
