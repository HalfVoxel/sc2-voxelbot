import torch
import numpy as np
import torch.nn as nn


class MovementStepper:
    def __init__(self, model, batch, device, time_threshold, loss_fn, step_size):
        self.batch = None
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.time_threshold = time_threshold
        self.step_size = step_size
        self.timestep = 0
        self.losses_by_time = []
        self.outputs = None
        self.minimap_cache = None

        if batch is not None:
            self.set_batch(batch)
            self.init_hidden_states(len(self.batch.states))

        # MovementTargetTrace = namedtuple('MovementTargetTrace', ['states', 'target_positions', 'unit_type_counts', 'replay_path', 'minimap_states', 'data_path', 'playerID'])
    
    def init_hidden_states(self, batch_size):
        self.hidden_states = self.model.init_hidden_states(batch_size, device=self.device)

    def set_batch(self, batch):
        if self.minimap_cache is None:
            self.minimap_cache = self.forward_minimap(batch.pathfinding_minimap)
        
        self.timestep = 0
        self.batch = type(batch)(
            states=[x.to(device=self.device, non_blocking=True) for x in batch.states],
            target_positions=[x.to(device=self.device, non_blocking=True) for x in batch.target_positions],
            unit_type_counts=[x.to(device=self.device, non_blocking=True) for x in batch.unit_type_counts],
            minimap_states=[x.to(device=self.device, non_blocking=True) for x in batch.minimap_states],
            pathfinding_minimap=self.minimap_cache,
            replay_path=batch.replay_path,
            data_path=batch.data_path,
            playerID=batch.playerID,
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
            target_positions = self.batch.target_positions[self.timestep] if len(self.batch.target_positions) > 0 else None
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

            self.timestep += self.step_size
            if target_positions is not None:
                # NLL loss, NN outputs logsoftmax, and batch_outputs is one-hot, so this is correct
                losses = -(outputs * target_positions.view(outputs.size()[0], -1)).sum()
                loss = losses.sum()
                steps = outputs.size()[0]
                return loss, steps
            else:
                return None
        else:
            return None
