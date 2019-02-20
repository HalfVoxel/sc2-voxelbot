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
        self.batch = type(self.batch)(
            states=[x.to(device=self.device, non_blocking=True) for x in self.batch.states],
            raw_unit_states=self.batch.raw_unit_states,
            raw_unit_coords=self.batch.raw_unit_coords,
            movement=self.batch.movement,
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
        raw_unit_states = self.batch.raw_unit_states[:in_progress_threshold]
        raw_unit_coords = self.batch.raw_unit_coords[:in_progress_threshold]
        movement = self.batch.movement[:in_progress_threshold]
        minimap = self.batch.minimap_states[:in_progress_threshold]

        states = [x[self.timestep] for x in states]
        raw_unit_states = [x[self.timestep].to(device=self.device, non_blocking=True) for x in raw_unit_states]
        raw_unit_coords = [x[self.timestep].to(device=self.device, non_blocking=True, dtype=torch.long) for x in raw_unit_coords]
        movement = [x[self.timestep].to(device=self.device, non_blocking=True) for x in movement]
        minimap = [x[self.timestep] for x in minimap]

        if len(states) > 0:
            batch_inputs = torch.stack(states)
            batch_outputs3 = torch.stack(minimap)
            batch_outputs = torch.nn.utils.rnn.pad_sequence(movement, batch_first=True)
            batch_inputs2 = torch.nn.utils.rnn.pad_sequence(raw_unit_states, batch_first=True)
            batch_inputs4 = torch.nn.utils.rnn.pad_sequence(raw_unit_coords, batch_first=True)
            max_unit_count = batch_outputs.size()[1]

            maskTensor = torch.zeros((len(raw_unit_states), max_unit_count), dtype=torch.float, device=self.device)
            for i in range(len(raw_unit_states)):
                maskTensor[i, :raw_unit_states[i].size()[0]] = 1

            # assert batch_inputs.size() == (len(states), *states[0].shape)
            # batch_outputs = torch.tensor(labels, dtype=torch.long, device=self.device)

            outputs, self.hidden_states = self.model(
                globalState=batch_inputs,
                rawUnits=batch_inputs2,
                rawUnitCoords=batch_inputs4,
                hiddenState=self.hidden_states[:in_progress_threshold],
                minimap=batch_outputs3
            )

            self.outputs = outputs
            losses = self.loss_fn(outputs.view(-1, 2), batch_outputs.view(-1))
            # Don't consider units that do not exist
            losses = losses * maskTensor.view(-1)
            loss = losses.sum()
            steps = maskTensor.sum()
            self.timestep += self.step_size
            return loss, steps
        else:
            return None
