import torch
import numpy as np
import torch.nn as nn


class MovementStepper:
    def __init__(self, model, batch, device, time_threshold, loss_fn, step_size):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.time_threshold = time_threshold
        self.step_size = step_size
        self.timestep = 0
        self.losses_by_time = []
        self.outputs = None
        self.batch = None
        if batch is not None:
            self.set_batch(batch, True)
            self.init_hidden_states(len(self.batch.states))

    def init_hidden_states(self, batch_size):
        self.hidden_states = self.model.init_hidden_states(batch_size, device=self.device)

    def set_batch(self, batch, non_blocking=False):
        self.timestep = 0
        self.batch = type(batch)(
            states=[x.to(device=self.device, non_blocking=non_blocking) for x in batch.states],
            raw_unit_states=[x.to(device=self.device, non_blocking=non_blocking) for x in batch.raw_unit_states],
            raw_unit_coords=[x.to(device=self.device, non_blocking=non_blocking) for x in batch.raw_unit_coords],
            movement=[x.to(device=self.device, non_blocking=non_blocking) for x in batch.movement],
            order_changed=[x.to(device=self.device, non_blocking=non_blocking) for x in batch.order_changed],
            minimap_states=[x.to(device=self.device, non_blocking=non_blocking) for x in batch.minimap_states],
            replay_path=batch.replay_path,
            data_path=batch.data_path,
            playerID=batch.playerID,
        )

    def detach(self):
        self.losses_by_time = []
        self.hidden_states = self.hidden_states.detach()

    def step(self):
        if self.timestep < len(self.batch.states):
            states = self.batch.states[self.timestep]
            raw_unit_states = self.batch.raw_unit_states[self.timestep]
            raw_unit_coords = self.batch.raw_unit_coords[self.timestep]
            movement = self.batch.order_changed[self.timestep] if len(self.batch.order_changed) > 0 else None
            minimap_states = self.batch.minimap_states[self.timestep]

            in_progress_threshold = states.size()[0]
            assert in_progress_threshold > 0

            # batch_inputs = torch.stack(states)
            # batch_outputs3 = torch.stack(minimap_states)
            # batch_outputs = torch.nn.utils.rnn.pad_sequence(movement, batch_first=True)
            # batch_inputs2 = torch.nn.utils.rnn.pad_sequence(raw_unit_states, batch_first=True)
            # batch_inputs4 = torch.nn.utils.rnn.pad_sequence(raw_unit_coords, batch_first=True)

            # assert batch_inputs.size() == (len(states), *states[0].shape)
            # batch_outputs = torch.tensor(labels, dtype=torch.long, device=self.device)

            outputs, self.hidden_states = self.model(
                globalState=states,
                rawUnits=raw_unit_states,
                rawUnitCoords=raw_unit_coords,
                hiddenState=self.hidden_states[:in_progress_threshold],
                minimap=minimap_states
            )

            self.outputs = outputs

            if movement is not None:
                max_unit_count = movement.size()[1]

                maskTensor = torch.zeros((len(raw_unit_states), max_unit_count), dtype=torch.float, device=self.device)
                for i in range(len(raw_unit_states)):
                    maskTensor[i, :raw_unit_states[i].size()[0]] = 1

                losses = self.loss_fn(outputs.view(-1, 2), movement.view(-1))
                # Don't consider units that do not exist
                losses = losses * maskTensor.view(-1)
                loss = losses.sum()
                steps = maskTensor.sum()
                self.timestep += self.step_size
                return loss, steps
            else:
                return None
        else:
            return None
