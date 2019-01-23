import torch
import numpy as np
import torch.nn.functional as F
from replay_memory import Transition
from training_algorithm import TrainingAlgorithm

class A2C(TrainingAlgorithm):
    def __init__(self, policy_net, target_net, optimizer, device, gamma_per_second, input_shape):
        super().__init__()
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.device = device
        self.gamma_per_second = gamma_per_second
        self.input_shape = input_shape

    def train(self, memory, batch_size):
        self.policy_net.train()
        transitions = memory.sample(batch_size)
        calculated_values, expected_values, loss, transition_losses = self.evaluate_batch(transitions)
        memory.insert(transitions, transition_losses)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item(), expected_values

    def test(self, memory):
        self.policy_net.eval()
        transitions = memory.get_all()
        _, _, loss, _ = self.evaluate_batch(transitions)
        return loss.item()

    def evaluate_batch(self, transitions):
        batch_size = len(transitions)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        



        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.uint8)

        # No non-final states, will cause some torch errors
        any_non_final = len([s for s in batch.next_state if s is not None]) > 0
        if any_non_final > 0:
            non_final_next_states = torch.cat([s.unsqueeze(0) for s in batch.next_state if s is not None])

        state_batch = torch.cat([s.unsqueeze(0) for s in batch.state])
        assert state_batch.size() == (batch_size, self.input_shape), (state_batch.size(), (batch_size, self.input_shape))

        action_batch = torch.tensor(batch.action)
        assert(action_batch.size() == (batch_size,))
        reward_batch = torch.tensor(batch.reward, dtype=torch.float)
        assert reward_batch.size() == (batch_size,)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        assert state_action_values.size() == (batch_size, 1)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(batch_size, device=self.device)
        if any_non_final:
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        assert next_state_values.size() == (batch_size,)

        # torch.set_printoptions(threshold=10000)

        # Compute the expected Q values
        gammas = np.power(self.gamma_per_second, batch.deltaTime)
        expected_state_action_values = (next_state_values * torch.tensor(gammas, dtype=torch.float)) + reward_batch
        assert expected_state_action_values.size() == (batch_size,)

        # Compute Huber loss
        transition_losses = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='none')
        loss = transition_losses.mean()
        return state_action_values, expected_state_action_values.unsqueeze(1), loss, transition_losses
