import copy
from typing import Optional
import torch
from agent import DQNAgent
from agent import device
import random

class DDQNAgent(DQNAgent):
    """Deep Q Learning Agent + Maximin

    Args:
        state_size (int): Size of the input domain
        mem_size (int): Size of the replay buffer
        discount (float): How important is the future rewards compared to the immediate ones [0,1]
        epsilon_val (list of tuples): List of (step, epsilon) pairs defining the epsilon decay schedule
        model: Neural network model
        replay_start_size: Minimum size needed to train
    """

    def __init__(self, mem_size=100000, discount=0.95,
                 epsilon_val = [(0, 1), (500*1000, 0.2), (2500*1000, 0.05), (5000*1000, 0.02)],
                 model = None, replay_start_size=None, device = device,
                memory_n_step = 7, update_target_every = 10000,
                loaded_step=None, loaded_epsilon_val=None):
                 

        super().__init__(mem_size, discount, epsilon_val, model, replay_start_size, device, memory_n_step)
        self.target_model = copy.deepcopy(self.model)
        self.target_model.load_state_dict(self.model.state_dict())
        self.update_target_every = update_target_every
        
        # Restore loaded parameters if provided
        if loaded_step is not None:
            self.step = loaded_step

            if loaded_epsilon_val is not None:
                self.epsilon_val = loaded_epsilon_val
                epsilon_found = False
                for phase in range(len(self.epsilon_val)-1):
                    t, val = self.epsilon_val[phase]
                    self.epsilon_phase = phase
                    epsilon_diff = (self.epsilon_val[self.epsilon_phase][1] - self.epsilon_val[self.epsilon_phase+1][1])
                    t_diff = (self.epsilon_val[self.epsilon_phase+1][0] - self.epsilon_val[self.epsilon_phase][0])
                    self.epsilon_decay = epsilon_diff / t_diff
                    self.epsilon = val - (loaded_step - t) * self.epsilon_decay
                    epsilon_found = True
                    break
                if not epsilon_found:
                    # If loaded_step exceeds all defined phases, set to last phase's epsilon
                    self.epsilon_phase = len(self.epsilon_val) - 2
                    self.epsilon = self.epsilon_val[-1][1]
                    self.epsilon_decay
            

    def _calc_prior(self, state_0, next_state_n, R, done_n, n):
        with torch.no_grad():
            # Get current Q prediction for state_0
            # Note: Use self.model for prediction, not target_model here
            current_q = self.model.predict(state_0, device=self.device)
    
            # Get target Q from next_state_n
            next_q = self.target_model.predict(next_state_n, device=self.device)
            target_q = R + (self.discount ** n) * next_q * (1.0 - done_n)
            
            # The priority is the absolute TD Error
            new_priority = abs(current_q - target_q) + 1e-6 # small constant to avoid 0 priority
            return new_priority

    def load_model(self, onlinefilename: str, offlinefilename: Optional[str] = None):
        """Loads the model from a file"""
        self.model.load_state_dict(torch.load(onlinefilename))
        if offlinefilename:
            self.target_model.load_state_dict(torch.load(offlinefilename))
        else:
            self.target_model.load_state_dict(self.model.state_dict())
    
    
    def train(self, batch_size=32):
        if len(self.memory) < max(self.replay_start_size, batch_size):
            return 0.0, 0.0

        batch, indices, weights = self.memory.sample(batch_size)
        weights = weights.to(self.device)

        sample_state = batch[0][0]
        is_tuple_state = isinstance(sample_state, (tuple, list)) and len(sample_state) == 2

        rewards = torch.tensor(
            [x[2] for x in batch],
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)

        dones = torch.tensor(
            [x[3] for x in batch],
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)

        n_steps = torch.tensor(
            [x[4] for x in batch],
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)

        if is_tuple_state:
            state_boards = torch.tensor(
                [x[0][0] for x in batch],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(1)

            state_scalars = torch.tensor(
                [x[0][1] for x in batch],
                dtype=torch.float32,
                device=self.device
            )

            next_state_boards = torch.tensor(
                [x[1][0] for x in batch],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(1)

            next_state_scalars = torch.tensor(
                [x[1][1] for x in batch],
                dtype=torch.float32,
                device=self.device
            )

            with torch.no_grad():
                next_values = self.target_model.forward(
                    next_state_boards,
                    next_state_scalars
                )
                targets = rewards + (1.0 - dones) * (self.discount ** n_steps) * next_values

            predictions = self.model.forward(state_boards, state_scalars)

        else:
            states = torch.tensor(
                [x[0] for x in batch],
                dtype=torch.float32,
                device=self.device
            )

            next_states = torch.tensor(
                [x[1] for x in batch],
                dtype=torch.float32,
                device=self.device
            )

            with torch.no_grad():
                next_values = self.target_model.forward(next_states)
                targets = rewards + (1.0 - dones) * (self.discount ** n_steps) * next_values

            predictions = self.model.forward(states)

        # ----- TD error (per-sample) -----
        td_errors = (predictions - targets).detach().abs().squeeze()

        # ----- PER-weighted loss -----
        loss = torch.nn.functional.smooth_l1_loss(predictions, targets, reduction='none')
        loss = (weights * loss).mean()

        self.model.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.model.optimizer.step()

        # ----- priority update -----
        self.memory.update_priorities(indices, td_errors.cpu().numpy())

        return predictions.mean().item(), td_errors.mean().item()

    
    def increment_step(self):
        # Target network update
        super().increment_step()
        if self.step % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())