from model import QNetwork
from typing import ValuesView, List, Optional
import torch
import pickle
from memory_buffer import PrioritizedReplayBuffer

from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Available device: {device}")

class DQNAgent:
    """Deep Q Learning Agent + Maximin

    Args:
        state_size (int): Size of the input domain
        mem_size (int): Size of the replay buffer
        discount (float): How important is the future rewards compared to the immediate ones [0,1]
        epsilon (float): Exploration (probability of random values given) value at the start
        epsilon_min (float): At what epsilon value the agent stops decrementing it
        epsilon_stop_step (int): At what step the agent stops decreasing the exploration variable
        model: Neural network model
        replay_start_size: Minimum size needed to train
    """

    def __init__(self, mem_size=100000, discount=0.95,
                 epsilon=1, epsilon_min=0, epsilon_stop_step=500*100,
                 start_temp = 1, temp_decay_rate = 0.999, end_temp = 0.05, transition_len = 500,
                 model = None, replay_start_size=None, device = device,
                memory_n_step = 7):
                 

        self.device = device
        self.memory = PrioritizedReplayBuffer(capacity=mem_size)
        self.mem_size = mem_size
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon
        self.epsilon_decay = (self.epsilon_max - self.epsilon_min) / epsilon_stop_step
        self.start_temp = start_temp
        self.temp_decay_rate = temp_decay_rate
        self.end_temp = end_temp
        self.temp = start_temp
        self.transition_len = transition_len
        self.temp_start_step = epsilon_stop_step
        self.temp = start_temp
        if not replay_start_size:
            replay_start_size = mem_size / 2
        self.replay_start_size = replay_start_size
        self.model = QNetwork().to(self.device) if model is None else model.to(device)
        self.step = 0
        self.memory_n_step = memory_n_step
        self.memory_n_step_buffer = deque(maxlen=self.memory_n_step)

    def _calc_prior(self, state_0, next_state_n, R, done_n, n):
        return self.memory.max_priority
        
    def add_to_memory(self, state, next_state, reward, done):
        # Store transition temporarily
        self.memory_n_step_buffer.append((state, next_state, reward, done))

        # Not enough steps yet
        if len(self.memory_n_step_buffer) < self.memory_n_step:
            return

        # Compute n-step return
        R = 0.0
        n = 0
        for i, (_, _, r, d) in enumerate(self.memory_n_step_buffer):
            R += (self.discount ** i) * r
            n += 1
            if d:
                break

        state_0, _, _, _ = self.memory_n_step_buffer[0]
        _, next_state_n, _, done_n = self.memory_n_step_buffer[-1]

        prior = self._calc_prior(state_0, next_state_n, R, done_n, n)
        self.memory.add(
            (state_0, next_state_n, R, done_n, n),
            priority=prior
        )

        # If step ended, flush remaining transitions
        if done:
            self._flush_n_step_buffer()

    def _flush_n_step_buffer(self):
        while len(self.memory_n_step_buffer) > 0:
            R = 0.0
            n = 0

            for i, (_, _, r, d) in enumerate(self.memory_n_step_buffer):
                R += (self.discount ** i) * r
                n += 1
                if d:
                    break

            state_0, _, _, _ = self.memory_n_step_buffer[0]
            _, next_state_n, _, done_n = self.memory_n_step_buffer[-1]

            prior = self._calc_prior(state_0, next_state_n, R, done_n, n)
            self.memory.add(
                (state_0, next_state_n, R, done_n, n),
                priority=prior
            )

            # Remove the first transition and recompute
            self.memory_n_step_buffer.popleft()
    
    def save_memory(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.memory.state_dict(), f)

    def load_memory(self, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.memory.load_state_dict(state)


    def load_model(self, onlinefilename: str):
        """Loads the model from a file"""
        self.model.load_state_dict(torch.load(onlinefilename))

    def predict_value(self, state) -> float:
        """Predicts the score for a certain state.
        
        Delegates to model.predict() which handles state unpacking.
        """
        return self.model.predict(state, device=self.device)

    def best_state(self, states: ValuesView[List[int]], exploration=False) -> List[int]:
        """Returns the best state for a given collection of states"""
        states_list = list(states)
        
        # Non-exploration mode: pure greedy
        if not exploration:
            return self._greedy_selection(states_list)
        
        # Phase 1: Epsilon-greedy
        if self.step < self.temp_start_step:
            return self._epsilon_greedy_selection(states_list)
        
        # Phase 2: Transition (blend epsilon-greedy and Boltzmann)
        elif self.step < self.temp_start_step + self.transition_len:
            return self._transition_selection(states_list)
        
        # Phase 3: Pure Boltzmann
        else:
            return self._boltzmann_selection(states_list)
    
    def _greedy_selection(self, states):
        """Pure greedy selection - no exploration (batched)"""
        if not states:
            return None
        
        # Batch predict all states at once
        values = self.model.batch_predict(states, device=self.device)
        
        # Find best state
        best_idx = max(range(len(values)), key=lambda i: values[i])
        return states[best_idx] 
    
    def _epsilon_greedy_selection(self, states):
        if random.random() <= self.epsilon:
            return random.choice(list(states))
        else:
            return self._greedy_selection(states)
    
    def _boltzmann_selection(self, states):
        """Boltzmann (softmax) exploration (batched)"""
        if not states:
            return None
        
        # Batch predict all states at once
        values = self.model.batch_predict(states, device=self.device)
        
        # Apply softmax with temperature
        values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
        probabilities = torch.softmax(values_tensor / self.temp, dim=0)
        
        # Sample based on probabilities
        chosen_idx = torch.multinomial(probabilities, 1).item()
        return states[chosen_idx]

    def _transition_selection(self, states_list):
        """Blend between epsilon-greedy and Boltzmann"""
        # Calculate transition progress (0.0 to 1.0)
        progress = (self.step - self.temp_start_step) / self.transition_len
        
        # Decide which strategy to use based on progress
        # Early in transition: mostly epsilon-greedy
        # Late in transition: mostly Boltzmann
        if random.random() < (1 - progress):
            # Use epsilon-greedy (but with reduced epsilon)
            reduced_epsilon = self.epsilon * (1 - progress)
            if random.random() <= reduced_epsilon:
                return random.choice(states_list)
            else:
                return self._greedy_selection(states_list)
        else:
            # Use Boltzmann
            return self._boltzmann_selection(states_list)
    
    
    def train(self, batch_size=32, epochs=1):
        """Trains the agent"""
        N = len(self.memory)
        if N < self.replay_start_size or N < batch_size:
            return 0.0, 0.0
        batch, indices, weights = self.memory.sample(batch_size)
        weights = weights.to(self.device)
        
        # Determine state format from first batch item
        sample_state = batch[0][1]  # next_state from first sample
        is_tuple_state = isinstance(sample_state, (tuple, list)) and len(sample_state) == 2
        
        if is_tuple_state:
            # -------- unpack batch --------
            boards = torch.tensor(
                [x[0][0] for x in batch],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(1)

            scalars = torch.tensor(
                [x[0][1] for x in batch],
                dtype=torch.float32,
                device=self.device
            )

            next_boards = torch.tensor(
                [x[1][0] for x in batch],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(1)

            next_scalars = torch.tensor(
                [x[1][1] for x in batch],
                dtype=torch.float32,
                device=self.device
            )

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

            # -------- compute targets --------
            with torch.no_grad():
                next_values = self.model.forward(next_boards, next_scalars)
                targets = rewards + (1.0 - dones) * (self.discount ** n_steps) * next_values

            # -------- forward + loss --------
            self.model.train()
            predictions = self.model.forward(boards, scalars)
        else:
            # -------- flat state version --------
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

            with torch.no_grad():
                next_values = self.model.forward(next_states)
                targets = rewards + (1.0 - dones) * (self.discount ** n_steps) * next_values

            self.model.train()
            predictions = self.model.forward(states)
        # -------- TD error (per-sample) --------
        td_errors = (predictions - targets).detach().abs().squeeze()

        # -------- PER-weighted loss --------
        loss = loss = torch.nn.functional.smooth_l1_loss(predictions, targets, reduction='none')
        loss = (weights * loss).mean()

        # -------- optimize --------
        self.model.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.model.optimizer.step()

        self.memory.update_priorities(indices, td_errors.cpu().numpy())
        return predictions.mean().item(), targets.mean().item()
    
    def increment_step(self):
        """Increments the internal step counter"""
        self.step += 1
        # Epsilon decay
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        # Temperature decay
        if self.step >= self.temp_start_step:
            self.temp *= self.temp_decay_rate
            self.temp = max(self.end_temp, self.temp)
