from typing import Optional
import os
import torch
from ddqn_agent import DDQNAgent
from env import AdvEnv, DQNEnv, intEnv
from datetime import datetime
from statistics import mean
from logs import CustomTensorBoard
from tqdm import tqdm

from model import QNetwork, LargeModel

class AgentConf:
    def __init__(self, resume_from_checkpoint: Optional[str] = None, resume_step: Optional[int] = None):
        self.batch_size = 32
        self.total_steps = 5000000  # Total training steps
        self.epsilon_val = [(0, 1.0), (500*1000, 0.2), (2500*1000, 0.05), (self.total_steps, 0.02)]  # Epsilon decay schedule
        self.transition = 600000

        self.mem_size = 250000
        self.discount = 0.99
        self.replay_start_size = 25000
        self.epochs = 1
        self.render_every = None
        self.train_every = 10
        self.log_every = 5000  # Log every N steps
        self.save_every = 100000  # Save every N steps
        self.update_target_every = 10000
        
        # Resume configuration
        self.resume_from_checkpoint = resume_from_checkpoint
        self.resume_step = resume_step
        
#for logging
stats = {
    'score': {'avg': 0.0, 'min': 0.0, 'max': 0.0},
    'damages': {'avg': 0.0, 'min': 0.0, 'max': 0.0},
    'q_value': {'predicted': 0.0, 'td error': 0.0},
    'epsilon': 0.0,
    'height_at_t': {"step_10" : 0.0, "step_20": 0.0}
}

def calc_reward(r1_line, r2_damage, done, env, step):
    """Calculate reward based on lines cleared, damage sent, and game over."""
    return r1_line + (-0.02 * max(env._height(env.board)[1] - 4, 0))* step/conf.total_steps
    

# Run dqn with Tetris
# noinspection PyShadowingNames
def dqn(conf: AgentConf):
    env = AdvEnv()

    # Determine if resuming and set loaded parameters
    loaded_step = None
    loaded_epsilon_val = None

    log_dir = None
    resume_step = None
    
    if conf.resume_from_checkpoint:
        checkpoint_dir = conf.resume_from_checkpoint
        resume_step = conf.resume_step
        
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Use existing log directory for resuming
        log_dir = checkpoint_dir
        
        if resume_step is not None:
            model_file = f'{checkpoint_dir}/model_step{resume_step}.pth'
            memory_file = f'{checkpoint_dir}/replay_step{resume_step}.pkl'
            
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Model checkpoint not found: {model_file}")
            if not os.path.exists(memory_file):
                raise FileNotFoundError(f"Memory checkpoint not found: {memory_file}")
            
            # Calculate loaded parameters based on step
            loaded_step = resume_step
            
            print(f"Resuming from checkpoint at step {resume_step}")
            print(f"  - Model: {model_file}")
            print(f"  - Memory: {memory_file}")
            print(f"  - Loaded epsilon: {loaded_epsilon_val:.6f}")
        else:
            raise ValueError("resume_step must be specified when resuming from checkpoint")
    else:
        # New training
        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f'logs/tetris-{timestamp_str}-ts{conf.total_steps}-bs{conf.batch_size}-d{conf.discount}'

    print(f"AGENT_CONF = {log_dir}")

    agent = DDQNAgent(device = "cuda", #device
                     mem_size=conf.mem_size, replay_start_size=conf.replay_start_size, #replay
                     discount=conf.discount,
                     epsilon_val =conf.epsilon_val,
                     model = LargeModel(board_dim=(10, 20), scalar_feature_dim=14), update_target_every=conf.update_target_every,
                     loaded_step=loaded_step, loaded_epsilon_val = loaded_epsilon_val
                     )

    # Load checkpoint if resuming
    if conf.resume_from_checkpoint and resume_step is not None:
        model_file = f'{conf.resume_from_checkpoint}/model_step{resume_step}.pth'
        memory_file = f'{conf.resume_from_checkpoint}/replay_step{resume_step}.pkl'
        agent.load_model(model_file)
        agent.load_memory(memory_file)
        print(f"Loaded model and memory from step {resume_step}")

    log = CustomTensorBoard(log_dir=log_dir)

    episode_scores = []
    episode_damages = []
    episode_height_at_10 = []
    episode_height_at_20 = []

    q_values = []
    td_errors = []
    best_score = -float('inf')
    best_step = 0

    current_state = env.reset()
    episode_step = 0
    
    # Set up training loop range
    if conf.resume_from_checkpoint and resume_step is not None:
        start_step = resume_step + 1
        remaining_steps = conf.total_steps - resume_step
        steps_wrapped = tqdm(range(remaining_steps), initial=resume_step, total=conf.total_steps)
    else:
        start_step = 0
        steps_wrapped = tqdm(range(conf.total_steps))

    for i, _ in enumerate(steps_wrapped):
        global_step = start_step + i if conf.resume_from_checkpoint and resume_step is not None else i
        
        # update render flag
        render = conf.render_every and global_step % conf.render_every == 0
        mode = "realistic" if render else "fast"

        # Game step
        next_states = env.get_next_states()
        best_state = agent.best_state((v[0] for v in next_states.values()), exploration=True)

        # find the action, that corresponds to the best state
        best_action = None
        final_seq = [" "]
        for action, state_move in next_states.items():
            state, move_seq = state_move
            if state == best_state:
                best_action = action
                final_seq = move_seq
                break

        r1_line, r2_damage, done = env.play_moves(best_action, render=render, mode="fast", move_seq=final_seq)
        reward = calc_reward(r1_line, r2_damage, done, env, global_step)
        agent.add_to_memory(current_state, next_states[best_action][0], reward, done)
        current_state = next_states[best_action][0]
        episode_step += 1

        if env.game_step == 10:
            env.height_at_10 = env._height(env.board)[1]
        elif env.game_step == 20:
            env.height_at_20 = env._height(env.board)[1]

        # Train
        if agent.step % conf.train_every == 0:
            q, td_error = agent.train(batch_size=conf.batch_size)
            q_values.append(q)
            td_errors.append(td_error)

        # Save checkpoint
        if conf.save_every and global_step > 0 and global_step % conf.save_every == 0:
            torch.save(agent.model.state_dict(), f'{log_dir}/model_step{global_step}.pth')
            agent.save_memory(f'{log_dir}/replay_step{global_step}.pkl')

        agent.increment_step()
        
        # Episode end
        if done:
            episode_score, episode_damage = env.get_round_score()
            episode_scores.append(episode_score)
            episode_damages.append(episode_damage)
            episode_height_at_10.append(env.height_at_10)
            episode_height_at_20.append(env.height_at_20)
            
            if episode_score > best_score:
                best_score = episode_score
                best_step = global_step
                torch.save(agent.model.state_dict(), f'{log_dir}/model_best.pth')

            current_state = env.reset()
            episode_step = 0

        # Logging
        if conf.log_every and global_step > 0 and global_step % conf.log_every == 0:
            if episode_scores:
                stats['score']['avg'] = mean(episode_scores[-100:])  # Last 100 episodes
                stats['score']['min'] = min(episode_scores[-100:])
                stats['score']['max'] = max(episode_scores[-100:])
            if episode_damages:
                stats['damages']['avg'] = mean(episode_damages[-100:])  # Last 100 episodes
                stats['damages']['min'] = min(episode_damages[-100:])
                stats['damages']['max'] = max(episode_damages[-100:])
            if q_values:
                stats['q_value']['predicted'] = mean(q_values[-100:])
            if td_errors:
                stats['q_value']['td error'] = mean(td_errors[-100:])
            if episode_height_at_10:
                stats['height_at_t']['step_10'] = mean(episode_height_at_10)
                episode_height_at_10 = []  # reset after logging
            if episode_height_at_20:
                stats['height_at_t']['step_20'] = mean(episode_height_at_20)
                episode_height_at_20 = []  # reset after logging
            stats['epsilon'] = agent.epsilon

            log.log(global_step, stats)
            steps_wrapped.set_description(f"Score: {stats['score']['avg']:.1f} | Epsilon: {agent.epsilon:.4f} | Step: {global_step}")

    # save_model
    torch.save(agent.model.state_dict(), f'{log_dir}/model_final.pth')
    agent.save_memory(f'{log_dir}/replay_final.pkl')
    print(f"Training complete! Best score: {best_score} at step {best_step}")


if __name__ == "__main__":
    # Configure to resume from checkpoint at step 3.3M
    conf = AgentConf(

    )
    dqn(conf)
    # to avoid jump to console when run under IDE
    exit(0)
