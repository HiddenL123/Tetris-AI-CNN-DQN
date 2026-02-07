import os
from typing import List

import cv2
import torch
from env import DQNEnv, intEnv, AdvEnv
from agent import DQNAgent
from ddqn_agent import DDQNAgent
from model import QNetwork, LargeModel
from run_train import AgentConf


def run_eval(dir_name: str, model_name: str = "model.pth", episodes: int = 100, render: bool = False) -> List[int]:
    agent_conf = AgentConf()
    env = AdvEnv()
    agent = DDQNAgent(mem_size=agent_conf.mem_size, model = LargeModel(),
                     discount=agent_conf.discount, replay_start_size=agent_conf.replay_start_size, device="cpu")

    # timestamp_str = "20190730-165821"
    # log_dir = f'logs/tetris-nn={str(agent_conf.n_neurons)}-mem={agent_conf.mem_size}' \
    #     f'-bs={agent_conf.batch_size}-e={agent_conf.epochs}-{timestamp_str}'

    # tetris-20190731-221411-nn=[32, 32]-mem=25000-bs=512-e=1 good

    log_dir = 'logs/' + dir_name

    # load_model
    agent.model.load_state_dict(torch.load(f"{log_dir}/{model_name}"))
    agent.epsilon = 0
    scores = []
    for episode in range(episodes):
        env.reset()
        done = False

        while not done:
            next_states = env.get_next_states()
            best_state = agent.best_state([v[0] for v in next_states.values()])

            # find the action, that corresponds to the best state
            best_action = None
            best_seq = []
            for action, state_move in next_states.items():
                state, move_seq = state_move
                if state == best_state:
                    best_action = action
                    best_seq = move_seq
                    break
            _, _, done = env.play_moves(best_action, render=render, mode = "realistic", move_seq=best_seq)
            current_state = env._get_board_props(env.board)
            if best_state[1:] == current_state[1:]:
                #print(True)
                pass
            else:
                #print(best_state, current_state, move_seq, best_action)
                pass
        scores.append(env.score)
        # print results at the end of the episode
        print(f'episode {episode} => {env.score}')
    return scores

def enumerate_run_eval(episodes: int = 128, render: bool = False):
    dirs = [name for name in os.listdir('logs') if os.path.isdir(os.path.join('logs', name))]
    dirs.sort(reverse=True)
    dirs = [dirs[0]]  # take the most recent model
    # dirs = [
    #     'tetris-20190802-221032-ms25000-e1-ese2000-d0.99',
    #     'tetris-20190802-033219-ms20000-e1-ese2000-d0.95',
    # ]
    dirs = ['tetris-20260207-003526-ts5000000-bs32-d0.99']
    #dirs = ['tetris-20260123-014235-ms25000-e1-ese10000-d0.99']
    max_scores = []
    for d in dirs:
        print(f"Evaluating dir '{d}'")
        scores = run_eval(d, episodes=episodes, render=render, model_name="model_step2100000.pth")
        max_scores.append((d, max(scores)))

    max_scores.sort(key=lambda t: t[1], reverse=True)
    for k, v in max_scores:
        print(f"{v}\t{k}")


if __name__ == "__main__":
    enumerate_run_eval(episodes=16, render=True)
    cv2.destroyAllWindows()
    exit(0)
