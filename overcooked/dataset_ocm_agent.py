import torch, os
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from overcooked.env_to_vis import Kitchen
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

class ExpertSet(Dataset):
    def __init__(self, csv_file, max_episode, test_iteration):
        self.file_name = csv_file
        self.target_onion, self.target_dish, self.target_stove = [float('inf'), float('inf')], [float('inf'), float('inf')], [float('inf'), float('inf')]
        if os.path.exists(self.file_name):
            print("loading file...")
            self.load(self.file_name)
        else:
            steps = 0
            self.data = []        
            kitchen = Kitchen()
            ae = AgentEvaluator.from_layout_name({"layout_name": "cramped_room"}, {"horizon": 101})
            trajs = ae.evaluate_human_model_pair()

            for ep, mdp in enumerate(trajs["ep_states"][0]):
                if ep+1 < len(trajs["ep_states"][0]):
                    state = kitchen.create_state(mdp)
                    next_state = kitchen.create_state(trajs["ep_states"][0][ep+1])
                    steps += 1
                    for i in range(5):
                        self.data.append([state, next_state, self.convert_actions(trajs["ep_actions"][0][ep]), trajs["ep_rewards"][0][ep], False])
                if ep%(100) == 0: self.save(str(ep) + self.file_name)

            self.save(self.file_name)

    def convert_actions(self, actions):
        converted_actions = [4,4]
        for i, action in enumerate(actions):
            if action == 'interact': converted_actions[i] = 5
            elif action == (0, 1): converted_actions[i] = 0
            elif action == (0, -1): converted_actions[i] = 1
            elif action == (1, 0): converted_actions[i] = 2
            elif action == (-1, 0): converted_actions[i] = 3
            elif action == (0, 0): converted_actions[i] = 4
        return np.concatenate([F.one_hot(torch.tensor(converted_actions[0], dtype=torch.int64),6).numpy(), F.one_hot(torch.tensor(converted_actions[1], dtype=torch.int64),6).numpy()]).astype(np.int8)

    def save(self, file_name):
        torch.save(self.data, file_name)
        
    def load(self, file_name):
        self.data = torch.load(file_name)

    def __len__(self):
        return len(self.data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.data), size=batch_size)
        x = np.array([self.data[i][0] for i in ind])
        y = np.array([self.data[i][1] for i in ind])
        u = np.array([self.data[i][2] for i in ind], copy=False)
        r = np.array([self.data[i][3] for i in ind], copy=False)
        d = np.array([self.data[i][4] for i in ind], copy=False).reshape(-1, 1)
        return x, y, u, r, d
    