import torch, random, imageio, os, cv2
import numpy as np
import torch.optim as optim
from models import Critic_Intersection
from intersection.dataset import ExpertSet
from highway_env.envs.intersection_env import MultiAgentIntersectionEnv
from train_update import update_setups
from collections import deque
import statistics
import torch.nn as nn

dir = "ma_models/"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Replay_buffer():
    def __init__(self, capacity):
        self.storage = []
        self.max_size = capacity
        self.ptr = 0

    def push(self, data):
        if len(self.storage) >= self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []
        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X.cpu(), copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
        return np.array(x), np.array(y), np.array(u), np.array(r), np.array(d).reshape(-1, 1)
    
    def save(self, file_name):
        torch.save(self.storage, file_name)
        
    def load(self, file_name):
        self.storage = torch.load(file_name)

class OfflineSACIQLearn:
    def __init__(self, expert_data, env_config, setup):
        self.config = env_config
        self.replay_buffer = Replay_buffer(self.config["capacity"])
        self.replay_buffer.load("expert_data/buffer_init/rb_intersection.csv")
        obs, _, acts, _, _ = expert_data.sample(1)
        if setup < 4: self.critic = Critic_Intersection(len(obs[0]), len(np.eye(3)[acts[0]].flatten())).to(device)
        elif setup == 4: self.critic = Critic_Intersection(len(obs[0])).to(device)
        elif setup==5: self.critic = Critic_Intersection(len(obs[0]), len(np.eye(3)[acts[0]].flatten())//self.config['num_agents']).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config['critic_lr']) #, weight_decay=0.0001
        self.expert_data = expert_data
        self.max_reward = 0
        self.criterion = torch.nn.MSELoss()
        self.setup = setup
        self.running_pred_rew = deque(maxlen=100)
        self.mse_loss = nn.MSELoss()

    def learn(self, agent, episode, critics):
        [critic.eval() for critic in critics]
        self.critic.train()
        obs, next_obs, acts, r, d = self.expert_data.sample(self.config['batch_size'])
        obs = torch.tensor(obs, dtype=torch.float32, requires_grad=True).to(device)
        actions = torch.tensor(acts, dtype=torch.int64).to(device)
        r = torch.tensor(r, dtype=torch.float32, requires_grad=True).to(device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, requires_grad=True).to(device) 
        d = torch.reshape(torch.tensor(d, dtype=torch.float32, requires_grad=True).to(device), (-1,))

        actions_one_hot = torch.zeros(actions.size(0), actions.size(1), 3).to(device)
        actions_one_hot.scatter_(2, actions.unsqueeze(-1), 1).to(device)
        actions = actions_one_hot.view(actions.size(0), -1).to(device)
        acts = actions[:, (agent*3):(agent*3+3)].clone().detach()

        if self.setup == 1:
            pre_reg_loss, mse_predicted_reward = update_setups().iql_ma(self.critic, obs, next_obs, actions, acts, self.config['gamma'], d, self.config['alpha'], r, agent)
        elif self.setup == 2:
            pre_reg_loss, mse_predicted_reward = update_setups().mamql_offline(self.mse_loss, self.critic, obs, next_obs, actions, acts, self.config['gamma'], d, self.config['alpha'], r, agent)
        elif self.setup == 4:
            pre_reg_loss, mse_predicted_reward = update_setups().mamql_online_several_agents(critics, obs, next_obs, actions, self.config['gamma'], 
                agent, d, self.config['batch_size'], self.config['num_agents'], 
                device, self.replay_buffer, self.config['action_space'])
        elif self.setup == 5:
            pre_reg_loss, mse_predicted_reward = update_setups().behavioral_clone(self.critic, obs, acts, device)
        
        self.running_pred_rew.append(mse_predicted_reward)
        
        l2_reg = torch.tensor(0.).to(device)
        for param in self.critic.parameters():
            l2_reg += torch.norm(param)**2
        loss = pre_reg_loss + 0.001 * l2_reg
        
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        self.critic.eval()
        if (episode+1) % 100 == 0: eval = 10 
        else: eval = 1
        env = MultiAgentIntersectionEnv(render_mode='rgb_array')
        env.configure({
            "id": "intersection-multi-agent-v0",
            "import_module": "highway_env",
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "vehicles_count": 3,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [20, 40],
                        "vy": [20, 40]
                    },
                    "absolute": True,
                    "order": "shuffled"
                }
            },
            "initial_vehicle_count": 0,
            "controlled_vehicles": 4,
            "collision_reward": -4,
            "high_speed_reward": 0,
            "arrived_reward": 2
        })
        avg_reward = 0
        for _ in range(eval):
            total_reward = 0
            (obs, _), done = env.reset(), False
            while not done:
                obs = np.concatenate(obs, axis=0).reshape(1, -1)[0]
                obs = torch.tensor(obs, dtype=torch.float32).to(device)
                action_space = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0]).to(device)
                if self.setup == 5: action_space = torch.tensor([0, 0, 0]).to(device)
                actions = []
                for i in range(4):
                    if self.setup != 4:
                        actions.append(torch.argmax(critics[i](obs, action_space)).cpu())
                    else:
                        actions.append(torch.argmax(critics[i](obs)).cpu())
                next_obs, reward, done, _, _ = env.step(tuple(actions))
                if (episode+1) % 100 != 0:
                    self.replay_buffer.push((obs, torch.tensor(np.concatenate(next_obs, axis=0).reshape(1, -1)[0], dtype=torch.float32).cpu(), actions, reward, done))
                obs = next_obs
                total_reward += reward
            avg_reward += total_reward

        if (episode+1) % 100 == 0: 
            avg_reward /= 10
            print(str(agent)+" Episode: " + str(episode+1) +" Total Reward: " + str(total_reward)+" Loss: " + str(loss.item()))
            print("Imitation_Loss: "+ str(statistics.mean(self.running_pred_rew)))
            if True and total_reward >= self.max_reward:
                self.max_reward = total_reward
                for i in range(4):
                    torch.save(critics[i].state_dict(), dir +'intersection_'+ str(i)+'_episode_'+'_'+str(episode+1)+'_' + str(self.max_reward) + '_critic.pth')

class intersection_ma_irl:
    def __init__(self, configs, setup):
        self.config = configs
        self.setup = setup
    
    def train(self):
        expert_data = ExpertSet("expert_data/intersection_dataset.csv", self.config['dataset_episodes'], self.config['test_iteration'])
        print("loaded data")
        agent1 = OfflineSACIQLearn(expert_data, self.config, self.setup)
        agent2 = OfflineSACIQLearn(expert_data, self.config, self.setup)
        agent3 = OfflineSACIQLearn(expert_data, self.config, self.setup)
        agent4 = OfflineSACIQLearn(expert_data, self.config, self.setup)
        agents = [
            (agent1, 0, (agent1.critic, agent2.critic, agent3.critic, agent4.critic)),
            (agent2, 1, (agent1.critic, agent2.critic, agent3.critic, agent4.critic)),
            (agent3, 2, (agent1.critic, agent2.critic, agent3.critic, agent4.critic)),
            (agent4, 3, (agent1.critic, agent2.critic, agent3.critic, agent4.critic))
        ]
        for episode in range(self.config['episodes']):
            random.shuffle(agents)
            for agent, idx, critics in agents:
                agent.learn(idx, episode, critics)

    def test(self):
        model_directory = "ma_models/"
        assert os.path.exists(model_directory)
        mp4_dir = os.path.join(model_directory, 'mp4')
        if not os.path.exists(mp4_dir):
            os.makedirs(mp4_dir)
        
        expert_data = ExpertSet("expert_data/intersection_dataset.csv", self.config['dataset_episodes'], self.config['test_iteration'])    
        obs, _, acts, _, _ = expert_data.sample(1)

        if self.setup < 4: 
            critics = [Critic_Intersection(len(obs[0]), len(np.eye(3)[acts[0]].flatten())).to(device) for _ in range(4)]
        elif self.setup == 4: 
            critics = [Critic_Intersection(len(obs[0])).to(device) for _ in range(4)]
        elif self.setup == 5: 
            critics = [Critic_Intersection(len(obs[0]), len(np.eye(3)[acts[0]].flatten())//self.config['num_agents']).to(device) for _ in range(4)]
        
        critics[0].load_state_dict(torch.load("ma_models/intersection_0_episode__2000_3.5_critic.pth"))
        critics[1].load_state_dict(torch.load("ma_models/intersection_1_episode__2000_3.5_critic.pth"))
        critics[2].load_state_dict(torch.load("ma_models/intersection_2_episode__2000_3.5_critic.pth"))
        critics[3].load_state_dict(torch.load("ma_models/intersection_3_episode__2000_3.5_critic.pth"))
        [critic.eval() for critic in critics]
        
        avg_reward = 0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(mp4_dir, f'intersect.mp4'), fourcc, 2.0, (600, 600))
        for episode_index in range(10):
            env = MultiAgentIntersectionEnv(render_mode='rgb_array')
            env.configure({
                "id": "intersection-multi-agent-v0",
                "import_module": "highway_env",
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {
                        "type": "Kinematics",
                        "vehicles_count": 3,
                        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                        "features_range": {
                            "x": [-100, 100],
                            "y": [-100, 100],
                            "vx": [20, 40],
                            "vy": [20, 40]
                        },
                        "absolute": True,
                        "order": "shuffled"
                    }
                },
                "initial_vehicle_count": 0,
                "controlled_vehicles": 4,
                "collision_reward": -4,
                "high_speed_reward": 0,
                "arrived_reward": 2
            })
            
            total_reward = 0
            (obs, _), done = env.reset(), False

            while not done:
                obs = np.concatenate(obs, axis=0).reshape(1, -1)[0]
                obs = torch.tensor(obs, dtype=torch.float32).to(device)
                action_space = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0]).to(device)
                if self.setup == 5: action_space = torch.tensor([0, 0, 0]).to(device)
                actions = []
                for i in range(4):
                    if self.setup != 4:
                        actions.append(torch.argmax(critics[i](obs, action_space)).cpu())
                    else:
                        actions.append(torch.argmax(critics[i](obs)).cpu())
                next_obs, reward, done, _, _ = env.step(tuple(actions))
                obs = next_obs
                image_rgb = env.render()
                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                out.write(image_rgb)
                total_reward += reward
            avg_reward += total_reward
        out.release()
        print(avg_reward/10)
