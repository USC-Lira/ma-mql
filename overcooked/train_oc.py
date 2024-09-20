import torch, random, cv2, time, io, gym, os, statistics, imageio
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import torch.nn as nn
from overcooked.env_ocm import OverCooked
from models import Critic_OC
from overcooked.dataset_ocm_agent import ExpertSet
from overcooked.env_to_vis import Kitchen
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from train_update import update_setups

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
            x.append(np.array(X, copy=False))
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
        self.replay_buffer.load("expert_data/buffer_init/rb_oc.csv")
        obs, _, acts, _, _ = expert_data.sample(1)
        if setup < 4: self.critic = Critic_OC(len(obs[0]), len(acts[0])).to(device)
        elif setup==4: self.critic = Critic_OC(len(obs[0]), 0).to(device)
        elif setup==5: self.critic = Critic_OC(len(obs[0]), len(acts[0])//self.config['num_agents']).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config['critic_lr'])
        self.expert_data = expert_data
        self.env = OverCooked()
        self.setup = setup
        self.running_pred_rew = deque(maxlen=100)
        self.mse_loss = nn.MSELoss()
    
    def getV(self, obs, acts2):
        q = self.critic(obs, acts2)
        v = self.config['alpha'] * torch.logsumexp(q/self.config['alpha'], dim=1)
        return v

    def learn(self, agent, episode, critic2):
        self.critic.train()
        critic2.eval()
        obs, next_obs, acts, r, d = self.expert_data.sample(self.config['batch_size'])
        obs = torch.tensor(obs, dtype=torch.float32, requires_grad=True).to(device)
        actions = torch.tensor(acts, dtype=torch.float32, requires_grad=True).to(device)
        r = torch.tensor(r, dtype=torch.float32, requires_grad=True).to(device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, requires_grad=True).to(device) 
        d = torch.reshape(torch.tensor(d, dtype=torch.float32, requires_grad=True).to(device), (-1,))
        if agent: acts2, acts = actions[:, :6], actions[:, 6:]
        else: acts2, acts = actions[:, 6:], actions[:, :6]

        if self.setup == 1:
            pre_reg_loss, mse_predicted_reward = update_setups().iql_ma(self.critic, obs, next_obs, actions, acts, self.config['gamma'], d, self.config['alpha'], r, agent)
        elif self.setup == 2:
            pre_reg_loss, mse_predicted_reward = update_setups().mamql_offline(self.mse_loss, self.critic, obs, next_obs, actions, acts, self.config['gamma'], d, self.config['alpha'], r, agent)
        elif self.setup == 4:
            pre_reg_loss, mse_predicted_reward = update_setups().mamql_online([self.critic, critic2], obs, next_obs, actions, self.config['gamma'], 
                agent, d, self.config['batch_size'], self.config['num_agents'], 
                device, self.replay_buffer, self.config['action_space'])
        elif self.setup == 5:
            pre_reg_loss, mse_predicted_reward = update_setups().behavioral_clone(self.critic, obs, acts, device)
        
        self.running_pred_rew.append(mse_predicted_reward)
                                                
        l2_reg = torch.tensor(0.).to(device)
        for param in self.critic.parameters():
            l2_reg += torch.norm(param)**2
        loss = pre_reg_loss + 0.0001 * l2_reg
        
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        
        if (episode+1) % 100 == 0:
            self.critic.eval()
            total_reward = 0
            shaped_rewards, discrete_rewards = np.array([0,0]), np.array([0,0])
            mdp = OvercookedGridworld.from_layout_name("cramped_room")
            mdp.start_player_positions = [(1,2),(3,1)]
            base_env = OvercookedEnv.from_mdp(mdp, horizon=500)
            env = gym.make("Overcooked-v0",base_env = base_env, featurize_fn =base_env.featurize_state_mdp)  
            obs_oc = env.reset()
            kitchen = Kitchen()
            steps = 0
            while steps < self.config['test_iteration']:
                state = kitchen.create_state(obs_oc["overcooked_state"])
                action_space = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).to(device)
                if self.setup == 5: action_space = torch.tensor([0, 0, 0, 0, 0, 0]).to(device)
                if self.setup != 4:
                    action0 = torch.argmax(F.softmax(self.critic(torch.tensor(state, dtype=torch.float32).to(device), action_space), dim = 0).squeeze()).item()
                    action1 = torch.argmax(F.softmax(critic2(torch.tensor(state, dtype=torch.float32).to(device), action_space), dim = 0).squeeze()).item()
                else:
                    action0 = torch.argmax(F.softmax(self.critic(torch.tensor(state, dtype=torch.float32).to(device)), dim = 0).squeeze()).item()
                    action1 = torch.argmax(F.softmax(critic2(torch.tensor(state, dtype=torch.float32).to(device)), dim = 0).squeeze()).item()

                if agent: action1, action0 = action0, action1
                else: action1, action0 = action1, action0
                if action0 == 1 or action0 == 0: action0 = abs(1-action0)
                if action1 == 1 or action1 == 0: action1 = abs(1-action1)
                obs_oc, reward, done, env_info = env.step((action0, action1))
                if (episode+1) % 100 != 0: self.replay_buffer.push((state, kitchen.create_state(obs_oc["overcooked_state"]), [action0, action1], reward, done))
                shaped_rewards += np.array(env_info["shaped_r_by_agent"])
                discrete_rewards += np.array(env_info["sparse_r_by_agent"])
                total_reward += reward
                steps += 1

            print(str(episode+1) + "_" +str(agent)+" avg_reward: "+ str(total_reward/10) +" real_reward: " + str(discrete_rewards[agent]/10) +" shaped_reward: " + str(shaped_rewards[agent]/10) +" Loss: " + str(loss.item()))# + " imitation_loss: " + str(imitation_loss.item()))
            print("Imitation_Loss: "+ str(statistics.mean(self.running_pred_rew)))
            if int(total_reward):    
                torch.save(self.critic.state_dict(), dir + "overcooked_" + str(agent) +"model_" + str(total_reward/1/2) + "_" + str(episode) + "_" + str(loss.item()) + ".critic.pth")
                torch.save(critic2.state_dict(), dir +"overcooked_"+ str(abs(agent-1)) +"model_" + str(total_reward/1/2) + "_" + str(episode) + "_" + str(loss.item()) + ".critic.pth")

class oc__ma_irl:
    def __init__(self, configs, setup):
        self.config = configs
        self.setup = setup
    
    def train(self):
        expert_data = ExpertSet("expert_data/oc_dataset.csv", self.config['dataset_episodes'], self.config['test_iteration'])
        print("loaded data")
        
        for _ in range(self.config['episodes']):
            agent1 = OfflineSACIQLearn(expert_data, self.config, self.setup)
            agent2 = OfflineSACIQLearn(expert_data, self.config, self.setup)
            agents = [
                (agent1, 0, agent2.critic),
                (agent2, 1, agent1.critic)
            ]
            for episode in range(30000):
                random.shuffle(agents)
                for agent, idx, critic2 in agents:
                    agent.learn(idx, episode, critic2)

    def test(self):
        model_directory = "ma_models/"
        assert os.path.exists(model_directory)
        mp4_dir = os.path.join(model_directory, 'mp4')
        if not os.path.exists(mp4_dir):
            os.makedirs(mp4_dir)
        expert_data = ExpertSet("expert_data/oc_dataset.csv", self.config['dataset_episodes'], self.config['test_iteration'])
        obs, _, acts, _, _ = expert_data.sample(1)
        if self.setup < 4: 
            model1, model2 = Critic_OC(len(obs[0]), len(acts[0])).to(device), Critic_OC(len(obs[0]), len(acts[0])).to(device)
        elif self.setup == 4: 
            model1, model2 = Critic_OC(len(obs[0]), 0).to(device), Critic_OC(len(obs[0]), 0).to(device)
        elif self.setup == 5: 
            model1, model2 = Critic_OC(len(obs[0]), len(acts[0])//self.config['num_agents']).to(device), Critic_OC(len(obs[0]), len(acts[0])//self.config['num_agents']).to(device)

        model1.load_state_dict(torch.load("ma_models/0model_1.0_1.02.critic.pth"))
        model1.eval()
        model2.load_state_dict(torch.load("ma_models/0model_1.0_1.02.critic.pth"))
        model2.eval()

        total_reward = np.array([0,0])
        for i in range(10):
            mdp = OvercookedGridworld.from_layout_name("cramped_room")
            mdp.start_player_positions = [(1,2),(3,1)]
            base_env = OvercookedEnv.from_mdp(mdp, horizon=500)
            env = gym.make("Overcooked-v0",base_env = base_env, featurize_fn =base_env.featurize_state_mdp)  
            steps = 0
            obs_oc = env.reset()
            test_frame = env.render()
            frame_height, frame_width, _ = test_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(os.path.join(mp4_dir, f'episode_{i}.mp4'), fourcc, 5.0, (frame_width, frame_height))
            kitchen = Kitchen()

            while steps < self.config['test_iteration']:
                state = kitchen.create_state(obs_oc["overcooked_state"])
                image_rgb = env.render()
                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                out.write(image_rgb)

                action_space = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).to(device)
                if self.setup == 5: action_space = torch.tensor([0, 0, 0, 0, 0, 0]).to(device)
    
                if self.setup != 4:
                    action0 = torch.argmax(F.softmax(model1(torch.tensor(state, dtype=torch.float32).to(device), action_space), dim=0).squeeze()).item()
                    action1 = torch.argmax(F.softmax(model2(torch.tensor(state, dtype=torch.float32).to(device), action_space), dim=0).squeeze()).item()
                else:
                    action0 = torch.argmax(F.softmax(model1(torch.tensor(state, dtype=torch.float32).to(device)), dim=0).squeeze()).item()
                    action1 = torch.argmax(F.softmax(model2(torch.tensor(state, dtype=torch.float32).to(device)), dim=0).squeeze()).item()

                if action0 == 1 or action0 == 0: action0 = abs(1-action0)
                if action1 == 1 or action1 == 0: action1 = abs(1-action1)

                obs_oc, reward, _, env_info = env.step((action0, action1))
                total_reward += np.array(env_info["shaped_r_by_agent"])
                steps += 1
            out.release()
        print("Reward: " + str(total_reward / 1000))
