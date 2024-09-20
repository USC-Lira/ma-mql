import torch, os, io, random, imageio
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from gems.env import maEnv
from models import Critic_Gems
from gems.dataset import ExpertSet
import torch.optim.lr_scheduler as lr_scheduler
from train_update import update_setups
from collections import deque
import statistics
import torch.nn as nn

dir = "ma_models/"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
env = maEnv()

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
        self.replay_buffer.load("expert_data/buffer_init/rb_gem.csv")
        obs, _, acts, _, _ = expert_data.sample(1)
        if setup < 4: self.critic = Critic_Gems(len(acts[0])).to(device)
        elif setup == 4: self.critic = Critic_Gems().to(device)
        elif setup == 5: self.critic = Critic_Gems(len(acts[0])//self.config['num_agents']).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config['critic_lr']) #, weight_decay=0.0001
        self.scheduler = lr_scheduler.LinearLR(self.critic_optimizer, start_factor=1.0, end_factor=0.5, total_iters=40)
        self.expert_data = expert_data
        self.max_reward = 6
        self.setup = setup
        self.running_pred_rew = deque(maxlen=100)
        self.mse_loss = nn.MSELoss()
    
    def learn(self, agent, episode, critic2):
        self.critic.train()
        critic2.eval()
        obs, next_obs, acts, r, d = self.expert_data.sample(self.config['batch_size'])
        obs = torch.tensor(obs, dtype=torch.float32, requires_grad=True).to(device)
        actions = torch.tensor(acts, dtype=torch.float32, requires_grad=True).to(device)
        r = torch.tensor(r, dtype=torch.float32, requires_grad=True).to(device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, requires_grad=True).to(device) 
        d = torch.reshape(torch.tensor(d, dtype=torch.float32, requires_grad=True).to(device), (-1,))
        if agent: acts2, acts = actions[:, :5], actions[:, 5:]
        else: acts2, acts = actions[:, 5:], actions[:, :5]

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
        loss = pre_reg_loss + 0.00001 * l2_reg 

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        self.critic.eval()
        test_env = maEnv()
        reward_ep = []
        if (episode+1) % 100 == 0: eval = 10 
        else: eval = 1
        for _ in range(eval):
            total_reward = 0
            curr_state = test_env.reset()
            done, steps = False, 0
            while steps < self.config['test_iteration'] and not done:
                curr_state = torch.tensor(curr_state, dtype=torch.float32).to(device).unsqueeze(0)
                action_space = torch.zeros(1, 10).to(device)
                if self.setup == 5: action_space = torch.zeros(1, 5).to(device)
                if self.setup != 4:
                    action0 = F.one_hot(torch.multinomial(F.softmax(self.critic(curr_state, action_space), dim=1), num_samples=1)[0], 5)[0]
                    action1 = F.one_hot(torch.multinomial(F.softmax(critic2(curr_state, action_space), dim=1), num_samples=1)[0], 5)[0]
                else:
                    action0 = F.one_hot(torch.multinomial(F.softmax(self.critic(curr_state), dim=1), num_samples=1)[0], 5)[0]
                    action1 = F.one_hot(torch.multinomial(F.softmax(critic2(curr_state), dim=1), num_samples=1)[0], 5)[0]
                if agent: action0, action1 = action1, action0
                next_state, reward, done, _ = test_env.step([np.argmax(action0.cpu()), np.argmax(action1.cpu())])
                if (episode+1) % 100 != 0:
                    self.replay_buffer.push((curr_state, next_state, [torch.argmax(action0).cpu(), torch.argmax(action1).cpu()], reward, done))
                total_reward += reward
                curr_state = next_state
                steps += 1
            reward_ep.append(total_reward)
            
        if (episode+1) % 100 == 0:
            avg_reward = np.mean(reward_ep)
            print(str(agent)+"_Episode_" + str(episode+1) +"_Loss_" +str(loss.item())+"_Reward_" + str(avg_reward)+"_" + str(reward_ep))
            print("Imitation_Loss: "+ str(statistics.mean(self.running_pred_rew)))
            if self.max_reward <= avg_reward:
                self.max_reward = avg_reward
                torch.save(self.critic.state_dict(), dir +"gems_"+ str(agent) +"model_" + str(avg_reward)+"_ep_"+str(episode)+".critic.pth")
                torch.save(critic2.state_dict(), dir +"gems_"+ str(abs(agent-1)) +"model_" + str(avg_reward)+"_ep_"+str(episode)+".critic.pth")

class gems_ma_irl:
    def __init__(self, configs, setup):
        self.config = configs
        self.setup = setup
    
    def train(self):
        expert_data = ExpertSet("expert_data/gem_dataset.csv", self.config['dataset_episodes'], self.config['test_iteration'])
        print("loaded data")
        
        agent1 = OfflineSACIQLearn(expert_data, self.config, self.setup)
        agent2 = OfflineSACIQLearn(expert_data, self.config, self.setup)
        agents = [
            (agent1, 0, agent2.critic),
            (agent2, 1, agent1.critic)
        ]
        for episode in range(self.config['episodes']):
            random.shuffle(agents)
            for agent, idx, critic2 in agents:
                agent.learn(idx, episode, critic2)

    def test(self):
        model_directory = "ma_models/"
        assert os.path.exists(model_directory)
        mp4_dir = os.path.join(model_directory, 'mp4')
        if not os.path.exists(mp4_dir):
            os.makedirs(mp4_dir)
        expert_data = ExpertSet("expert_data/gem_dataset.csv", self.config['dataset_episodes'], self.config['test_iteration'])    
        obs, _, acts, _, _ = expert_data.sample(1)    
        if self.setup < 4: 
            self.critic = Critic_Gems(len(acts[0])).to(device)
        elif self.setup == 4: 
            self.critic = Critic_Gems().to(device)
        elif self.setup == 5: 
            self.critic = Critic_Gems(len(acts[0])//self.config['num_agents']).to(device)

        model1 = Critic_Gems().to(device)
        model1.load_state_dict(torch.load("ma_models/gems_0model_13.7_ep_1399.critic.pth"))
        model1.eval()
        model2 = Critic_Gems().to(device)
        model2.load_state_dict(torch.load("ma_models/gems_1model_13.7_ep_1399.critic.pth"))
        model2.eval()
        frames = [] 
        testing_rewards = []
        test_env = maEnv()
        for i in range(10):
            curr_state = test_env.reset()
            done = False
            steps, curr_total_reward = 0, 0
            
            while steps < self.config['test_iteration']:
                curr_state = torch.tensor(curr_state, dtype=torch.float32).to(device).unsqueeze(0)
                action_space = torch.zeros(1, 10).to(device)
                if self.setup == 5: action_space = torch.zeros(1, 5).to(device)
                if self.setup != 4:
                    action0 = F.one_hot(torch.multinomial(F.softmax(model1(curr_state, action_space), dim=1), num_samples=1)[0], 5)[0]
                    action1 = F.one_hot(torch.multinomial(F.softmax(model2(curr_state, action_space), dim=1), num_samples=1)[0], 5)[0]
                else:
                    action0 = F.one_hot(torch.multinomial(F.softmax(model1(curr_state), dim=1), num_samples=1)[0], 5)[0]
                    action1 = F.one_hot(torch.multinomial(F.softmax(model2(curr_state), dim=1), num_samples=1)[0], 5)[0]
                next_state, reward, done, _ = test_env.step([np.argmax(action0.cpu()), np.argmax(action1.cpu())])
                curr_total_reward += reward
                test_env.render()
                curr_state = next_state
                steps += 1
                jpeg_array = test_env.render(mode="other")
                jpeg_bytes = jpeg_array.tobytes()
                image = Image.open(io.BytesIO(jpeg_bytes))
                frames.append(image)
                if done: break
            testing_rewards.append(curr_total_reward)
        with imageio.get_writer(os.path.join(mp4_dir, f'{i}.mp4'), fps=10) as writer:
            for frame in frames:
                writer.append_data(np.array(frame))
        print("Reward: " + str(np.mean(testing_rewards)))
            