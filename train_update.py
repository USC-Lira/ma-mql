import torch
import torch.nn.functional as F
from gems.env import maEnv

class update_setups:
    def __init__(self):
        self.gem_env = maEnv()
        self.mse_loss = torch.nn.MSELoss()

    def getV(critic, obs, acts2, alpha):
        q = critic(obs, acts2)
        v = alpha * torch.logsumexp(q/alpha, dim=1)
        return v
    
    def behavioral_clone(self, critic, obs, acts, device):
        outputs = critic(obs, acts).to(device)
        return F.cross_entropy(outputs, acts), 0
    
    def iql_ma(self, critic, obs, next_obs, actions, acts, gamma, d, alpha, rewards, agent):
        current_Q = torch.sum(critic(obs, actions) * acts, dim = 1)
        y = (1 - d) * gamma * update_setups.getV(critic, next_obs, actions, alpha)
        reward = current_Q - y
        loss = -(reward).mean()
        value_loss = (update_setups.getV(critic, obs, actions, alpha) - y).mean()
        loss += value_loss
        chi2_loss = 1/(4 * alpha) * (reward**2).mean()  
        if rewards.ndim == 2 and rewards.shape[1] > 1: rewards = rewards[:, agent]
        return loss +chi2_loss, self.mse_loss(rewards, reward).item() 
    
    def mamql_offline_variant(self, criterion, critic, obs, next_obs, actions, acts, gamma, d, alpha, rewards, agent):
        current_Q = torch.sum(critic(obs, actions) * acts, dim = 1)
        y = (1 - d) * gamma * update_setups.getV(critic, next_obs, actions, alpha)
        predicted_reward = current_Q - y
        value_loss = update_setups.getV(critic, obs, actions, alpha) - y
        if rewards.ndim == 2 and rewards.shape[1] > 1: rewards = rewards[:, agent]
        return criterion(value_loss, predicted_reward), self.mse_loss(rewards, predicted_reward).item() 
    
    def mamql_offline(self, critics, obs, next_obs, actions, gamma, alpha, done, batch_size, num_agents, device, replay_buffer, action_space):
        inits_sample = torch.tensor(np.array([self.gem_env.reset() for _ in range(batch_size)]), dtype=torch.float32, requires_grad=True).to(device)
        M, N = batch_size, num_agents
        expert_states, expert_next_states, expert_actions = obs, next_obs, actions
        expert_actions = [torch.nonzero(sample).squeeze().tolist() for sample in expert_actions]
        expert_actions = torch.tensor([[index if i == 0 else index - 5 for i, index in enumerate(indices)] for indices in expert_actions], dtype=torch.float32, requires_grad=True).to(device)
        critic_losses = torch.zeros(N).to(device)

        expert_curr_qs = torch.zeros(batch_size, N, 5).to(device)
        expert_next_qs = torch.zeros(batch_size, N, 5).to(device)
        init_qs = torch.zeros(batch_size, N, 5)
        for i in range(N):
            expert_curr_qs[:,i,:] = critics[i](expert_states)
            expert_next_qs[:,i,:] = critics[i](expert_next_states)
            init_qs[:,i,:] = critics[i](inits_sample)
        expert_next_vals = torch.logsumexp(expert_next_qs, dim=2).to(device)
        critic_losses = ((1 - gamma) * torch.mean(torch.logsumexp(init_qs, dim=2), dim=0)).to(device)
        expert_reward_est = torch.gather(expert_curr_qs, 2, expert_actions.reshape(-1, N, 1).long()).to(device).squeeze() 
        expert_reward_est -= gamma * (1-done).unsqueeze(1) * expert_next_vals
        expert_reward_est = ((expert_reward_est - expert_reward_est**2) / 4)
        critic_losses -= expert_reward_est.to(device).mean(dim=0)
        return critic_losses[0]

    def mamql_online(self, critics, obs, next_obs, actions, gamma, agent, done, batch_size, num_agents, device, replay_buffer, action_space):
        M, N = batch_size, num_agents
        expert_states, expert_next_states, expert_actions = obs, next_obs, actions
        expert_actions = [torch.nonzero(sample).squeeze().tolist() for sample in expert_actions]
        expert_actions = torch.tensor([[index if i == 0 else index - action_space for i, index in enumerate(indices)] for indices in expert_actions], dtype=torch.float32, requires_grad=True).to(device)
        if agent: expert_actions = torch.flip(expert_actions, dims=[1])
        x, y, u, r, d = replay_buffer.sample(M)
        pol_states = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device).squeeze()
        rewards = torch.tensor(r, dtype=torch.float32, requires_grad=True).to(device)
        pol_next_states = torch.tensor(y, dtype=torch.float32, requires_grad=True).to(device).squeeze()
        pol_done = torch.tensor(1-d, dtype=torch.float32, requires_grad=True).squeeze().to(device)

        critic_losses = torch.zeros(N)
        expert_curr_qs = torch.zeros(batch_size, N, action_space)
        expert_next_qs = torch.zeros(batch_size, N, action_space)
        pol_curr_qs = torch.zeros(batch_size, N, action_space)
        pol_next_qs = torch.zeros(batch_size, N, action_space)
        for i in range(N):
            expert_curr_qs[:,i,:] = critics[i](expert_states)
            expert_next_qs[:,i,:] = critics[i](expert_next_states)
            pol_curr_qs[:,i,:] = critics[i](pol_states)
            pol_next_qs[:,i,:] = critics[i](pol_next_states)
        expert_next_vals = torch.logsumexp(expert_next_qs, dim=2).to(device)
        pol_vals = torch.logsumexp(pol_curr_qs, dim=2).to(device)
        pol_next_vals = torch.logsumexp(pol_next_qs, dim=2).to(device)
        critic_losses = torch.mean(pol_vals - gamma * (pol_done.unsqueeze(1) * pol_next_vals), dim=0) 
        expert_reward_est = torch.gather(expert_curr_qs.to(device), 2, expert_actions.reshape(-1, N, 1).long()).to(device).squeeze() 
        expert_reward_est -= gamma * (1-done).unsqueeze(1) * expert_next_vals
        expert_reward_est = expert_reward_est - 0.25 * expert_reward_est**2 # chi^2 reg
        critic_losses -= expert_reward_est.to(device).mean(dim=0)
        if rewards.ndim == 2 and rewards.shape[1] > 1: rewards = rewards[:, agent]
        return critic_losses[0], self.mse_loss(rewards, (pol_vals - gamma * (pol_done.unsqueeze(1) * pol_next_vals))[:, agent]).item() 

    def mamql_online_several_agents(self, critics, obs, next_obs, actions, gamma, agent, done, batch_size, num_agents, device, replay_buffer, action_space):
        M, N = batch_size, num_agents
        expert_states, expert_next_states, expert_actions = obs, next_obs, actions
        expert_actions = expert_actions.view(-1, 4, 3)
        expert_actions = torch.argmax(expert_actions, dim=2)
        x, y, u, r, d = replay_buffer.sample(M)
        pol_states = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device).squeeze()
        rewards = torch.tensor(r, dtype=torch.float32, requires_grad=True).to(device)
        pol_next_states = torch.tensor(y, dtype=torch.float32, requires_grad=True).to(device).squeeze()
        pol_done = torch.tensor(1-d, dtype=torch.float32, requires_grad=True).squeeze().to(device)

        critic_losses = torch.zeros(N)
        expert_curr_qs = torch.zeros(batch_size, N, action_space)
        expert_next_qs = torch.zeros(batch_size, N, action_space)
        pol_curr_qs = torch.zeros(batch_size, N, action_space)
        pol_next_qs = torch.zeros(batch_size, N, action_space)
        for i in range(N):
            expert_curr_qs[:,i,:] = critics[i](expert_states)
            expert_next_qs[:,i,:] = critics[i](expert_next_states)
            pol_curr_qs[:,i,:] = critics[i](pol_states)
            pol_next_qs[:,i,:] = critics[i](pol_next_states)
        expert_next_vals = torch.logsumexp(expert_next_qs, dim=2).to(device)
        pol_vals = torch.logsumexp(pol_curr_qs, dim=2).to(device)
        pol_next_vals = torch.logsumexp(pol_next_qs, dim=2).to(device)
        critic_losses = torch.mean(pol_vals - gamma * (pol_done.unsqueeze(1) * pol_next_vals), dim=0) 
        expert_reward_est = torch.gather(expert_curr_qs.to(device), 2, expert_actions.reshape(-1, N, 1).long()).to(device).squeeze() 
        expert_reward_est -= gamma * (1-done).unsqueeze(1) * expert_next_vals
        expert_reward_est = expert_reward_est - 0.25 * expert_reward_est**2 # chi^2 reg
        critic_losses -= expert_reward_est.to(device).mean(dim=0)
        return critic_losses[agent], self.mse_loss(rewards, (pol_vals - gamma * (pol_done.unsqueeze(1) * pol_next_vals))[:, agent]).item() 
    