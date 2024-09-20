import gym, cv2, time, random, torch
import numpy as np

class maEnv(gym.Env):
    def __init__(self):
        super(maEnv, self).__init__()

    def step(self, actions, order=[0,1]):
        reward = np.array([0.0,0.0])
        players_on_purple_gem = [False, False]
        player_gems = [-1, -1] 

        for name in order:
            min_distance = float('inf')
            for i in range(3):
                gem_position = self.gem_positions[i+6]
                player_position = self.players[name]
                distance = np.linalg.norm(np.array(gem_position) - np.array(player_position))
                if distance < min_distance:
                    min_distance = distance

            for i in range(3):
                if self.players[name] == self.gem_positions[i+6]:
                    players_on_purple_gem[name] = True
                    player_gems[name] = i+6

            self.num_steps += 1
            action = actions[name]
            if action == 1 and self.players[name][0] < 700:
                self.players[name][0] += 50
            elif action == 3 and self.players[name][0] > 0:
                self.players[name][0] -= 50
            elif action == 0 and self.players[name][1] < 700:
                self.players[name][1] += 50
            elif action == 2 and self.players[name][1] > 0:
                self.players[name][1] -= 50

            for i in range(3):
                if self.players[name] == self.gem_positions[i+6]: 
                    players_on_purple_gem[name] = True
                    player_gems[name] = i+6
            
            gem_reward = 0
            for gem_index in range(3):
                gem_index += 3*name
                if self.players[name] == self.gem_positions[gem_index]:
                    gem_reward = 1
                    self.total_gems += 1
                    self.gem_positions[gem_index] = [random.randrange(1, 15) * 50, random.randrange(1, 15) * 50]
            reward[name] += gem_reward

        if all(players_on_purple_gem):
                reward += 6
                for gem_index in set(player_gems): 
                    if gem_index != -1:
                        self.total_gems += 1
                        self.gem_positions[gem_index] = [random.randrange(1, 15) * 50, random.randrange(1, 15) * 50]
        self.sum_reward += np.sum(reward)
        if self.total_gems > 6: self.done = True
        return self.get_obs_map(), reward, self.done, {}


    def render(self, mode="human"):
        self.img = np.zeros((750, 750, 3), dtype='uint8')
        color = [(0, 0, 255), (255, 0, 0), (255, 0, 255)]
        # Display Gem
        for i in range(len(self.gem_positions)):
            gem_position = self.gem_positions[i]
            
            if i < 3:
                cv2.circle(self.img, (gem_position[0]+25, gem_position[1]+25), 25, color[0], -1)
            elif i >= 3 and i < 6:
                cv2.circle(self.img, (gem_position[0]+25, gem_position[1]+25), 25, color[1], -1)
            else:
                cv2.circle(self.img, (gem_position[0]+25, gem_position[1]+25), 25, color[2], -1)

        # Display Players
        for i in range(len(self.players)):
            position = self.players[i]
            cv2.rectangle(self.img, (position[0], position[1]), (position[0] + 50, position[1] + 50), color[i], -1)

        if mode == "human":
            cv2.imshow('MADDPG', self.img)
            t_end = time.time() + 0.10
            k = -1
            while time.time() < t_end:
                if k == -1: 
                    k = cv2.waitKey(1)
        else:
            _, buffer = cv2.imencode('.jpg', self.img)
            return np.array(buffer)
        
    def reset(self):
        self.img = np.zeros((750, 750, 3), dtype='uint8')
        self.num_steps = 0
        self.sum_reward = 0
        self.total_gems = 0
        # Initial Player and Gem position
        p = np.array(np.random.choice(range(0, 15), 22)) * 50
        self.players = [[p[0], p[1]], [p[2], p[3]]]
        self.gem_positions = [[p[4], p[5]], [p[6], p[7]], [p[8], p[9]], 
                              [p[10], p[11]], [p[12], p[13]], [p[14], p[15]], 
                              [p[16], p[17]], [p[18], p[19]], [p[20], p[21]]]
        self.done = False
        return self.get_obs_map()

    def get_obs_map(self):
        cord = torch.tensor([self.players[0][0], self.players[0][1], self.players[1][0], self.players[1][1], 
                       self.gem_positions[0][0], self.gem_positions[0][1], 
                       self.gem_positions[1][0], self.gem_positions[1][1], 
                       self.gem_positions[2][0], self.gem_positions[2][1],
                       self.gem_positions[3][0], self.gem_positions[3][1],
                       self.gem_positions[4][0], self.gem_positions[4][1],
                       self.gem_positions[5][0], self.gem_positions[5][1],
                       self.gem_positions[6][0], self.gem_positions[6][1],
                       self.gem_positions[7][0], self.gem_positions[7][1],
                       self.gem_positions[8][0], self.gem_positions[8][1]
                       ])//50
        
        new_obs = np.zeros((5, 15, 15))
        new_obs[0, cord[0], cord[1]] = 1
        new_obs[1, cord[2], cord[3]] = 1
        new_obs[2, cord[4], cord[5]] = 1
        new_obs[2, cord[6], cord[7]] = 1
        new_obs[2, cord[8], cord[9]] = 1
        new_obs[3, cord[10], cord[11]] = 1
        new_obs[3, cord[12], cord[13]] = 1
        new_obs[3, cord[14], cord[15]] = 1
        new_obs[4, cord[16], cord[17]] = 1
        new_obs[4, cord[18], cord[19]] = 1
        new_obs[4, cord[20], cord[21]] = 1
        return new_obs
    