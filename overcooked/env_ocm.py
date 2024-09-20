from gym import spaces
import numpy as np
import cv2, gym, random, time

chef_locations = set()

class Chef(gym.Env):
    def __init__(self):
        super(Chef, self).__init__()
        self.position = [random.randrange(1, 4) * 50, random.randrange(1, 3) * 50]
        while True:
            position = (random.randrange(1, 4) * 50, random.randrange(1, 3) * 50)
            if position not in chef_locations:
                chef_locations.add(position)
                break
        self.position = [position[0], position[1]]
        self.color = tuple(map(int, np.random.randint(0, 256, 3)))
        self.inventory = [0, 0, 0] # onion, plate, food/dish
        self.direction = 3

class Kitchen:
    def __init__(self):
        super(Kitchen, self).__init__()
        self.occupied_locations = set()
        self.num_onions = 2
        self.onions = self.generate_unique_locations(self.num_onions, self.occupied_locations)

        self.num_stoves = 1
        self.stoves = self.generate_unique_locations(self.num_stoves, self.occupied_locations)
        self.stove_cooking = [[0, 0, 0] for _ in range(self.num_stoves)] # num_ingrdients, cook_time, stove_full

        self.num_dishes = 1
        self.dishes = self.generate_unique_locations(self.num_dishes, self.occupied_locations)

        self.num_counters = 1
        self.counters = self.generate_unique_locations(self.num_counters, self.occupied_locations)

    def generate_unique_locations(self, num_items, occupied_locations):
        items = []
        i = 0
        for _ in range(num_items):
            while True:
                possible_locations = [(0, 50), (200, 50), (150, 150), (50, 150), (100, 0), (150, 0)]
                location = possible_locations[i]
                i+=1
                if location not in occupied_locations:
                    occupied_locations.add(location)
                    items.append(location)
                    break
        return items
        
class OverCooked(gym.Env):
    def __init__(self):
        super(OverCooked, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-500, high=500, shape=(10,), dtype=np.float32)
        self.num_agents = 2

    def is_agent_near_onion(self, agent):
        agent_x, agent_y = agent.position
        for onion_x, onion_y in self.onion_positions:
            if (abs(agent_x - onion_x) == 50 and abs(agent_y - onion_y) == 0) or (abs(agent_x - onion_x) == 0 and abs(agent_y - onion_y) == 50):
                return True
        return False
    
    def is_agent_near_stove(self, agent):
        agent_x, agent_y = agent.position
        stove_num = 0
        for stove_x, stove_y in self.stove_positions:
            if (abs(agent_x - stove_x) == 50 and abs(agent_y - stove_y) == 0) or (abs(agent_x - stove_x) == 0 and abs(agent_y - stove_y) == 50):
                return True, stove_num
            stove_num+=1
        return False, 0
    
    def is_agent_near_dishes(self, agent):
        agent_x, agent_y = agent.position
        for dishes_x, dishes_y in self.dishes_positions:
            if (abs(agent_x - dishes_x) == 50 and abs(agent_y - dishes_y) == 0) or (abs(agent_x - dishes_x) == 0 and abs(agent_y - dishes_y) == 50):
                return True
        return False
    
    def is_agent_near_counter(self, agent):
        agent_x, agent_y = agent.position
        for counter_x, counter_y in self.counters_positions:
            if (abs(agent_x - counter_x) == 50 and abs(agent_y - counter_y) == 0) or (abs(agent_x - counter_x) == 0 and abs(agent_y - counter_y) == 50):
                return True
        return False
    
    def step(self, actions, order=[0,1]):
        gem_reward = 0
        for agent, i in zip(self.agents, order):
            other_agent = self.agents[abs(1-i)].position
            if actions[i] < 4: agent.direction = actions[i]
            if actions[i] == 0 and agent.position[0] < 150 and other_agent != [agent.position[0]+50, agent.position[1]]:
                agent.position[0] += 50
            elif actions[i] == 1 and agent.position[0] > 50 and other_agent != [agent.position[0]-50, agent.position[1]]:
                agent.position[0] -= 50
            elif actions[i] == 2 and agent.position[1] < 100 and other_agent != [agent.position[0], agent.position[1]+50]:
                agent.position[1] += 50
            elif actions[i] == 3 and agent.position[1] > 50 and other_agent != [agent.position[0], agent.position[1]-50]:
                agent.position[1] -= 50
            elif actions[i] == 4:
                continue
            elif actions[i] == 5:
                near_stove, stove_num = self.is_agent_near_stove(agent)
                print(self.is_agent_near_counter(agent), agent.inventory[2] == 1)
                if self.is_agent_near_onion(agent):
                    agent.inventory = [0,0,0]
                    agent.inventory[0] = 1
                    print(str(i) +" agent got onion")
                elif agent.inventory[0] == 1 and near_stove:
                    if self.stove_cooking[stove_num][0] != self.ingred_num:
                        agent.inventory[0] = 0
                        self.stove_cooking[stove_num][0] += 1
                        print(str(i) +" you put onion on the stove")
                elif self.is_agent_near_dishes(agent):
                    agent.inventory = [0,0,0]
                    agent.inventory[1] = 1
                    print(str(i) +" you picked up a plate")
                elif near_stove and self.stove_cooking[stove_num][2] == 1:
                    self.stove_cooking[stove_num] = [0,0,0]
                    agent.inventory = [0,0,1]
                    print(str(i) +" you finished cooking!")
                elif self.is_agent_near_counter(agent) and agent.inventory[2] == 1:
                    agent.inventory = [0,0,0]
                    gem_reward += 20
                    self.total_served +=1
                    print(str(i) +" you finished an order!!!")
                    if self.total_served >= 2: self.done = True

        for i in range(self.num_stoves):
            if self.stove_cooking[i][0] == self.ingred_num and self.stove_cooking[i][1] < self.cooking_time:
                self.stove_cooking[i][1] += 1
            elif self.stove_cooking[i][0] == self.ingred_num and self.stove_cooking[i][1] >= self.cooking_time:
                self.stove_cooking[i][2] = 1
        print(self.stove_cooking)

        player_positions = [coordinate for player in self.agents for coordinate in player.position]
        player_inventories = np.array([agent.inventory for agent in self.agents]).flatten()
        observation = np.concatenate((player_positions, 
                                      player_inventories,
                                      np.array(self.onion_positions).flatten(), 
                                      np.array(self.dishes_positions).flatten(),
                                      np.array(self.counters_positions).flatten(),
                                      np.array(self.stove_positions).flatten(),
                                      np.array(self.stove_cooking).flatten(),
                                      np.array([agent.direction for agent in self.agents])))
        return observation, gem_reward, self.done, {}

    def render(self, mode="human"):
        self.img = np.zeros((500, 500, 3), dtype='uint8')
        # Display Kitchen Floor
        cv2.rectangle(self.img, (50, 50), (200, 150), (255, 255, 255), -1)
        # Display Onions
        for onion_x, onion_y in self.onion_positions:
            cv2.rectangle(self.img, (onion_x, onion_y), (onion_x + 50, onion_y + 50), (0, 102, 204), -1)

        # Display Stoves
        for stove_x, stove_y in self.stove_positions:
            cv2.rectangle(self.img, (stove_x, stove_y), (stove_x + 50, stove_y + 50), (128,128,128), -1)

        # Display Dishes
        for dishes_x, dishes_y in self.dishes_positions:
            cv2.rectangle(self.img, (dishes_x, dishes_y), (dishes_x + 50, dishes_y + 50), (250, 249, 246), -1)

        # Display Counters
        for counters_x, counters_y in self.counters_positions:
            cv2.rectangle(self.img, (counters_x, counters_y), (counters_x + 50, counters_y + 50), (92, 64, 51), -1)

        # Display Players
        for agent in self.agents:
            cv2.rectangle(self.img, (agent.position[0], agent.position[1]), 
                  (agent.position[0] + 50, agent.position[1] + 50), 
                  agent.color, -1)
            center_x = agent.position[0] + 25
            center_y = agent.position[1] + 25
            if agent.direction == 0:  # Right
                start_point = (center_x - 15, center_y)
                end_point = (center_x + 15, center_y)
            elif agent.direction == 1:  # Left
                start_point = (center_x + 15, center_y)
                end_point = (center_x - 15, center_y)
            elif agent.direction == 3:  # Up
                start_point = (center_x, center_y + 15)
                end_point = (center_x, center_y - 15)
            elif agent.direction == 2:  # Down
                start_point = (center_x, center_y - 15)
                end_point = (center_x, center_y + 15)
            cv2.arrowedLine(self.img, start_point, end_point, (0, 0, 0), 2)

        if mode == "human":
            cv2.imshow('Overcooked', self.img)
            t_end = time.time() + 0.5
            while time.time() < t_end:
                cv2.waitKey(1)
        else:
            _, buffer = cv2.imencode('.jpg', self.img)
            return np.array(buffer)

    def reset(self):
        self.done = False
        self.ingred_num = 3
        self.cooking_time = 20
        self.total_served = 0
        self.kitchen = Kitchen()
        self.onion_positions = self.kitchen.onions
        
        self.dishes_positions = self.kitchen.dishes
        self.counters_positions = self.kitchen.counters

        self.num_stoves = self.kitchen.num_stoves
        self.stove_positions = self.kitchen.stoves
        self.stove_cooking = self.kitchen.stove_cooking

        self.agents = [Chef() for _ in range(self.num_agents)]
        player_positions = [coordinate for player in self.agents for coordinate in player.position]
        player_inventories = np.array([agent.inventory for agent in self.agents]).flatten()
        observation = np.concatenate((player_positions, 
                                      player_inventories,
                                      np.array(self.onion_positions).flatten(), 
                                      np.array(self.dishes_positions).flatten(),
                                      np.array(self.counters_positions).flatten(),
                                      np.array(self.stove_positions).flatten(),
                                      np.array(self.stove_cooking).flatten(),
                                      np.array([agent.direction for agent in self.agents])))
        return observation

    def _get_obs(self):
        player_positions = [coordinate for player in self.agents for coordinate in player.position]
        player_inventories = np.array([agent.inventory for agent in self.agents]).flatten()
        observation = np.concatenate((player_positions, 
                                      player_inventories,
                                      np.array(self.onion_positions).flatten(), 
                                      np.array(self.dishes_positions).flatten(),
                                      np.array(self.counters_positions).flatten(),
                                      np.array(self.stove_positions).flatten(),
                                      np.array(self.stove_cooking).flatten(),
                                      np.array([agent.direction for agent in self.agents])))
        return {observation}
    