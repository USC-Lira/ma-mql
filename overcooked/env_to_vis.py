from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
import gym, cv2, time
import matplotlib.pyplot as plt
import numpy as np

class Kitchen:
    def __init__(self):
        super(Kitchen, self).__init__()
        mdp = OvercookedGridworld.from_layout_name("cramped_room")
        base_env = OvercookedEnv.from_mdp(mdp, horizon=500)
        self.env = gym.make("Overcooked-v0",base_env = base_env, featurize_fn =base_env.featurize_state_mdp)
        self.obs = self.env.reset()
        self.state = [1, 2, 1, 3,    
                      0, 0, 0, 0, 0, 0,  
                      0, 1, 4,  1,  1, 3, 3, 3, 2, 0, 
                      0, 0, 0, 0, 0, 0, 0]

    def generate_env(self):
        obs, reward, done, env_info = self.env.step((3,3))
        self.obs = obs
        image_rgb = self.env.render()

        plt.imshow(image_rgb)
        plt.axis('off') 
        plt.show()

    def convert_inventory(self, obs, agent_loc):
        if obs.all_objects_by_type["dish"]:
            for dish in obs.all_objects_by_type["dish"]:
                if np.array_equal(np.array(dish.position), np.array(agent_loc)):
                    return [0,1,0]
        if obs.all_objects_by_type["onion"]:
            for onion in obs.all_objects_by_type["onion"]:
                if np.array_equal(np.array(onion.position), np.array(agent_loc)):
                    return [1,0,0]
        if obs.all_objects_by_type["soup"]:
            for soup in obs.all_objects_by_type["soup"]:
                if np.array_equal(np.array(soup.position), np.array(agent_loc)):
                    return [0,0,1]
        return [0,0,0]

    def reset(self):
        state = [1, 2, 1, 3,    
                      0, 0, 0, 0, 0, 0,  
                      0, 1, 4,  1,  1, 3, 3, 3, 2, 0, 
                      0, 0, 0, 0, 0, 0, 0]
        # player 
        state[0:2] = np.array(self.obs["overcooked_state"].player_positions[0])
        # player inventory
        state[4:7] = self.convert_inventory(self.obs["overcooked_state"], state[0:2])

        # player_2 
        state[2:4] = np.array(self.obs["overcooked_state"].player_positions[1])
        # player_2 inventory
        state[7:10] = self.convert_inventory(self.obs["overcooked_state"], state[2:4])

        # stove_0_ingredients
        if self.obs["overcooked_state"].all_objects_by_type["soup"]:
            state[20] = len(self.obs["overcooked_state"].all_objects_by_type["soup"][0].ingredients)
            # stove_0_timer 
            state[21] = self.obs["overcooked_state"].all_objects_by_type["soup"][0]._cooking_tick
            # stove_full 
            state[22] = int(self.obs["overcooked_state"].all_objects_by_type["soup"][0].is_cooking)
        else: state[20:23] = [0,0,0]
        state[23:27] = np.array(self.obs["overcooked_state"].player_orientations).flatten()
        return state
    
    def render(self, mode="human"):
        image_rgb = self.env.render()
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if mode == "human":
            cv2.imshow('Overcooked', image_bgr)
            t_end = time.time() + 0.1
            while time.time() < t_end:
                cv2.waitKey(1)
        else:
            _, buffer = cv2.imencode('.jpg', image_bgr)
            return np.array(buffer)
        
    def create_state(self, obs):
        state = [1, 2, 1, 3,    
                      0, 0, 0, 0, 0, 0,  
                      0, 1, 4,  1,  1, 3, 3, 3, 2, 0, 
                      0, 0, 0, 0, 0, 0, 0]
        # player 
        state[0:2] = np.array(obs.player_positions[0])
        # player inventory
        state[4:7] = self.convert_inventory(obs, state[0:2])

        # player_2 
        state[2:4] = np.array(obs.player_positions[1])
        # player_2 inventory
        state[7:10] = self.convert_inventory(obs, state[2:4])

        # stove_0_ingredients
        if obs.all_objects_by_type["soup"]:
            state[20] = len(obs.all_objects_by_type["soup"][0].ingredients)
            # stove_0_timer 
            state[21] = obs.all_objects_by_type["soup"][0]._cooking_tick
            # stove_full 
            state[22] = int(obs.all_objects_by_type["soup"][0].is_cooking)
        else: state[20:23] = [0,0,0]
        state[23:27] = np.array(obs.player_orientations).flatten()
        return state
    