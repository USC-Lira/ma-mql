import argparse, torch, json
from intersection.train_intersection import intersection_ma_irl
from overcooked.train_oc import oc__ma_irl
from gems.train_gem import gems_ma_irl

iqlearn_dir = "ma_models/"
parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default='intersection', type=str) 
parser.add_argument('--config_path', default='', type=str) 
parser.add_argument('--setup', default=1, type=int) 
parser.add_argument('--mode', default='train', type=str) 
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.config_path == "":
    with open(args.env_name+'/configs/baseline.json', 'r') as f:
        env_config = json.load(f)
else:
    with open(args.config_path, 'r') as f:
        env_config = json.load(f)

active_env = None
if args.env_name == 'intersection':
    active_env = intersection_ma_irl(env_config, args.setup)
elif args.env_name == 'overcooked':
    active_env = oc__ma_irl(env_config, args.setup)
elif args.env_name == 'gems':
    active_env = gems_ma_irl(env_config, args.setup)
else:
    print("selected invalid environment")

if args.mode == 'train':
    active_env.train()
else: active_env.test()
