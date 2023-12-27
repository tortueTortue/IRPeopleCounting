import json
import os

def load_config(argv):
    name = argv[1] if isinstance(argv, list) and len(argv) > 1 else argv if isinstance(argv, str) else "default_configs"
    with open(f'{os.getcwd()}/configs/{name}.json') as c:
    # with open(f'{os.getcwd()}/DistechCrowdCounting/configs/{name}.json') as c:
        config = json.load(c)
    return config

def get_dataset_root_folder(argv):
    return argv[2] if len(argv) > 2 else ""