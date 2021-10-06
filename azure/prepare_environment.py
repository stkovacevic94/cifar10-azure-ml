from azureml.core import Workspace, Environment
import json

ws = Workspace.from_config('./.config/azureml_workspace.json')
with open("./.config/auth_keys.json") as auth_keys_file:
    auth_keys = json.load(auth_keys_file)

env = Environment.from_conda_specification('PyTorch-1.9.0', 'environment.yml')
env.docker.base_image = ('mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn8-ubuntu18.04')
env.register(ws);