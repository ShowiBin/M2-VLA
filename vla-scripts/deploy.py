import os.path
import json_numpy
json_numpy.patch()
import json
import logging
import numpy as np
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from experiments.robot.openvla_utils import get_vla, get_vla_action, get_action_head, get_processor, get_proprio_projector
from experiments.robot.robot_utils import get_image_resize_size
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX

def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    return f'In: What action should the robot take to {instruction.lower()}?\nOut:'

class OpenVLAServer:

    def __init__(self, cfg) -> Path:
        self.cfg = cfg
        self.vla = get_vla(cfg)
        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, PROPRIO_DIM)
        self.action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            self.action_head = get_action_head(cfg, self.vla.llm_dim)
        assert cfg.unnorm_key in self.vla.norm_stats, f'Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!'
        self.processor = None
        self.processor = get_processor(cfg)
        self.resize_size = get_image_resize_size(cfg)

    def get_server_action(self, payload: Dict[str, Any]) -> str:
        try:
            if (double_encode := ('encoded' in payload)):
                assert len(payload.keys()) == 1, 'Only uses encoded payload!'
                payload = json.loads(payload['encoded'])
            observation = payload
            instruction = observation['instruction']
            action = get_vla_action(self.cfg, self.vla, self.processor, observation, instruction, action_head=self.action_head, proprio_projector=self.proprio_projector, use_film=self.cfg.use_film)
            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:
            logging.error(traceback.format_exc())
            logging.warning("Your request threw an error; make sure your request complies with the expected format:\n{'observation': dict, 'instruction': str}\n")
            return 'error'

    def run(self, host: str='0.0.0.0', port: int=8777) -> None:
        self.app = FastAPI()
        self.app.post('/act')(self.get_server_action)
        uvicorn.run(self.app, host=host, port=port)

@dataclass
class DeployConfig:
    host: str = '0.0.0.0'
    port: int = 8777
    model_family: str = 'openvla'
    pretrained_checkpoint: Union[str, Path] = ''
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 3
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 25
    unnorm_key: Union[str, Path] = ''
    use_relative_actions: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    seed: int = 7

@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OpenVLAServer(cfg)
    server.run(cfg.host, port=cfg.port)
if __name__ == '__main__':
    deploy()
