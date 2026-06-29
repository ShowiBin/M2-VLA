import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import draccus
import requests
import torch
from PIL import Image
from prismatic import load
from prismatic.overwatch import initialize_overwatch
overwatch = initialize_overwatch(__name__)
DEFAULT_IMAGE_URL = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'

@dataclass
class GenerateConfig:
    model_path: Union[str, Path] = 'prism-dinosiglip+7b'
    hf_token: Union[str, Path] = Path('.hf_token')
    do_sample: bool = False
    temperature: float = 1.0
    max_new_tokens: int = 512
    min_length: int = 1

@draccus.wrap()
def generate(cfg: GenerateConfig) -> None:
    overwatch.info(f'Initializing Generation Playground with Prismatic Model `{cfg.model_path}`')
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    vlm = load(cfg.model_path, hf_token=hf_token)
    vlm.to(device, dtype=torch.bfloat16)
    image = Image.open(requests.get(DEFAULT_IMAGE_URL, stream=True).raw).convert('RGB')
    prompt_builder = vlm.get_prompt_builder()
    system_prompt = prompt_builder.system_prompt
    print(f"[*] Dropping into Prismatic VLM REPL with Default Generation Setup => Initial Conditions:\n       => Prompt Template:\n\n{prompt_builder.get_potential_prompt('<INSERT PROMPT HERE>')}\n\n       => Default Image URL: `{DEFAULT_IMAGE_URL}`\n===\n")
    repl_prompt = '|=>> Enter (i)mage to fetch image from URL, (p)rompt to update prompt template, (q)uit to exit, or any other key to enter input questions: '
    while True:
        user_input = input(repl_prompt)
        if user_input.lower().startswith('q'):
            print('\n|=>> Received (q)uit signal => Exiting...')
            return
        elif user_input.lower().startswith('i'):
            url = input('\n|=>> Enter Image URL: ')
            image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
            prompt_builder = vlm.get_prompt_builder(system_prompt=system_prompt)
        elif user_input.lower().startswith('p'):
            if system_prompt is None:
                print('\n|=>> Model does not support `system_prompt`!')
                continue
            system_prompt = input('\n|=>> Enter New System Prompt: ')
            prompt_builder = vlm.get_prompt_builder(system_prompt=system_prompt)
            print(f"\n[*] Set New System Prompt:\n    => Prompt Template:\n{prompt_builder.get_potential_prompt('<INSERT PROMPT HERE>')}\n\n")
        else:
            print('\n[*] Entering Chat Session - CTRL-C to start afresh!\n===\n')
            try:
                while True:
                    message = input('|=>> Enter Prompt: ')
                    prompt_builder.add_turn(role='human', message=message)
                    prompt_text = prompt_builder.get_prompt()
                    generated_text = vlm.generate(image, prompt_text, do_sample=cfg.do_sample, temperature=cfg.temperature, max_new_tokens=cfg.max_new_tokens, min_length=cfg.min_length)
                    prompt_builder.add_turn(role='gpt', message=generated_text)
                    print(f'\t|=>> VLM Response >>> {generated_text}\n')
            except KeyboardInterrupt:
                print('\n===\n')
                continue
if __name__ == '__main__':
    generate()
