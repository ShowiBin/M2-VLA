from typing import Optional
from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder
SYS_PROMPTS = {'prismatic': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.', 'openvla': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'}

class QwenPromptBuilder(PromptBuilder):

    def __init__(self, model_family: str, system_prompt: Optional[str]=None) -> None:
        super().__init__(model_family, system_prompt)
        self.system_prompt = (SYS_PROMPTS[model_family] if system_prompt is None else self.system_prompt).strip()
        self.bos = self.start = '<|im_start|>'
        self.eos = '<|endoftext|>'
        self.end = '<|im_end|>'
        self.wrap_system = lambda msg: f'{self.start}system\n{msg}{self.end}\n'
        self.wrap_human = lambda msg: f'{self.start}user\n{msg}{self.end}\n{self.start}assistant\n'
        self.wrap_gpt = lambda msg: f"{(msg if msg != '' else ' ')}{self.end}\n"
        (self.prompt, self.turn_count) = ('', 0)

    def add_turn(self, role: str, message: str) -> str:
        assert role == 'human' if self.turn_count % 2 == 0 else role == 'gpt'
        message = message.replace('<image>', '').strip()
        if self.turn_count == 0 and self.system_prompt is not None:
            self.prompt += self.wrap_system(self.system_prompt)
        if self.turn_count % 2 == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message
        self.prompt += wrapped_message
        self.turn_count += 1
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        prompt_copy = str(self.prompt)
        human_message = self.wrap_human(message)
        prompt_copy += human_message
        return prompt_copy

    def get_prompt(self) -> str:
        if self.turn_count % 2 == 0:
            assert self.prompt[-1] == '\n', f'malformed prompt ({self.prompt}) missing newline before EOS append!'
            return self.prompt[:-1] + self.eos
        return self.prompt
