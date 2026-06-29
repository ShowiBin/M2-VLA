from typing import Optional
from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder
SYS_PROMPTS = {'prismatic': "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.", 'openvla': "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."}

class VicunaV15ChatPromptBuilder(PromptBuilder):

    def __init__(self, model_family: str, system_prompt: Optional[str]=None) -> None:
        super().__init__(model_family, system_prompt)
        self.system_prompt = (SYS_PROMPTS[self.model_family] if system_prompt is None else system_prompt).strip() + ' '
        (self.bos, self.eos) = ('<s>', '</s>')
        self.wrap_human = lambda msg: f'USER: {msg} ASSISTANT: '
        self.wrap_gpt = lambda msg: f"{(msg if msg != '' else ' ')}{self.eos}"
        (self.prompt, self.turn_count) = ('', 0)

    def add_turn(self, role: str, message: str) -> str:
        assert role == 'human' if self.turn_count % 2 == 0 else role == 'gpt'
        message = message.replace('<image>', '').strip()
        if self.turn_count == 0:
            sys_message = self.system_prompt + self.wrap_human(message)
            wrapped_message = sys_message
        elif self.turn_count % 2 == 0:
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
        if self.turn_count == 0:
            sys_message = self.system_prompt + self.wrap_human(message)
            prompt_copy += sys_message
        else:
            human_message = self.wrap_human(message)
            prompt_copy += human_message
        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        return self.prompt.removeprefix(self.bos).rstrip()
