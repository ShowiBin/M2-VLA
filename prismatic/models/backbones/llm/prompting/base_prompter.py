from abc import ABC, abstractmethod
from typing import Optional

class PromptBuilder(ABC):

    def __init__(self, model_family: str, system_prompt: Optional[str]=None) -> None:
        self.model_family = model_family
        self.system_prompt = system_prompt

    @abstractmethod
    def add_turn(self, role: str, message: str) -> str:
        ...

    @abstractmethod
    def get_potential_prompt(self, user_msg: str) -> None:
        ...

    @abstractmethod
    def get_prompt(self) -> str:
        ...

class PurePromptBuilder(PromptBuilder):

    def __init__(self, model_family: str, system_prompt: Optional[str]=None) -> None:
        super().__init__(model_family, system_prompt)
        (self.bos, self.eos) = ('<s>', '</s>')
        self.wrap_human = lambda msg: f'In: {msg}\nOut: '
        self.wrap_gpt = lambda msg: f"{(msg if msg != '' else ' ')}{self.eos}"
        (self.prompt, self.turn_count) = ('', 0)

    def add_turn(self, role: str, message: str) -> str:
        assert role == 'human' if self.turn_count % 2 == 0 else role == 'gpt'
        message = message.replace('<image>', '').strip()
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
        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        return self.prompt.removeprefix(self.bos).rstrip()
