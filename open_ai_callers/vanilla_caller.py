import time

import openai
import traceback
from typing import Union, List

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class GPTAPIInterface:
    def __init__(self, model: str, max_tokens: int, api_time_interval: int = 2):
        self.model = model
        self.max_tokens = max_tokens
        self.api_time_interval = api_time_interval

    def __call__(self, text: str):
        raise NotImplementedError


class GPTTurbo(GPTAPIInterface):
    def __init__(self,
                 model: str = "gpt-3.5-turbo",
                 max_tokens: int = 2048,
                 temperature: float = 0.0,
                 top_p: int = 1,
                 frequency_penalty: float = 0.0,
                 presence_penalty: float = 0.0,
                 api_time_interval: int = 2,
                 organization: str = ""):
        super().__init__(model, max_tokens, api_time_interval)
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.organization = organization

    def __call__(self, text: Union[str, List[str]]):
        if isinstance(text, list):
            if isinstance(text[0], str):
                assert len(text) == 1, "Currently we only support one input in single batch."
                text = text[0]
            else:
                assert isinstance(text[0], dict)
                assert "role" in text[0] and "content" in text[0]

        flag = False
        error_time = 0
        response = None
        max_tokens = self.max_tokens
        while not flag:
            try:
                if isinstance(text, str):
                    messages = [{"role": "user", "content": text}]
                else:
                    messages = text
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    organization=self.organization if self.organization else None,
                )
                error_time = 0
                flag = True
            except Exception as exc:
                logger.warning(exc)
                logger.warning(">>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                err_msg = traceback.format_exc()
                # logger.warning(traceback.print_exc())
                logger.warning(err_msg)
                logger.warning(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                if "maximum context" in err_msg:
                    max_tokens -= 100
                    logger.warning("max_tokens: {}".format(max_tokens))
                error_time += 1
                if error_time > 20:
                    logger.warning("Too many errors. Sleep 60s.")
                    time.sleep(60)
                    return {"response": ""}

        if self.api_time_interval:
            time.sleep(self.api_time_interval)
        return {"response": response["choices"][0]["message"]["content"]}