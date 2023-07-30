from vllm import LLM, SamplingParams

from typing import Optional, List


class VLLMWrapper:
    def __init__(self,
                 pretrained_model_name_or_path: str,
                 tensor_parallel_size: int = 4,
                 sampling_params: Optional[SamplingParams] = None):
        self.llm = LLM(pretrained_model_name_or_path, tensor_parallel_size=tensor_parallel_size)
        self.sampling_params = sampling_params

    def eval(self):
        pass

    def __call__(self, prompts: List[str]):
        return self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
