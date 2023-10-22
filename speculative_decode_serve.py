import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.llama import LlamaForCausalLMWithAssistant

model = LlamaForCausalLMWithAssistant.from_pretrained_eval_tp("pretrained-models/Llama-2-70b-chat-hf/", assistant_model_config=None,
                                                              assistant_model=AutoModelForCausalLM.from_pretrained(
                                                                  "pretrained-models/Llama-2-7b-hf/", torch_dtype=torch.bfloat16))

tokenizer = AutoTokenizer.from_pretrained("pretrained-models/Llama-2-70b-chat-hf/")

