import argparse
import os.path
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field

import torch
import transformers
from transformers.models.llama.modeling_llama import LlamaConfig
from transformers.models.llama.tokenization_llama import LlamaTokenizer

from general_util.tokenization_utils import expand_special_tokenizer, PreTrainedTokenizer


def smart_tokenizer_and_embedding_resize(
        tokenizer: transformers.PreTrainedTokenizer,
        weight: torch.Tensor,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    # TODO: padding embedding size for being divisible by 64.
    original_vocab_size = weight.shape[0]
    num_new_tokens = len(tokenizer) - original_vocab_size

    new_embeddings = torch.nn.Embedding(len(tokenizer), weight.shape[1], dtype=weight.dtype)
    new_embeddings.weight.data[:weight.shape[0]] = weight
    new_embeddings = new_embeddings.weight.data

    if num_new_tokens > 0:
        input_embeddings_avg = new_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        new_embeddings[-num_new_tokens:] = input_embeddings_avg

    return new_embeddings


def write_ckpt(outpath: Path, model: torch.nn.Module, model_config: LlamaConfig, mp: int):
    loaded = model.state_dict()

    n_layers = model_config.num_hidden_layers
    # embedding
    sd = {"weight": loaded['model.embed_tokens.weight']}
    torch.save(sd, outpath / "layer_00-model_00-model_states.pt")
    # norm
    sd = {f"weight": loaded['model.norm.weight']}
    torch.save(sd, outpath / f"layer_{n_layers + 1}-model_00-model_states.pt")
    # lm head
    sd = {f"weight": loaded['lm_head.weight']}
    torch.save(sd, outpath / f"layer_{n_layers + 2}-model_00-model_states.pt")
    # decoder layers
    for layer_i in range(n_layers):
        sd = {nm.replace(f"model.layers.{layer_i}.", f""): weight for nm, weight in loaded.items() if
              nm.startswith(f"model.layers.{layer_i}.")}
        torch.save(sd, outpath / f"layer_{layer_i + 1:02d}-model_00-model_states.pt")

    model_state = {
        "dp_world_size": 1,
        "mp_world_size": mp,
        "module": None,
        "optimizer": None,
        "global_steps": 1,
        "skipped_steps": 1,
        "iteration": 1,
    }
    for rank in range(mp):
        torch.save(model_state, outpath / f"mp_rank_{rank:02d}_model_states.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_states", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    tokenizer: PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)

    weights = torch.load(args.model_states)["weight"]
    extended_weights = smart_tokenizer_and_embedding_resize(tokenizer, weights)

    print(weights.size())
    print(extended_weights.size())

    state_name = args.model_states.split("/")[-1]
    torch.save({"weight": extended_weights}, os.path.join(args.output_dir, state_name))



if __name__ == "__main__":
    main()
