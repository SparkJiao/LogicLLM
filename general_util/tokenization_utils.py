from transformers import PreTrainedTokenizer

from data.data_utils import tokenizer_get_name
from general_util.logger import get_child_logger

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

logger = get_child_logger(__name__)


def expand_special_tokenizer(tokenizer: PreTrainedTokenizer):
    if "llama" in tokenizer_get_name(tokenizer):
        special_tokens_map = {}
        if not tokenizer.eos_token:
            special_tokens_map["eos_token"] = DEFAULT_EOS_TOKEN
        if not tokenizer.bos_token:
            special_tokens_map["bos_token"] = DEFAULT_BOS_TOKEN
        if not tokenizer.unk_token:
            special_tokens_map["unk_token"] = DEFAULT_UNK_TOKEN
        if not tokenizer.pad_token:
            special_tokens_map["pad_token"] = DEFAULT_PAD_TOKEN

        new_tokens = tokenizer.add_special_tokens(
            special_tokens_dict=special_tokens_map
        )
        # new_tokens = tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN))
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        # assert new_tokens == 1
        logger.info(f"PAD TOKEN ID = {tokenizer.pad_token_id}")