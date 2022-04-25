import json

from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PaddingStrategy

from data.collators.dict2dict import DictTensorDataset
from general_util.logger import get_child_logger

logger = get_child_logger("SeqCls")


def deduction_classification_for_infer(file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int):
    """
    Deductions are classified as "yes" or "no".
    For inference only.
    """
    data = json.load(open(file_path, "r"))
    all_input_a = []
    all_input_b = []

    for item in data:
        passage = item['passage']
        for q in item['questions']:
            ques = q['question']
            if "prediction" in q:
                pred_deduction = q['prediction']
                all_input_a.append(f"{passage} {ques}")
                all_input_b.append(pred_deduction)

    model_inputs = tokenizer(all_input_a, text_pair=all_input_b,
                             max_length=max_seq_length, return_tensors="pt",
                             padding=PaddingStrategy.LONGEST, truncation=True)

    return DictTensorDataset(model_inputs)
