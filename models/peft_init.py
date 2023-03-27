from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    TaskType,
    get_peft_config,
    PeftConfig,
    PeftModel,
)
from transformers import PreTrainedModel


def load_model_via_peft_lora(model: PreTrainedModel, lora_config: LoraConfig = None):
    if lora_config is None:
        lora_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

    model = get_peft_model(model, lora_config)
    return model


def load_model_via_peft_from_pretrained(model_path, model: PreTrainedModel):
    peft_config = PeftConfig.from_pretrained(model_path)
    model = PeftModel.from_pretrained(model, peft_model_id)
