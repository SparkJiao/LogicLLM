import collections
import json
import os
import re
from typing import Dict, List, Any, Union, Callable

import numpy as np
import torch
from torch import distributed as dist

from post_processors.dist_mixin import DistGatherMixin
from transformers import AutoTokenizer, PreTrainedTokenizer


class NumpySaver(DistGatherMixin):
    def __init__(self, save_copy: bool = False, num_choices: int = 1):
        self.predictions = []
        self.index = []
        self.save_copy = save_copy
        self.num_choices = num_choices

    def __call__(self, meta_data: List[Dict[str, Any]], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        logits = batch_model_outputs["logits"].detach().float()
        if logits.dim() == 1:
            assert self.num_choices > 1
            logits = logits.reshape(-1, self.num_choices)
        _, pred = logits.max(dim=-1)
        pred = pred.tolist()

        index = None
        if ddp:
            assert meta_data
            if isinstance(meta_data, list):
                index = [meta['index'].item() for meta in meta_data]
            elif isinstance(meta_data, dict):
                if isinstance(meta_data["index"], torch.Tensor):
                    index = meta_data["index"].tolist()
                else:
                    index = meta_data["index"]
            else:
                raise RuntimeError()
            obj = [pred, index]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                pred = []
                index = []
                for item in gather_res:
                    pred.extend(item[0])
                    index.extend(item[1])

        if index is not None:
            self.index.extend(index)
        self.predictions.extend(pred)

    def get_results(self, output_dir: str):
        # output_file = os.path.join(output_dir, "eval_predictions.npy")
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.npy")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.npy")

        if len(self.index):
            assert len(self.index) == len(self.predictions)
            predictions = {idx: pred for idx, pred in zip(self.index, self.predictions)}
            predictions = sorted(predictions.items(), key=lambda x: x[0])
            predictions = [pred[1] for pred in predictions]
            np.save(output_file, np.array(predictions))
        else:
            np.save(output_file, np.array(self.predictions))

        if self.save_copy:
            if dist.is_initialized():
                output_file = os.path.join(output_dir, f"eval_predictions_copy_rank{dist.get_rank()}.bin")
            else:
                output_file = os.path.join(output_dir, "eval_predictions_copy.bin")

            torch.save({
                "index": self.index,
                "predictions": self.predictions
            }, output_file)

        return {}, self.predictions


class TaggingSaver(DistGatherMixin):
    def __init__(self):
        self.logits = []
        self.index = []

    def __call__(self, meta_data: List[Dict[str, Any]], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        tagging_logits = batch_model_outputs["tagging_logits"].detach().float().tolist()

        index = None
        if ddp:
            assert meta_data
            if isinstance(meta_data, list):
                index = [meta['index'].item() for meta in meta_data]
            elif isinstance(meta_data, dict):
                index = meta_data["index"].tolist()
            else:
                raise RuntimeError()
            obj = [tagging_logits, index]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                tagging_logits = []
                index = []
                for item in gather_res:
                    tagging_logits.extend(tagging_logits)
                    index.extend(item[1])

        if index is not None:
            self.index.extend(index)
        self.logits.extend(tagging_logits)

    def get_results(self, output_dir: str):
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"tagging_logits_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "tagging_logits.json")

        if len(self.index):
            assert len(self.index) == len(self.logits)
            predictions = {idx: pred for idx, pred in zip(self.index, self.logits)}
            predictions = sorted(predictions.items(), key=lambda x: x[0])
            predictions = [pred[1] for pred in predictions]
            json.dump(predictions, open(output_file, "w"))
        else:
            json.dump(self.logits, open(output_file, "w"))

        return {}, self.logits


def answer_clean(pred_seq: str, reverse: bool = False, answer_trigger: str = "The answer is"):
    if answer_trigger:
        pred_seq = pred_seq.split(answer_trigger)[1]
    # pred = re.findall(r'A|B|C|D|E', pred_seq)
    pred = re.findall(r'A|B|C|D', pred_seq)
    if len(pred) == 0:
        return ""
    if reverse:
        return pred[-1]
    return pred[0]


class GeneratorPredictor(DistGatherMixin):
    def __init__(self, reverse: bool = False, answer_trigger: str = "The answer is"):
        self.predictions = []
        self.reverse = reverse
        self.answer_trigger = answer_trigger

    def __call__(self, meta_data: Union[List[Dict[str, Any]], Dict[str, Any]], batch_model_outputs, ddp: bool = False):
        labels = meta_data["label"]
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        prompt_index = meta_data["prompt_index"]
        if isinstance(meta_data["index"], torch.Tensor):
            index = meta_data["index"].tolist()
        else:
            index = meta_data["index"]
        inputs = meta_data["input"]

        pred_seq = batch_model_outputs["generated_seq"]
        assert len(labels) == len(prompt_index) == len(index) == len(inputs), (len(labels), len(prompt_index), len(index), len(inputs))
        if len(pred_seq) == len(labels):
            pass
        elif len(pred_seq) % len(labels) == 0:
            mod = len(pred_seq) // len(labels)
            pred_seq = [pred_seq[i * mod: (i + 1) * mod] for i in range(len(labels))]
        else:
            raise ValueError((len(pred_seq), len(labels)))

        predictions = [
            {
                "label": label,
                "index": idx,
                "prompt_index": prompt_idx,
                "output": res,
                "cleaned_output": [answer_clean(item, self.reverse, self.answer_trigger) for item in res]
                if isinstance(res, list) else answer_clean(res, self.reverse, self.answer_trigger),
                "input": src,
            } for label, idx, prompt_idx, res, src in zip(labels, index, prompt_index, pred_seq, inputs)
        ]

        if ddp:
            gather_res = self.gather_object(predictions)
            if dist.get_rank() == 0:
                tmp = []
                for item in gather_res:
                    tmp.extend(item)
                predictions = tmp

        self.predictions.extend(predictions)

    def get_results(self, output_dir: str):
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"decode_results_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "decode_results.json")

        json.dump(self.predictions, open(output_file, "w"), indent=2)
        self.predictions = sorted(self.predictions, key=lambda x: x["index"])

        correct = 0
        existing_ids = set()
        npy_outputs = []
        for pred in self.predictions:
            if pred["index"] in existing_ids:
                continue
            existing_ids.add(pred["index"])
            if pred["label"] == pred["cleaned_output"]:
                correct += 1

            if isinstance(pred["cleaned_output"], list) and len(pred["cleaned_output"]) > 0:
                npy_outputs.append(ord(pred["cleaned_output"][0]) - ord("A"))
            elif pred["cleaned_output"]:
                npy_outputs.append(ord(pred["cleaned_output"]) - ord("A"))
            else:
                npy_outputs.append(0)

        metrics = {"acc": correct / len(existing_ids)}

        if not dist.is_initialized() or dist.get_rank() == 0:
            np.save(os.path.join(output_dir, "decode_results.npy"), np.array(npy_outputs))
            json.dump(metrics, open(os.path.join(output_dir, "metrics.json"), "w"))
        assert len(npy_outputs) == len(existing_ids), (len(npy_outputs), len(self.predictions), len(existing_ids))
        return metrics, self.predictions


class GeneratorPredictorV2(DistGatherMixin):
    def __init__(self, answer_cleaner: Callable):
        self.predictions = []
        self.answer_cleaner = answer_cleaner

    def __call__(self, meta_data: Union[List[Dict[str, Any]], Dict[str, Any]], batch_model_outputs, ddp: bool = False):
        labels = meta_data["label"]
        prompt_index = meta_data["prompt_index"]
        index = meta_data["index"].tolist()
        inputs = meta_data["input"]

        pred_seq = batch_model_outputs["generated_seq"]
        assert len(labels) == len(prompt_index) == len(index) == len(inputs), (len(labels), len(prompt_index), len(index), len(inputs))
        if len(pred_seq) == len(labels):
            pass
        elif len(pred_seq) % len(labels) == 0:
            pass
        else:
            raise ValueError((len(pred_seq), len(labels)))

        predictions = [
            {
                "label": label,
                "index": idx,
                "prompt_index": prompt_idx,
                "output": res,
                "cleaned_output": self.answer_cleaner(res, src),
                "input": src,
            } for label, idx, prompt_idx, res, src in zip(labels, index, prompt_index, pred_seq, inputs)
        ]

        if ddp:
            gather_res = self.gather_object(predictions)
            if dist.get_rank() == 0:
                tmp = []
                for item in gather_res:
                    tmp.extend(item)
                predictions = tmp

        self.predictions.extend(predictions)

    def get_results(self, output_dir: str):
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"decode_results_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "decode_results.json")

        json.dump(self.predictions, open(output_file, "w"))
        self.predictions = sorted(self.predictions, key=lambda x: x["index"])

        correct = 0
        existing_ids = set()
        npy_outputs = []
        for pred in self.predictions:
            if pred["index"] in existing_ids:
                continue
            existing_ids.add(pred["index"])
            if pred["label"] == pred["cleaned_output"]:
                correct += 1
            if pred["cleaned_output"] and len(pred["cleaned_output"]) == 1:
                npy_outputs.append(ord(pred["cleaned_output"]) - ord("A"))
            else:
                npy_outputs.append(0)

        metrics = {"acc": correct / len(existing_ids)}

        if not dist.is_initialized() or dist.get_rank() == 0:
            np.save(os.path.join(output_dir, "decode_results.npy"), np.array(npy_outputs))
            json.dump(metrics, open(os.path.join(output_dir, "metrics.json"), "w"))
        assert len(npy_outputs) == len(existing_ids), (len(npy_outputs), len(self.predictions), len(existing_ids))
        return metrics, self.predictions


class BBHPredictor(DistGatherMixin):
    def __init__(self, answer_trigger: Callable, matcher: Callable):
        self.predictions = []
        self.answer_trigger = answer_trigger
        self.matcher = matcher

    def __call__(self, meta_data: Union[List[Dict[str, Any]], Dict[str, Any]], batch_model_outputs, ddp: bool = False):
        labels = meta_data["target"]
        index = meta_data["index"]
        inputs = meta_data["input"]
        modes = meta_data["task"]

        pred_seq = batch_model_outputs["generated_seq"]
        assert len(labels) == len(modes) == len(index) == len(inputs), (len(labels), len(modes), len(index), len(inputs))
        if len(pred_seq) == len(labels):
            pass
        elif len(pred_seq) % len(labels) == 0:
            pass
        else:
            raise ValueError((len(pred_seq), len(labels)))

        predictions = [
            {
                "label": label,
                "index": idx,
                "task": task,
                "output": res,
                "cleaned_output": self.matcher(self.answer_trigger(res), task),
                "input": src,
            } for label, idx, task, res, src in zip(labels, index, modes, pred_seq, inputs)
        ]

        if ddp:
            gather_res = self.gather_object(predictions)
            if dist.get_rank() == 0:
                tmp = []
                for item in gather_res:
                    tmp.extend(item)
                predictions = tmp

        self.predictions.extend(predictions)

    def get_results(self, output_dir: str):
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"decode_results_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "decode_results.json")

        predictions = collections.defaultdict(list)
        for pred in self.predictions:
            predictions[pred["task"]].append(pred)
        predictions = {k: sorted(v, key=lambda x: x["index"]) for k, v in predictions.items()}
        json.dump(predictions, open(output_file, "w"), indent=2)

        metrics = {}
        total_correct = 0
        total_samples = 0
        for task, task_predictions in predictions.items():
            correct = 0
            existing_ids = set()
            # npy_outputs = []
            for pred in task_predictions:
                if pred["index"] in existing_ids:
                    continue
                existing_ids.add(pred["index"])
                if task == "word_sorting":
                    cleaned_output = pred["cleaned_output"].replace(".", "")
                    for i in range(10):
                        cleaned_output = cleaned_output.replace(str(i), "")
                    cleaned_output = " ".join(cleaned_output.split())
                    label = " ".join(pred["label"].split())
                    if cleaned_output == label:
                        correct += 1
                    continue

                if pred["label"].strip().lower() == pred["cleaned_output"].replace(".", "").strip().lower():
                    correct += 1

            metrics[task] = {"acc": correct / len(existing_ids)}
            total_correct += correct
            total_samples += len(existing_ids)

        metrics["overall"] = {"acc": total_correct / total_samples}
        # computed unweighted accuracy
        sum_acc = 0
        for task in predictions.keys():
            sum_acc += metrics[task]["acc"]
        metrics["unweighted_acc"] = {"acc": sum_acc / len(predictions.keys())}

        if not dist.is_initialized() or dist.get_rank() == 0:
            json.dump(metrics, open(os.path.join(output_dir, "metrics.json"), "w"))
        return metrics, self.predictions


class MultiChoiceMetricSaver(DistGatherMixin):
    def __init__(self, ):
        self.predictions = []

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        if isinstance(meta_data["index"], torch.Tensor):
            index = meta_data["index"].float().tolist()
        else:
            index = meta_data["index"]
        labels = meta_data["label"].tolist()
        inputs = meta_data["input"]

        logits = batch_model_outputs["logits"].detach().float()
        if len(logits.size()) == 1:
            logits = logits.reshape(len(labels), -1)
        _, pred = logits.max(dim=-1)
        pred = pred.tolist()

        if ddp:
            obj = [pred, index, labels, inputs]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                pred = []
                index = []
                labels = []
                inputs = []
                for item in gather_res:
                    pred.extend(item[0])
                    index.extend(item[1])
                    labels.extend(item[2])
                    inputs.extend(item[3])

        self.predictions.extend([{
            "index": idx,
            "pred": p,
            "label": t,
            "input": src,
        } for idx, p, t, src in zip(index, pred, labels, inputs)])

    def get_results(self, output_dir: str):
        # output_file = os.path.join(output_dir, "eval_predictions.npy")
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.json")

        self.predictions = sorted(self.predictions, key=lambda x: x["index"])

        correct = 0
        existing_ids = set()
        npy_outputs = []
        for pred in self.predictions:
            if pred["index"] in existing_ids:
                continue
            existing_ids.add(pred["index"])
            if pred["pred"] == pred["label"]:
                correct += 1
            npy_outputs.append(pred["pred"])

        metrics = {"acc": round(correct / len(existing_ids), 4)}

        if not dist.is_initialized() or dist.get_rank() == 0:
            np.save(os.path.join(output_dir, "decode_results.npy"), np.array(npy_outputs))
            json.dump(metrics, open(os.path.join(output_dir, "metrics.json"), "w"))
            json.dump(self.predictions, open(output_file, "w"))
        assert len(npy_outputs) == len(existing_ids), (len(npy_outputs), len(self.predictions), len(existing_ids))
        return metrics, self.predictions


class MultiChoiceMetricLogitsSaver(DistGatherMixin):
    def __init__(self, ):
        self.predictions = []

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        if isinstance(meta_data["index"], torch.Tensor):
            index = meta_data["index"].float().tolist()
        else:
            index = meta_data["index"]
        labels = meta_data["label"].tolist()
        inputs = meta_data["input"]

        logits = batch_model_outputs["logits"].detach().float()
        if len(logits.size()) == 1:
            logits = logits.reshape(len(labels), -1)
        _, pred = logits.max(dim=-1)
        pred = pred.tolist()

        logits = logits.tolist()
        if ddp:
            obj = [pred, index, labels, inputs, logits]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                pred = []
                index = []
                labels = []
                inputs = []
                logits = []
                for item in gather_res:
                    pred.extend(item[0])
                    index.extend(item[1])
                    labels.extend(item[2])
                    inputs.extend(item[3])
                    logits.extend(item[4])

        self.predictions.extend([{
            "index": idx,
            "pred": p,
            "label": t,
            "input": src,
            "logits": logit,
        } for idx, p, t, src, logit in zip(index, pred, labels, inputs, logits)])

    def get_results(self, output_dir: str):
        # output_file = os.path.join(output_dir, "eval_predictions.npy")
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.json")

        self.predictions = sorted(self.predictions, key=lambda x: x["index"])

        correct = 0
        existing_ids = set()
        npy_outputs = []
        for pred in self.predictions:
            if pred["index"] in existing_ids:
                continue
            existing_ids.add(pred["index"])
            if pred["pred"] == pred["label"]:
                correct += 1
            npy_outputs.append(pred["pred"])

        metrics = {"acc": round(correct / len(existing_ids), 4)}

        if not dist.is_initialized() or dist.get_rank() == 0:
            np.save(os.path.join(output_dir, "decode_results.npy"), np.array(npy_outputs))
            json.dump(metrics, open(os.path.join(output_dir, "metrics.json"), "w"))
            json.dump(self.predictions, open(output_file, "w"))
        assert len(npy_outputs) == len(existing_ids), (len(npy_outputs), len(self.predictions), len(existing_ids))
        return metrics, self.predictions


class MultiChoiceMetricFlatSaver(DistGatherMixin):
    def __init__(self, ):
        self.raw_predictions = []
        self.predictions = []

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        index = meta_data["index"]
        labels = meta_data["label"].tolist()
        inputs = meta_data["input"]

        logits = batch_model_outputs["logits"].detach().float().tolist()

        if ddp:
            obj = [logits, index, labels, inputs]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                logits = []
                index = []
                labels = []
                inputs = []
                for item in gather_res:
                    logits.extend(item[0])
                    index.extend(item[1])
                    labels.extend(item[2])
                    inputs.extend(item[3])

        self.raw_predictions.extend([{
            "index": idx,
            "logit": l,
            "label": t,
            "input": src,
        } for idx, l, t, src in zip(index, logits, labels, inputs)])

    def get_results(self, output_dir: str):
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.json")

        if dist.is_initialized() and dist.get_rank() != 0:
            return {}, []

        sample_groups = collections.defaultdict(list)
        for pred in self.raw_predictions:
            sample_id, op_id = pred["index"].split("_")
            sample_groups[sample_id].append((int(op_id), pred))

        for sample_id, sample_preds in sample_groups.items():
            sample_preds = sorted(sample_preds, key=lambda x: x[0])
            sample_logits = []
            op_id_set = set()
            for op_id, pred in sample_preds:
                if op_id in op_id_set:
                    continue
                op_id_set.add(op_id)
                sample_logits.append(pred["logit"])
            sample_pred = torch.tensor(sample_logits).argmax().item()
            self.predictions.append({
                "index": sample_id,
                "pred": sample_pred,
                "label": sample_preds[0][1]["label"],
                "input": sample_preds[0][1]["input"],
            })

        self.predictions = sorted(self.predictions, key=lambda x: int(x["index"]))

        correct = 0
        existing_ids = set()
        npy_outputs = []
        for pred in self.predictions:
            if pred["index"] in existing_ids:
                continue
            existing_ids.add(pred["index"])
            if pred["pred"] == pred["label"]:
                correct += 1
            npy_outputs.append(pred["pred"])

        metrics = {"acc": round(correct / len(existing_ids), 3)}

        if not dist.is_initialized() or dist.get_rank() == 0:
            np.save(os.path.join(output_dir, "decode_results.npy"), np.array(npy_outputs))
            json.dump(metrics, open(os.path.join(output_dir, "metrics.json"), "w"))
            json.dump(self.predictions, open(output_file, "w"))
        assert len(npy_outputs) == len(existing_ids), (len(npy_outputs), len(self.predictions), len(existing_ids))
        return metrics, self.predictions


class LogitsSaver(MultiChoiceMetricFlatSaver):
    def get_results(self, output_dir: str):
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.json")

        if dist.is_initialized() and dist.get_rank() != 0:
            return {}, []

        if not dist.is_initialized() or dist.get_rank() == 0:
            json.dump(self.raw_predictions, open(output_file, "w"))
        return {}, self.raw_predictions


class MultiChoiceMetricSelectionSaver(MultiChoiceMetricSaver):
    def __init__(self, tokenizer: Union[str, PreTrainedTokenizer]):
        super().__init__()
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)

        self.tokenizer = tokenizer

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        index = meta_data["index"].float().tolist()
        labels = meta_data["label"].tolist()
        inputs = meta_data["input"]
        batch_outputs = meta_data["output"]

        # Should be non-shifted logits, e.g., original outputs from huggingface models.
        logits = batch_model_outputs["logits"].detach().float()
        assert logits.dim() == 3, logits.dim()  # [batch_size, seq_len, vocab_size]
        pred = []
        if isinstance(batch_outputs[0][0], list) or isinstance(batch_outputs[0][0], tuple):
            for i, outputs in enumerate(batch_outputs):
                output_probs = []
                for output in outputs:
                    target_token_id = self.tokenizer.convert_tokens_to_ids(output[0])
                    output_probs.append(logits[i, -1, target_token_id].item())
                output_probs = torch.tensor(output_probs)
                tgt_id = output_probs.argmax().item()
                pred.append(outputs[tgt_id][1])
                assert isinstance(pred[-1], int)
        else:
            assert isinstance(batch_outputs[0][0], str)
            for i, outputs in enumerate(batch_outputs):
                output_probs = []
                for output in outputs:
                    target_token_id = self.tokenizer.convert_tokens_to_ids(output)
                    output_probs.append(logits[i, -1, target_token_id].item())
                output_probs = torch.tensor(output_probs)
                pred.append(output_probs.argmax().item())

        if ddp:
            obj = [pred, index, labels, inputs]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                pred = []
                index = []
                labels = []
                inputs = []
                for item in gather_res:
                    pred.extend(item[0])
                    index.extend(item[1])
                    labels.extend(item[2])
                    inputs.extend(item[3])

        self.predictions.extend([{
            "index": idx,
            "pred": p,
            "label": t,
            "input": src,
        } for idx, p, t, src in zip(index, pred, labels, inputs)])


class RewardSaver(DistGatherMixin):
    def __init__(self, tokenizer: Union[str, PreTrainedTokenizer]):
        super().__init__()
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)

        self.tokenizer = tokenizer
        self.predictions = []

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        index = meta_data["index"]
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        inputs = meta_data["input"]
        labels = meta_data["label"]
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        rewards = batch_model_outputs["logits"].detach().float().tolist()

        if ddp:
            obj = [rewards, index, labels, inputs]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                rewards = []
                index = []
                labels = []
                inputs = []
                for item in gather_res:
                    rewards.extend(item[0])
                    index.extend(item[1])
                    labels.extend(item[2])
                    inputs.extend(item[3])

        self.predictions.extend([{
            "index": idx,
            "pred": p,
            "label": t,
            "input": src,
        } for idx, p, t, src in zip(index, rewards, labels, inputs)])

    def get_results(self, output_dir: str):
        # output_file = os.path.join(output_dir, "eval_predictions.npy")
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.json")

        self.predictions = sorted(self.predictions, key=lambda x: x["index"])

        if not dist.is_initialized() or dist.get_rank() == 0:
            json.dump(self.predictions, open(output_file, "w"))
        return {}, self.predictions


class VLLMSaver(DistGatherMixin):
    def __init__(self):
        self.predictions = []

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: List, ddp: bool = False):
        index = meta_data["index"]
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        prompt = meta_data["prompt"]
        label = meta_data["label"]
        if isinstance(label, torch.Tensor):
            label = label.tolist()

        outputs = [output.outputs[0].text for output in batch_model_outputs]

        if ddp:
            obj = [outputs, index, label, prompt]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                outputs = []
                index = []
                label = []
                prompt = []
                for item in gather_res:
                    outputs.extend(item[0])
                    index.extend(item[1])
                    label.extend(item[2])
                    prompt.extend(item[3])

        self.predictions.extend([{
            "index": idx,
            "pred": p,
            "label": t,
            "prompt": src,
        } for idx, p, t, src in zip(index, outputs, label, prompt)])

    def get_results(self, output_dir: str):
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.json")

        self.predictions = sorted(self.predictions, key=lambda x: x["index"])

        if not dist.is_initialized() or dist.get_rank() == 0:
            json.dump(self.predictions, open(output_file, "w"))
        return {}, self.predictions
