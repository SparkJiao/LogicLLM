import torch


def stack_list(*tensor_list):
    res = []
    for tensors in tensor_list:
        res.append(torch.stack(tensors, dim=0))

    return res


class Tuple2DictCollator:
    def __call__(self, batch):
        if len(batch[0]) == 5:
            input_ids, attention_mask, token_type_ids, op_mask, labels = list(zip(*batch))
        elif len(batch[0]) == 4:
            input_ids, attention_mask, op_mask, labels = list(zip(*batch))
            token_type_ids = None
        else:
            raise RuntimeError(len(batch[0]))

        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        op_mask = torch.stack(op_mask, dim=0)
        labels = torch.stack(labels, dim=0)

        outputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'op_mask': op_mask,
            'labels': labels,
        }
        if token_type_ids is not None:
            outputs['token_type_ids'] = stack_list(token_type_ids)[0]

        return outputs


class NLITuple2DictCollator:
    def __call__(self, batch):
        if len(batch[0]) == 4:
            input_ids, attention_mask, token_type_ids, labels = list(zip(*batch))
        elif len(batch[0]) == 3:
            input_ids, attention_mask, labels = list(zip(*batch))
            token_type_ids = None
        else:
            raise RuntimeError(len(batch[0]))

        input_ids, attention_mask, labels = stack_list(input_ids, attention_mask, labels)

        outputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
        if token_type_ids is not None:
            outputs['token_type_ids'] = stack_list(token_type_ids)[0]

        return outputs
