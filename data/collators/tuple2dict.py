import torch


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
            outputs['token_type_ids'] = token_type_ids

        return outputs
