import copy
from typing import List, Set, Union, Dict

from transformers import PreTrainedTokenizer


def tokenizer_get_name(_tokenizer: PreTrainedTokenizer):
    tokenizer_name = _tokenizer.__class__.__name__
    tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
    tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()
    return tokenizer_name


def get_sep_tokens(_tokenizer: PreTrainedTokenizer):
    return [_tokenizer.sep_token] * (_tokenizer.max_len_single_sentence - _tokenizer.max_len_sentences_pair)


def dfs(src: List[int], vis: Set, state: List[int], ans: List[List[int]]):
    if len(state) == len(src):
        if not all(a == b for a, b in zip(src, state)):
            ans.append(state)

    for x in src:
        if x not in vis:
            new_vis = copy.deepcopy(vis)
            new_vis.add(x)
            new_state = copy.deepcopy(state)
            new_state.append(x)
            dfs(src, new_vis, new_state, ans)


def get_all_permutation(array: List[int]):
    res = []
    dfs(array, set(), list(), res)
    for state in res:
        assert not all(a == b for a, b in zip(state, array))
    return res


def recursive_find_path(node: Union[List, Dict, str], outputs: List[List[str]], res: List[str]):
    if isinstance(node, str):
        outputs.append(res + [node])
        return

    if isinstance(node, list):
        for x in node:
            recursive_find_path(x, outputs, res)
    elif isinstance(node, dict):
        for key, value in node.items():
            recursive_find_path(value, outputs, res + [key])
    else:
        raise ValueError('Unknown type: {}'.format(type(node)))


def recursive_bfs(deduction: Union[List, Dict]):
    res = ''

    queue = [deduction]
    while queue:
        node = queue.pop(0)
        if isinstance(node, str):
            res = res + ' ' + node
        elif isinstance(node, list):
            queue.extend(node)
        elif isinstance(node, dict):
            for key, value in node.items():
                queue.append(value)
                res = res + ' ' + key
        else:
            raise ValueError('Unknown type: {}'.format(type(node)))

    return res.strip()


def dfs_enumerate_all_assign(keys: List[str], values: List[str], relation: str, res: List[str], assign: str, key_vis: Set):
    if len(key_vis) == 0:
        res.append(assign)

    for key_id in key_vis:
        new_key_vis = copy.deepcopy(key_vis)
        new_key_vis.remove(key_id)
        for value in values:
            if value in keys[key_id]:
                continue
            new_assign = assign + ' ' + keys[key_id] + ' ' + relation + ' ' + value + '.'
            dfs_enumerate_all_assign(keys, values, relation, res, new_assign, new_key_vis)
