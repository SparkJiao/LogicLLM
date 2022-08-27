import copy
from typing import List, Set, Union, Dict, Tuple

from transformers import PreTrainedTokenizer
from transformers import RobertaTokenizer, RobertaTokenizerFast, AlbertTokenizer, AlbertTokenizerFast, DebertaTokenizer, \
    DebertaTokenizerFast, DebertaV2Tokenizer
from transformers.models.bert.tokenization_bert import whitespace_tokenize

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


def tokenizer_get_name(_tokenizer: PreTrainedTokenizer):
    tokenizer_name = _tokenizer.__class__.__name__
    tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
    tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()
    return tokenizer_name


def get_sep_tokens(_tokenizer: PreTrainedTokenizer):
    return [_tokenizer.sep_token] * (_tokenizer.max_len_single_sentence - _tokenizer.max_len_sentences_pair)


def find_span(text: str, span: str, start: int = 0):
    pos = text.find(span, start)
    if pos == -1:
        return []
    _e = pos + len(span)
    return [(pos, _e)] + find_span(text, span, start=_e)


def span_chunk(text: str, span_ls: List[str], space_tokenize: bool = False) -> Tuple[List[str], List[int]]:
    pos_ls = []
    for span in span_ls:
        span_pos_ls = find_span(text, span)
        pos_ls.extend(span_pos_ls)
    pos_ls = sorted(pos_ls, key=lambda x: x[0])

    text_spans = []
    indicate_mask = []
    last_e = 0
    for s, e in pos_ls:
        if last_e > s:
            logger.warning(f"Overlapped span: {text_spans[-1]}\t{text[s: e]}\t{text}")
            print(f"Overlapped span: {text_spans[-1]}\t{text[s: e]}\t{text}")
            continue
        if s > last_e:
            if space_tokenize:
                text_spans.extend(whitespace_tokenize(text[last_e: s]))
            else:
                tmp = text[last_e: s]
                if tmp.strip():
                    text_spans.append(tmp)
        indicate_mask = indicate_mask + [0] * (len(text_spans) - len(indicate_mask))

        text_spans.append(text[s: e])
        indicate_mask = indicate_mask + [1] * (len(text_spans) - len(indicate_mask))
        last_e = e

    rest = text[last_e:].strip()
    if rest:
        if space_tokenize:
            text_spans.extend(whitespace_tokenize(rest))
        else:
            text_spans.append(rest)

    _recovered_text = " ".join(text_spans)
    if _recovered_text != text:
        logger.warning(f"In consistent text during chunk:\n{_recovered_text}\n{text}")
        print(f"In consistent text during chunk:\n{_recovered_text}\n{text}")

    return text_spans, indicate_mask


def get_unused_tokens(_tokenizer: PreTrainedTokenizer, token_num: int = 4):
    if isinstance(_tokenizer, RobertaTokenizer) or isinstance(_tokenizer, RobertaTokenizerFast):
        _unused_token = "<unused{}>"
        _unused_tokens = []
        for i in range(token_num):
            _unused_tokens.append(_unused_token.format(str(i)))
        _tokenizer.add_tokens(_unused_tokens)
        return _unused_tokens
    elif isinstance(_tokenizer, AlbertTokenizer) or isinstance(_tokenizer, AlbertTokenizerFast):
        _unused_token = "[unused{}]"
        _unused_tokens = []
        for i in range(token_num):
            _unused_tokens.append(_unused_token.format(str(i)))
        _tokenizer.add_tokens(_unused_tokens)
        return _unused_tokens
    elif any([isinstance(_tokenizer, x) for x in [DebertaTokenizer, DebertaTokenizerFast, DebertaV2Tokenizer]]):
        _unused_token = "[unused{}]"
        _unused_tokens = []
        for i in range(token_num):
            _unused_tokens.append(_unused_token.format(str(i)))
        _tokenizer.add_tokens(_unused_tokens)
        return _unused_tokens


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


def dfs_enumerate_all_assign(keys: List[str], values: List[str], relation: str, res: List[str], assign: str,
                             key_vis: Set):
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


def dfs_load_assignment(assignment_list, res: List[Tuple[str, str]], cur_assign: str):
    for assignment in assignment_list:
        if assignment['flag'] is False:
            continue
        if assignment['flag'] is None:
            res.append((cur_assign + ' ' + assignment['deduction'], assignment['id']))
        elif assignment['flag'] is True:
            dfs_load_assignment(assignment['assignment'], res, cur_assign + ' ' + assignment['deduction'])
        else:
            raise ValueError('Unknown flag: {}'.format(assignment['flag']))
