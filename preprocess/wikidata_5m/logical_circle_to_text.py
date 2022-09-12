import argparse
import collections
import json
import os
import sys
import random
from functools import partial
from multiprocessing import Pool
from typing import List, Dict, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers.models.bert.tokenization_bert import whitespace_tokenize
from nltk import word_tokenize
import glob


sys.path.append("../../")

from data.data_utils import span_chunk, tokenizer_get_name, find_span, span_chunk_simple

_tokenizer: PreTrainedTokenizer
_id2ent: Dict[str, List[str]]
_id2rel: Dict[str, List[str]]
_triplet2sent: Dict[str, List[str]]
_edge2rel: Dict[str, List[str]]


def init(tokenizer: PreTrainedTokenizer,
         id2ent, id2rel, triplet2sent, edge2rel):
    global _tokenizer
    global _id2ent
    global _id2rel
    global _triplet2sent
    global _edge2rel
    _tokenizer = tokenizer
    _id2ent = id2ent
    _id2rel = id2rel
    _triplet2sent = triplet2sent
    _edge2rel = edge2rel


def load_triplet2sent(triplet2sent: str):
    triplet_sent_ls = json.load(open(triplet2sent))

    triplet2sent = collections.defaultdict(list)
    for item in triplet_sent_ls:
        s, rel, t, corpus = item["s"], item["rel"], item["t"], item["corpus"]
        key = "\t".join([s, rel, t])
        triplet2sent[key].extend(corpus)

    return triplet2sent


def triplet2texts(s, rel, t, triplet2sent, id2ent, id2rel) -> Union[List[Dict[str, str]], None]:
    key = "\t".join([s, rel, t])
    texts = []
    if key in triplet2sent:
        for item in triplet2sent[key]:
            s_alias = item["s"]
            t_alias = item["t"]
            text = item["text"]
            assert s_alias in text, (s_alias, t_alias, text)
            assert t_alias in text, (s_alias, t_alias, text)  # FIXED: Found an assertion error here.
            # if s_alias not in text:
            #     continue
            # if t_alias not in text:
            #     continue
            if len(set(word_tokenize(s_alias)) & set(
                    word_tokenize(
                        t_alias))):  # FIXED in `align_triplet_text.py`: This case should be removed during text-triplet aligning.
                continue
            texts.append({
                "s": s,
                "t": t,
                "rel": rel,
                "s_alias": s_alias,
                "t_alias": t_alias,
                "rel_alias": "",
                "text": text,
            })

    s_alias = random.choice(id2ent[s])
    t_alias = random.choice(id2ent[t])
    # FIXED: KeyError here. Should be checked in `align_triplet_text.py`.
    # if rel not in id2rel:
    #     if texts:
    #         return texts
    #     return None
    rel_alias = random.choice(id2rel[rel])

    text = ' '.join([s_alias, rel_alias, t_alias]) + '.'
    texts.append({
        "s": s,
        "t": t,
        "rel": rel,
        "s_alias": s_alias,
        "t_alias": t_alias,
        "rel_alias": rel_alias,
        "text": text
    })

    return texts


def text2triplet(triplet: str):
    return triplet.split("\t")


def doc_span_chunk(sentences: List[Dict[str, str]]):
    text = []
    spans = []
    indicate_mask = []
    for sent_dict in sentences:
        tgt_spans = [sent_dict["s_alias"], sent_dict["t_alias"]]
        if sent_dict["rel_alias"]:
            # Usually multiple words. So, pass.
            pass
        sent_spans, sent_indicate_mask = span_chunk(sent_dict["text"],
                                                    tgt_spans,
                                                    space_tokenize=True)

        spans.extend(sent_spans)
        indicate_mask.extend(sent_indicate_mask)
        text.append(sent_dict["text"])

        assert len(spans) == len(indicate_mask)

    text = " ".join(text)
    return text, spans, indicate_mask


def circle2text_mlm(path: List[Tuple[str, str, str]],
                    # id2ent: Dict[str, List[str]],
                    # id2rel: Dict[str, List[str]],
                    # triplet2sent: Dict[str, List[str]],
                    # edge2rel: Dict[str, List[str]],
                    shuffle_sentence: bool = True):
    assert len(path) >= 2

    # Obtain the relation with the given entity pair
    s = path[0][0]
    t = path[-1][-1]
    key = f"{s}\t{t}"

    assert len(_edge2rel[key])
    rel = random.choice(_edge2rel[key])

    # Symbols to text
    context = [triplet2texts(*_triplet, _triplet2sent, _id2ent, _id2rel) for _triplet in path]
    anchor = triplet2texts(s, rel, t, _triplet2sent, _id2ent, _id2rel)

    if any(sent_dict is None for sent_dict in context + [anchor]):
        return None

    # Sample a text group
    context = [random.choice(sent_dict) for sent_dict in context]
    anchor = random.choice(anchor)

    sentences = context + [anchor]
    if shuffle_sentence:
        random.shuffle(sentences)

    # Annotate the span (of entities) and obtain the indicating mask (for which word belongs to an entity)
    text, spans, indicate_mask = doc_span_chunk(sentences)
    tokens = _tokenizer.tokenize(text)

    """
                    ================== MLM =====================
    - Whole word mask is needed to avoid information leakage (right?)
    - For each triplet, different processing procedure for entity and relation (may be implicit) may be needed.
     
    - For MLM, we can only pre-process the token mask for each word and annotate the scope of entities,
      so that the we can perform online dynamic whole word masking during training.
    
    - The significance of masking entity and relation:
        - For entity, entity masking requires the model to predict correct subject and object within a triplet
          based on the context understanding.
          If there are more than one sentence involving entity masking, it will also propose the challenge about
          intermediate relations within a reasoning path.
        - For relation, relation masking makes the model learn the correct mapping (computation) in reasoning.
    """

    # FIXED: This method may generate inconsistency since different tokenization methods are used.
    #   Instead, we use another method where the words are matched to the subwords sequence one-by-one.
    #   And the unmatched words are processed separately.
    #   See following for details.
    # flag, recovered_text, tokens, _, token2word_index = generate_whole_word_index_mapping(text,
    #                                                                                       _tokenizer,
    #                                                                                       spans)
    # return flag, text, tokens, spans, token2word_index, indicate_mask

    # `token2word_index`: List[int], token num
    # `extended_word_cnt`: int, extended word num
    # `extended_indicate_mask`: List[int], `0` for non-entity word and `1` for entity word, extended word num
    res = annotate_span(tokens, spans, indicate_mask, _tokenizer)

    if res is None:
        return None

    flag, token2word_index, extended_word_cnt, extended_indicate_mask = res

    return flag, text, spans, token2word_index, extended_indicate_mask


def circle2text_mlm_simple(path: List[str]):
    s = path[0].split("\t")[0]
    t = path[-1].split("\t")[-1]
    key = f"{s}\t{t}"

    assert len(_edge2rel[key])
    rel = random.choice(_edge2rel[key])

    context = [triplet2texts(*text2triplet(_triplet), _triplet2sent, _id2ent, _id2rel) for _triplet in path]
    anchor = triplet2texts(s, rel, t, _triplet2sent, _id2ent, _id2rel)

    if any(sent_dict is None for sent_dict in context + [anchor]):
        return None

    # Sample a text group
    context = [random.choice(sent_dict) for sent_dict in context]
    anchor = random.choice(anchor)

    sentences = context + [anchor]
    random.shuffle(sentences)

    ent_spans = []
    text = []
    for sent_dict in sentences:
        ent_spans.extend([sent_dict["s_alias"], sent_dict["t_alias"]])
        text.append(sent_dict["text"])
    ent_spans = list(set(ent_spans))

    text = " ".join(text)
    normalized_text, token_spans = span_chunk_simple(text, ent_spans, _tokenizer)
    if normalized_text is None:
        return None

    return normalized_text, token_spans


def circle2text_seq2seq_v1(path: List[Tuple[str, str, str]],
                           context_len: int = 0):
    assert len(path) >= 2

    # Obtain the relation with the given entity pair
    s = path[0][0]
    t = path[-1][-1]
    key = f"{s}\t{t}"

    assert len(_edge2rel[key])
    rel = random.choice(_edge2rel[key])

    # Symbols to text
    context = [triplet2texts(*_triplet, _triplet2sent, _id2ent, _id2rel) for _triplet in path]
    anchor = triplet2texts(s, rel, t, _triplet2sent, _id2ent, _id2rel)

    if any(sent_dict is None for sent_dict in context + [anchor]):
        return None

    # Sample a text group
    context = [random.choice(sent_dict) for sent_dict in context]
    anchor = random.choice(anchor)

    random.shuffle(context)

    input_seq = [anchor] + context[:context_len]
    output_seq = context[context_len:]

    """
    ============================ Seq2Seq ================================
    
    For sequence-to-sequence task, currently, I think we only require indicating the entities (and relations)
    in output sequence, since we only compute the LM loss on the decoder side.
    """
    src_text = " ".join([sent_dict["text"] for sent_dict in input_seq])

    tgt_text, tgt_spans, tgt_indicate_mask = doc_span_chunk(output_seq)
    tgt_tokens = _tokenizer.tokenize(tgt_text)

    res = annotate_span(tgt_tokens, tgt_spans, tgt_indicate_mask, _tokenizer)
    if res is None:
        return None

    flag, token2word_index, extended_word_cnt, extended_indicate_mask = res

    return flag, src_text, tgt_spans, token2word_index, extended_indicate_mask


def circle2text_seq2seq_simple_v1(path: List[str], context_len: int = 0):
    assert len(path) >= 2

    # Obtain the relation with the given entity pair
    # s = path[0][0]
    # t = path[-1][-1]
    s = path[0].split("\t")[0]
    t = path[-1].split("\t")[-1]
    key = f"{s}\t{t}"

    assert len(_edge2rel[key])
    rel = random.choice(_edge2rel[key])

    # Symbols to text
    context = [triplet2texts(*text2triplet(_triplet), _triplet2sent, _id2ent, _id2rel) for _triplet in path]
    anchor = triplet2texts(s, rel, t, _triplet2sent, _id2ent, _id2rel)

    if any(sent_dict is None for sent_dict in context + [anchor]):
        return None

    # Sample a text group
    context = [random.choice(sent_dict) for sent_dict in context]
    anchor = random.choice(anchor)

    random.shuffle(context)

    input_seq = [anchor] + context[:context_len]
    output_seq = context[context_len:]

    src_text = " ".join([sent_dict["text"] for sent_dict in input_seq])
    tgt_text = " ".join([sent_dict["text"] for sent_dict in output_seq])

    return src_text, tgt_text


# TODO: Add sentence based text, named version v2.


def circle2text_seq2seq_entity_v1(path: List[str], context_len: int = 0, entity_prob: float = 0.7):
    assert len(path) >= 2

    # Obtain the relation with the given entity pair
    s = path[0].split("\t")[0]
    t = path[-1].split("\t")[-1]
    key = f"{s}\t{t}"

    assert len(_edge2rel[key])
    rel = random.choice(_edge2rel[key])

    # Symbols to text
    context = [triplet2texts(*text2triplet(_triplet), _triplet2sent, _id2ent, _id2rel) for _triplet in path]
    anchor = triplet2texts(s, rel, t, _triplet2sent, _id2ent, _id2rel)

    if any(sent_dict is None for sent_dict in context + [anchor]):
        return None

    # Sample a text group
    context = [random.choice(sent_dict) for sent_dict in context]
    anchor = random.choice(anchor)

    random.shuffle(context)

    input_seq = [anchor] + context[:context_len]
    output_seq = context[context_len:]

    src_text = " ".join([sent_dict["text"] for sent_dict in input_seq])
    tgt_text = " ".join([sent_dict["text"] for sent_dict in output_seq])

    kept_entities = []
    all_entities_tgt = set([sent_dict["s_alias"] for sent_dict in output_seq] + [
        sent_dict["t_alias"] for sent_dict in output_seq])
    for ent in all_entities_tgt:
        _r = random.random()
        if _r > entity_prob:
            kept_entities.append(ent)

    return src_text, tgt_text, kept_entities



def generate_whole_word_index_mapping(text: str, tokenizer: PreTrainedTokenizer, spans: List[str] = None):
    """

    Note: If `text` is indeed an input pair, the separator token, e.g., `[SEP]`, should be inserted into the text first.
    """
    if spans is None:
        words = whitespace_tokenize(text)
    else:
        # Using spans at the input to support span masking, e.g., treat an entity (phrase) as a single word.
        words = spans

    tokens = []
    # Used for `torch.scatter` method
    # i.e., `token_mask = torch.scatter(word_mask, dim=0, index=token2word_index)`,
    # where `word_mask` is a 0/1 tensor indicating whether a word should be masked.
    # word_mask: [word_num], token2word_index: [token_num]
    token2word_index = []
    for idx, word in enumerate(words):
        if idx > 0:
            sub_tokens = tokenizer.tokenize(' ' + word)
        else:
            sub_tokens = tokenizer.tokenize(word)

        tokens.extend(sub_tokens)
        token2word_index.extend([idx] * len(sub_tokens))

    # Consistency check
    target_tokens = tokenizer.tokenize(text)
    recovered_text = tokenizer.convert_tokens_to_string(tokens)
    flag = True
    if target_tokens != tokens:
        print("Warning: Inconsistent tokenization: ")
        print(f"Original:\t{target_tokens}")
        print(f"Pre-tokenized:\t{tokens}")
        flag = False
    if recovered_text != text:
        print("Warning: Inconsistent text: ")
        print(f"Original text:\t{text}")
        print(f"Recovered text:\t{recovered_text}")
        flag = False

    return flag, recovered_text, tokens, words, token2word_index


def annotate_span(subwords: List[str], words: List[str], indicate_mask: List[int], tokenizer: PreTrainedTokenizer):
    """
    The core idea:


    """
    subword_span_pos_ls = []
    last_e = 0

    def span_iterator(start, length):
        for a in range(start, length):
            for b in range(a + 1, length + 1):
                yield a, b

    unmatch = 0
    for word_i, word in enumerate(words):
        # for i in range(last_e, len(subwords)):
        #     for j in range(i + 1, len(subwords) + 1):
        find = False
        for i, j in span_iterator(last_e, len(subwords)):
            tmp = tokenizer.convert_tokens_to_string(subwords[i: j]).strip()
            if tmp == word:
                subword_span_pos_ls.append((i, j, word_i))
                last_e = j
                find = True
                break
        if not find:
            # last_e += 1
            unmatch += 1
    if unmatch / len(words) > 0.2:
        return None

    word_p = 0
    subword_p = 0
    extended_word_cnt = 0
    extended_indicate_mask = []
    token2word_index = []
    for span_s, span_e, word_idx in subword_span_pos_ls:
        # Ideally, the spans are continuous
        if span_s > subword_p:  # If the spans are not continuous:
            skipped_word_num = word_idx - word_p
            # skipped_span = (subword_p, span_s)
            skipped_len = span_s - subword_p

            if skipped_word_num == 1:  # If there are only one word not matched:
                # print("1111111111111111111")
                # print(words[word_p: word_idx])
                # Treat the skipped subwords as single word.
                token2word_index.extend([extended_word_cnt] * skipped_len)
                extended_indicate_mask.append(indicate_mask[word_p])
                extended_word_cnt += 1
                word_p += 1
                # print(words)
                # print(subwords)
            elif skipped_word_num > 1:  # If there are multiple words not matched:
                """
                A bad case here:
                The skipped subwords are ')' and ';', while the corresponding skipped subword is only single ');',
                but it seems do not matter?
                
                """
                # Treat the each skipped subword as single word.
                split_idx_seq = list(range(extended_word_cnt, extended_word_cnt + skipped_len))
                token2word_index.extend(split_idx_seq)
                extended_word_cnt += skipped_len

                # If any skipped word is entity, set all skipped subwords as entity.
                if any(symbol == 1 for symbol in indicate_mask[word_p: word_idx]):
                    extended_indicate_mask.extend([1] * skipped_len)
                else:
                    extended_indicate_mask.extend([0] * skipped_len)

                # print(words)
                # print(subwords)
                # print(f"Multiple skipped span warning: {words[word_p: word_idx]}\t"
                #       f"{tokenizer.convert_tokens_to_string(subwords[subword_p: span_s])}")

                word_p += skipped_word_num
            else:
                """
                The error case is usually caused by that multiple words cannot be find but the following words are
                matched by the frontier subwords.
                """
                # print(subword_span_pos_ls)
                # print(words)
                # print(subwords)
                # print(subwords[subword_p: span_s])
                # raise ValueError(skipped_word_num)
                # print(f"ValueError({skipped_word_num})")
                return None

        subword_p = span_e

        token2word_index.extend([extended_word_cnt] * (span_e - span_s))
        extended_indicate_mask.append(indicate_mask[word_idx])
        extended_word_cnt += 1
        # assert word_p == word_idx  # TODO: Found assertion error here.
        if word_p != word_idx:
            return None
        word_p += 1

    skipped_word_num = len(words) - word_p
    skipped_len = len(subwords) - subword_p
    if skipped_word_num == 1:  # Left one word exactly
        token2word_index.extend([extended_word_cnt] * skipped_len)
        extended_indicate_mask.append(indicate_mask[word_p])
        extended_word_cnt += 1
    elif skipped_word_num > 1:
        split_idx_seq = list(range(extended_word_cnt, extended_word_cnt + skipped_len))
        token2word_index.extend(split_idx_seq)

        if any(symbol == 1 for symbol in indicate_mask[word_p: len(words)]):
            extended_indicate_mask.extend([1] * skipped_len)
        else:
            extended_indicate_mask.extend([0] * skipped_len)

        extended_word_cnt += skipped_len

    assert len(token2word_index) == len(subwords)
    assert len(extended_indicate_mask) == extended_word_cnt  # TODO: Found assertion error here.
    assert all(idx < extended_word_cnt for idx in token2word_index)
    assert any(idx == extended_word_cnt - 1 for idx in token2word_index)
    assert word_p + skipped_word_num == len(words)

    if extended_word_cnt != len(words):
        flag = False
    else:
        flag = True

    return flag, token2word_index, extended_word_cnt, extended_indicate_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="mlm", help="Options: mlm, ctr, seq2seq")
    parser.add_argument("--path", type=str)
    parser.add_argument("--id2ent", type=str)
    parser.add_argument("--id2rel", type=str)
    parser.add_argument("--triplet2sent", type=str)
    parser.add_argument("--edge2rel", type=str)
    parser.add_argument("--context_len", type=int, default=0)
    parser.add_argument("--entity_prob", type=float, default=0.7)
    parser.add_argument("--tokenizer", type=str)
    # parser.add_argument("--shuffle_sentence", )
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--glob_mark", type=str, default=None)
    parser.add_argument("--dev_num", type=int, default=0)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    id2ent = json.load(open(args.id2ent))
    id2rel = json.load(open(args.id2rel))
    triplet2sent = load_triplet2sent(args.triplet2sent)
    edge2rel = json.load(open(args.edge2rel))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if os.path.exists(args.path):
        path_files = [args.path]
    else:
        path_files = sorted(list(glob.glob(args.path)))

    print(path_files)

    results = []
    output_file = None
    for path_file in path_files:
        print(f"Reading path file from {path_file}...")
        all_path = json.load(open(path_file))

        if args.debug:
            all_path = all_path[:100]

        print(len(all_path))

        if args.mode == "mlm":
            with Pool(args.num_workers, initializer=init,
                      initargs=(tokenizer, id2ent, id2rel, triplet2sent, edge2rel)) as p:
                _annotate = partial(circle2text_mlm)
                _results = list(tqdm(
                    p.imap(_annotate, all_path, chunksize=32),
                    total=len(all_path),
                    desc="constructing dataset",
                    dynamic_ncols=True,
                ))

            tokenizer_name = tokenizer_get_name(tokenizer)
            output_file = os.path.join(args.output_dir,
                                       f"logic_circle_data_v1_{tokenizer_name}_s{args.seed}_{args.mode}.json")
        elif args.mode == "mlm_simple":
            with Pool(args.num_workers, initializer=init,
                      initargs=(tokenizer, id2ent, id2rel, triplet2sent, edge2rel)) as p:
                _results = list(tqdm(
                    p.imap(circle2text_mlm_simple, all_path, chunksize=32),
                    total=len(all_path),
                    desc="constructing mlm simple dataset",
                    dynamic_ncols=True,
                ))
                tokenizer_name = tokenizer_get_name(tokenizer)
                file_name = path_file.split('/')[-1][:-5]
                output_file = os.path.join(args.output_dir,
                                           f"{file_name}_v1_s{args.seed}_mlm_simple_{tokenizer_name}.json")
        elif args.mode == "seq2seq":
            with Pool(args.num_workers, initializer=init,
                      initargs=(tokenizer, id2ent, id2rel, triplet2sent, edge2rel)) as p:
                _annotate = partial(circle2text_seq2seq_v1, context_len=args.context_len)
                _results = list(tqdm(
                    p.imap(_annotate, all_path, chunksize=32),
                    total=len(all_path),
                    desc="constructing seq2seq dataset",
                    dynamic_ncols=True,
                ))

            tokenizer_name = tokenizer_get_name(tokenizer)
            file_name = path_file.split('/')[-1][:-5]
            output_file = os.path.join(args.output_dir,
                                       f"{file_name}_v1_{tokenizer_name}_s{args.seed}_seq2seq_{args.context_len}.json")
        elif args.mode == "seq2seq_simple":
            with Pool(args.num_workers, initializer=init,
                      initargs=(tokenizer, id2ent, id2rel, triplet2sent, edge2rel)) as p:
                _annotate = partial(circle2text_seq2seq_simple_v1, context_len=args.context_len)
                _results = list(tqdm(
                    p.imap(_annotate, all_path, chunksize=32),
                    total=len(all_path),
                    desc="constructing seq2seq dataset",
                    dynamic_ncols=True,
                ))

            # tokenizer_name = tokenizer_get_name(tokenizer)
            file_name = path_file.split('/')[-1][:-5]
            output_file = os.path.join(args.output_dir,
                                       f"{file_name}_v1_s{args.seed}_seq2seq_simple_"
                                       f"{args.context_len}.json")
        elif args.mode == "seq2seq_entity":
            with Pool(args.num_workers, initializer=init,
                      initargs=(tokenizer, id2ent, id2rel, triplet2sent, edge2rel)) as p:
                _annotate = partial(circle2text_seq2seq_entity_v1, context_len=args.context_len,
                                    entity_prob=args.entity_prob)
                _results = list(tqdm(
                    p.imap(_annotate, all_path, chunksize=32),
                    total=len(all_path),
                    desc="constructing seq2seq dataset",
                    dynamic_ncols=True,
                ))

            # tokenizer_name = tokenizer_get_name(tokenizer)
            file_name = path_file.split('/')[-1][:-5]
            output_file = os.path.join(args.output_dir,
                                       f"{file_name}_v1_s{args.seed}_seq2seq_entity_"
                                       f"{args.context_len}_{args.entity_prob}.json")
        else:
            raise NotImplementedError()

        results.extend([res for res in _results if res is not None])
        print(len(results))
        del _results

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=False)

    print(len(results))

    if args.glob_mark:
        output_file = output_file[:-5] + f"_{args.glob_mark}.json"

    if args.dev_num > 0:
        dev_index = set(random.sample(list(range(len(results))), args.dev_num))
        train = []
        dev = []
        for idx in range(len(results)):
            if idx in dev_index:
                dev.append(results[idx])
            else:
                train.append(results[idx])
        json.dump(train, open(output_file[:-5] + "_train.json", "w"))
        json.dump(dev, open(output_file[:-5] + "_dev.json", "w"))
    else:
        json.dump(results, open(output_file, 'w'))


if __name__ == '__main__':
    main()
