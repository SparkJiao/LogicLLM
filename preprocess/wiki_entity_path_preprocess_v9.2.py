import collections
import copy
import glob
import json
import os
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from typing import Tuple, Dict, Set, List

from tqdm import tqdm

"""
Process the data from ERICA pretrain: ERICA: 
    Improving Entity and Relation Understanding for Pre-trained Language Models via Contrastive Learning

Core:
    Given a meta-path: <e_{a,i}, r_{i,j}, e_{a,j}>, ..., <e_{b,j}, r_{j,k}, e_{b,k}>,
    in which e_{i,j} represents the mention of the j-th entity in the i-th sentence,
    if there is a annotated relation between e_i and e_k, there is a logic path exists.
    We extract the path as a positive example.
    For negative samples, ...

Procedure:
    1.  枚举relation，判断是否存在一条非直接的path连接relation的头实体和尾实体。路径的游走要求：
        对于一对实体<e_i, e_j>:
        1.  存在relation连接<e_i, e_j>
        2.  <e_i, e_j>在同一个句子里。
        可选的，可以试试只用1或者只用2。 其中e_i和e_j都是可以跨句的（common mention）

Version 4.1 Update:
    1.  不再枚举relation，枚举<e_i, e_j>要求 e_i 和 e_j 只要出现在同一个句子里即可。

Version 5.0 Update:
    1.  实体保存形式从Tuple改成Dict方便对实体整体进行sample
    2.  修改bug：``ent2sent``和``sent2ent``的从list改成用set保存
    3.  修改bug: 上个版本中没有区分同一个实体在同一个句子里的不同位置上mention，
        这可能会导致对于某个需要被替换的实体只替换了其中一个mention，而另一个mention被保留

Version 6.0 Update:
    1.  对于meta-path上的每一条边记录这条边上的头实体和尾实体，与``pos``等同对待。
    2.  注释掉了Line 80，因为通过远程关系的边连接的两个实体不在同一个句子里（如果在同一个句子里会优先通过 path 1 找到），
        此时该句子可能包含别的实体，所以依然是可以延着这个句子搜索的，即再执行一次共句的实体搜索。

        此时会有一个现象，如果在``path``里存在一个句子的记录，该记录只包含头实体或者尾实体，此时该实体一定是一个
        通过远程关系连接的实体，此时我们假设了该句子的内容对path没有影响，即没有对逻辑关系进行了描述的贡献，
        因为对逻辑关系的描述隐含在了潜在的relation当中，而不是句子内容。

        带来的后果是我们其实不应当对这个实体进行实体替换，因为一旦替换，潜在的relation不再存在，逻辑关系也不再存在。
        如果严谨的话这里应该考虑对原有的context实体的替换逻辑进行修正，即首先筛选出这些孤立实体（以及参与了远程relation的实体？），
        只能在剩下的实体中进行替换。（或者直接去掉通过潜在relation连接的实体？）
Version 7.0 Update:
    TODO：
    1.  Version 6 做了一个假设，也就是每个句子只能用来连接两个实体，不能被重复使用
        这其实会导致数据减少，举例：e_1 经由 s_1 找到了 e_2，然后 s_1 被去掉，不能再用，如果只有 s_2 中包含了 e_3，
        那么此次搜索不可能再找到 e_3，因为 s_2 已经被拿来连接 e_1 和 e_2 了，但实际上是可以访问 e_3 的
        是否需要修复？

        好像其实是不需要修复的，因为在这个逻辑下，e_1 -> s_1 -> e_3 也是会被访问到的， 后续节点只要和 e_2 / e_3 中的任意一个相连就能被访问到。
    2.  ~~是否修复Version 6.0 Update 2 提到的这一点，加一个数组描述这个relation前后的实体不能被替换~~
    3.  ~~修复掉FIXME~~
Version 9.0 Update:
    1.  用BFS替换掉DFS来减少sentence的数量
    2.  取消掉通过纯关系的边的跳转，因为目前如果存在直接的句子连接，关系的边就被包含了，如果不存在一个直接的句子相连，这似乎是用到了常识知识，
        可能会引发与我们的预训练目标的不一致
Version 9.1 Update:
    1.  保存实体的时候用实体在wikidata里的id取代现在的enumerate得到的id，方便后续处理的时候判断是否是同一实体。
Version 9.2 Update:
    1.  通过指定path的长度增加更多的且更复杂的样本，原始的v9.1版本path（后续考虑增加从其他样本中找句子的可能）
"""


# TODO: 是不是考虑用DFS更好一些
def bfs(src_e_id, s_vis: Set, e_vis: Set, sent2ent, ent2sent, tgt_e_id, min_len):
    orig_path = ((src_e_id, -1),)
    queue = [(src_e_id, orig_path)]
    while len(queue) > 0:
        e_id, path = queue.pop(0)

        if e_id == tgt_e_id:
            assert len(path) > 1
            return path

        for next_s_id in ent2sent[e_id]:
            if next_s_id in s_vis:
                continue
            s_vis.add(next_s_id)
            for next_e_id in sent2ent[next_s_id]:
                if next_e_id in e_vis:
                    continue
                if next_e_id == tgt_e_id and e_id == src_e_id:  # Make sure that there are at least two sentences in the context.
                    continue
                if next_e_id == tgt_e_id and len(path) + 1 < min_len:
                    continue
                e_vis.add(next_e_id)
                queue.append((next_e_id, path + ((next_e_id, next_s_id),)))

    return None


def extract_entities_of_sent(sent_id, sent2ent, entities) -> Dict[str, List]:
    ent_ls = defaultdict(list)
    for e_id in sent2ent[sent_id]:
        # for pos_id, e in enumerate(entities[e_id]):
        #     if e["sent_id"] == sent_id:
        #         ent_ls[e_id][pos_id] = e
        for ent in entities:
            for pos_id, e in enumerate(ent):
                if e["id"] == e_id and e["sent_id"] == sent_id:
                    ent_ls[e_id].append(e)
    return ent_ls


def process_path(path: Tuple, sentences, entities, sent2ent):
    selected_sent_ids = set()
    sent_h_t_tuple = {}
    _last_ent_id = -1
    for edge_id, (e_id, e_sent_id) in enumerate(path):
        if edge_id == 0:
            assert e_sent_id == -1
        else:
            selected_sent_ids.add(e_sent_id)
            _h = _last_ent_id
            _t = e_id
            assert _t in sent2ent[e_sent_id]
            # 有几种情况：
            #   1.  如果是通过relation连接的，那么这个 e_sent_id 会出现（至少？）两次，第一次 h = -1，因为句子里没有 h
            #       如果有的话一定会先通过边连接的关系访问到
            #       此时 or 后面的部分会成立，第二次会把第一次的取代
            #   2.  如果是通过边连接的，e_sent_id只会出现一次
            # if e_sent_id not in sent_h_t_tuple or (sent_h_t_tuple[e_sent_id]["h"] == -1 or sent_h_t_tuple[e_sent_id]["t"] == -1):
            #     assert e_sent_id not in sent_h_t_tuple or sent_h_t_tuple[e_sent_id]["t"] != -1  # 确认应该不会有这种情况？
            #     sent_h_t_tuple[e_sent_id] = {
            #         "h": _h if _h in sent2ent[e_sent_id] else -1,
            #         "t": _t
            #     }
            # 现在没有通过relation连接的边，所以e_sent_id应该只会出现一次
            assert e_sent_id not in sent_h_t_tuple, (e_sent_id, sent_h_t_tuple, path)
            if e_sent_id not in sent_h_t_tuple:
                sent_h_t_tuple[e_sent_id] = {
                    "h": _h,
                    "t": _t,
                }
        _last_ent_id = e_id

    if len(selected_sent_ids) == len(sentences):
        return False, None, None
    if len(selected_sent_ids) == 0:
        return False, None, None

    selected_sent_ids = sorted(list(selected_sent_ids))
    selected_sentences = {
        s_id: {
            "sent": sentences[s_id],
            "ent": extract_entities_of_sent(s_id, sent2ent, entities),
            "h": sent_h_t_tuple[s_id]["h"],
            "t": sent_h_t_tuple[s_id]["t"]
        } for s_id in selected_sent_ids
    }

    # Extract the rest sentences and the contained entities to construct negative samples.
    rest_sentences = {
        s_id: {
            "sent": sentences[s_id],
            "ent": extract_entities_of_sent(s_id, sent2ent, entities)
        } for s_id in range(len(sentences)) if s_id not in selected_sentences
    }

    return True, selected_sentences, rest_sentences


def workflow(sample, max_min_len: int = 5):
    sample_id, sample = sample

    entities = sample["vertexSet"]
    relations = sample["labels"]
    sentences = sample["sents"]

    rel_edges = defaultdict(dict)
    for item in relations:
        rel_edges[item["h"]][item["t"]] = item["r"]

    ent2sent = defaultdict(set)

    for ent in entities:
        for e in ent:
            ent2sent[e["id"]].add(e["sent_id"])

    sent2ent = defaultdict(set)
    for ent in entities:
        for e in ent:
            sent2ent[e["sent_id"]].add(e["id"])

    ent_id2item = defaultdict(list)
    for ent in entities:
        for e in ent:
            ent_id2item[e["id"]].append(e)

    # Enumerate over each relation, try to find a path starting from the head entity
    # and ending with the tail entity.
    # Version 4.1: Enumerate over each entity pair <e_i, e_j> such that e_i and e_j has the common sentences.
    examples = []
    ent_pair_vis = set()
    ent_path_vis = set()
    for h in ent_id2item.keys():
        for t in ent_id2item.keys():
            if h == t:
                continue
            if (h, t) in ent_pair_vis or (t, h) in ent_pair_vis:
                continue
            h_sent_ids = ent2sent[h]
            t_sent_ids = ent2sent[t]
            common_sent_ids = h_sent_ids & t_sent_ids
            if len(common_sent_ids) == 0:
                continue
            # FIXED:
            #  这里不要遍历 entity h 的每一个句子的位置 可能有重复
            #  此外从 entity h 出发，初始化的 sent_vis 里面只需要包含 common_sent_ids 因为出发的这个句子也可能包含其他实体，是可以被检索的
            #  可以考虑找到一条path就返回，如果追求数量可以寻找足够多的path
            #  目前的写法影响除了可能产生比较多重复的数据意外应该没有其他的问题了
            sent_vis = deepcopy(common_sent_ids)
            for _min_len in range(3, max_min_len + 1):
                res = bfs(h, sent_vis, set(), sent2ent, ent2sent, t, min_len=_min_len)
                if res:
                    ent_path = "\t".join([tmp[0] for tmp in res])
                    if ent_path in ent_path_vis:
                        continue
                    ent_path_vis.add(ent_path)

                    assert len(res) >= 3
                    flag, selected, rest = process_path(res, sentences, entities, sent2ent)
                    if not flag:
                        print(11111)
                        continue
                    pos = [
                        {
                            "sent": sentences[pos_sent_id],
                            "h": h,
                            "t": t,
                            "ent": extract_entities_of_sent(pos_sent_id, sent2ent, entities)
                        } for pos_sent_id in common_sent_ids
                    ]
                    for c_s_id in common_sent_ids:
                        rest.pop(c_s_id)
                    examples.append({
                        "selected_sentences": selected,
                        "pos": pos,
                        "rest_sentences": rest,
                        "path": res,
                        "entity": {ent[0]["id"]: ent for ent in entities},
                        "all_sentences": sentences,
                        "id": sample_id,
                    })
                    ent_pair_vis.add((h, t))
                    ent_pair_vis.add((t, h))

    return examples


def extract_raw_text(sample):
    sentences = [" ".join(tokens) for tokens in sample["sents"]]
    return " ".join(sentences)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--sample', default=False, action='store_true')
    parser.add_argument('--max_min_len', type=int, default=5)

    args = parser.parse_args()

    if os.path.exists(args.input_file):
        input_files = [args.input_file]
    else:
        input_files = list(glob.glob(args.input_file))

    file_suffix = f'.path_v9.2_mm{args.max_min_len}.pkl'
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    all_examples_cnt = 0
    for _file_id, _file in enumerate(input_files):
        samples = json.load(open(_file))

        processed_samples = []
        with Pool(args.num_workers) as p:
            _annotate = partial(workflow, max_min_len=args.max_min_len)
            _results = list(tqdm(
                p.imap(_annotate, list(enumerate(samples)), chunksize=32),
                total=len(samples),
                desc='processing samples'
            ))

        no_negative_num = 0
        path_len_cnt = collections.Counter()
        for ex_id, _res in enumerate(_results):
            if _res:
                for _r in _res:
                    # _r["id"] = all_examples_cnt
                    _r["id"] = f"{_r['id']}_{_file_id}_{all_examples_cnt}"
                    all_examples_cnt += 1
                    processed_samples.append(_r)
                    path_len_cnt[len(_r["path"])] += 1

        for ex in processed_samples:
            if len(ex["rest_sentences"]) == 0:
                no_negative_num += 1

        print(f"Processed examples: {len(processed_samples)}")
        print(f"Example path length counter: {path_len_cnt}")
        print(f"Examples without hard negative samples: {no_negative_num}.")

        avg_sent_num = sum(map(lambda x: len(x["sents"]), samples)) / len(samples)
        print(f"Average sentence num: {avg_sent_num}")

        with Pool(args.num_workers) as p:
            _annotate = partial(extract_raw_text)
            _raw_texts = list(tqdm(
                p.imap(_annotate, samples, chunksize=32),
                total=len(samples),
                desc='extracting raw texts'
            ))

        raw_texts = [t for t in _raw_texts if t.strip()]
        print(f"Extracted {len(raw_texts)} segments of text.")

        if args.output_dir:
            _file_name = _file.split('/')[-1]
            output_file = os.path.join(args.output_dir, _file_name.replace('.json', file_suffix))
        else:
            output_file = _file.replace('.json', file_suffix)

        if args.sample and len(processed_samples) > 30:
            pickle.dump(processed_samples[:30], open(output_file.replace('.pkl', '.sample.pkl'), 'wb'))
            break

        pickle.dump({"examples": processed_samples, "raw_texts": raw_texts}, open(output_file, 'wb'))

    print("Done.")
