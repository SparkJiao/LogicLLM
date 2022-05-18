import argparse
import copy
import json
from multiprocessing import Pool
from typing import Set, Dict, List

from tqdm import tqdm


def find_all_sub_assignments_by_depth(assignment: List[Dict], depth, values: List[str], relation: str, key_vis: Set, idx: str):
    # assignment:
    #    key: str
    #    value: str
    #    depth: int
    #    deduction: str, the verbalized deduction.
    #    flag: bool, indicates if the current branch can be extended further.
    #    key_vis: Set[str], keys that can be visited in the current branch.
    #    assignment: Dict, the following deductions with the same structure.
    if len(assignment) == 0:
        cnt = 0
        for key in key_vis:
            new_key_vis = copy.deepcopy(key_vis)
            new_key_vis.remove(key)
            new_key_vis = list(new_key_vis)
            for value in values:
                assignment.append({
                    'key': key,
                    'value': value,
                    'depth': depth + 1,
                    'deduction': key + ' ' + relation + ' ' + value + '.',
                    'flag': None,
                    'key_vis': new_key_vis,
                    'assignment': [],
                    'id': idx + str(cnt)
                })
                cnt += 1
        if cnt > 100000:
            raise Exception('Too many deductions.')
        return True
    else:
        # Think about if the process could be pruned.
        flag = False
        for assignment_item in assignment:
            if assignment_item['flag'] is True:
                sub_flag = find_all_sub_assignments_by_depth(assignment_item['assignment'], assignment_item['depth'],
                                                             values, relation, set(assignment_item['key_vis']), assignment_item['id'])
                if sub_flag is True:
                    flag = True
        return flag


def func(item):
    if 'group1' not in item:
        return None

    # Some wrong data or too much combinations.
    if item['id'] in ['199402_2-G_1', '199402_2-G_3']:
        return None

    print("Item id {}".format(item['id']))

    ent_group1 = item['group1']
    ent_group2 = item['group2']
    relation = item['relation']
    if 'assignment' not in item:
        item['assignment'] = []

    flag = find_all_sub_assignments_by_depth(item['assignment'], 0, ent_group2, relation, set(ent_group1), '0')

    return item, flag


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()

    data = json.load(open(args.input_file))

    # with Pool(args.num_workers) as p:
    #     _results = list(tqdm(p.imap(func, data), total=len(data), desc="Enumerating all assignments"))
    _results = list(tqdm(map(func, data), total=len(data), desc="Enumerating all assignments"))

    id2res = {}
    no_extension = 0
    for _res in _results:
        if _res is not None:
            _new_item, _flag = _res
            if _flag is True:
                id2res[_new_item['id']] = _new_item
            else:
                no_extension += 1

    print("Number of no extension examples: {}".format(no_extension))

    if len(id2res) == 0:
        print("No extension found.")
        exit(0)

    output = []
    for _item in data:
        output.append(id2res[_item['id']] if _item['id'] in id2res else _item)
    assert len(output) == len(data)

    json.dump(output, open(args.input_file.replace(".json", f".assigned_{args.depth}.json"), 'w'), indent=2)
