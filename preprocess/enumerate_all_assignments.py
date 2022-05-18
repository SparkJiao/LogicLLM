import argparse
import copy
import json
from multiprocessing import Pool

from tqdm import tqdm
import os
import sys

pwd = os.getcwd()
f_pwd = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")

sys.path.append(f_pwd)

from data.data_utils import dfs_enumerate_all_assign


def func(item):
    if 'group1' not in item:
        return None

    ent_group1 = item['group1']
    ent_group2 = item['group2']
    relation = item['relation']
    assignments = []
    dfs_enumerate_all_assign(ent_group1, ent_group2, relation, assignments, '', set(range(len(ent_group1))))

    new_item = copy.deepcopy(item)
    new_item['all_assignments'] = assignments
    return new_item


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enumerate all assignments.')
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=16)

    args = parser.parse_args()

    # The number of combination exploded.
    data = json.load(open(args.input_file))
    with Pool(args.num_workers) as p:
        _results = list(tqdm(p.imap(func, data), total=len(data), desc="Enumerating all assignments"))

    id2res = {item['id']: item for item in _results if item is not None}
    # _results = list(tqdm(map(func, data), total=len(data), desc="Enumerating all assignments"))
    # id2res = {item['id']: item for item in _results if item is not None}

    output = []
    for item in data:
        output.append(id2res[item['id']] if item['id'] in id2res else item)
    assert len(output) == len(data)

    json.dump(data, open(args.input_file.replace(".json", ".assigned.json"), 'w'), indent=2)
