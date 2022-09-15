import argparse
from typing import List, Union, Dict, Any, Optional, Set, Tuple, AnyStr

"""
First construct positive and negative pairs following specific rules:
    1.  Given a reasoning pattern: a -> b -> c -> d; a -> d. (3-hop) 
        Define the positive pairs as: 
            (1) a -> b -> c; a -> c; if a -> c -> d exists (?). (double 2-hop)
            (2) a -> b -> c -> d; a -> d; (3-hop) 
                but the involved entities/concepts/events are different (can be obtained by manual augmentation ?)
            (3) a -> b -> c -> e -> d; 
        Any hard negative pairs?
        
        For reader training, we may add a simple post-processing technique to choose the retrieved patterns.
        That is all the relations should be matched (we may need to seek the help from sentence bert to calculate semantic distance).
        Entities are overlooked since the pattern is ignored with the entity.
        
        Positive examples retrieval process:
        

Then generate the corresponding text pairs.
"""


def _path2ent_key(path: List[str]):
    s, rel, t = path[0].split("\t")
    ent_ls = [s, t]
    for triplet in path[1:]:
        s, rel, t = triplet.split("\t")
        assert s == ent_ls[-1], path
        ent_ls.append(t)

    path_ent_key = "\t".join(ent_ls)
    return path_ent_key


def _path2key(path: List[str]):
    s, rel, t = path[0].split("\t")
    item_ls = [s, rel, t]
    for triplet in path[1:]:
        s, rel, t = triplet.split("\t")
        assert s == item_ls[-1]
        item_ls.extend([rel, t])
    return "\t".join(item_ls)


def _path2rel_key(path: List[str]):
    rel_ls = []
    for triplet in path:
        _, rel, _ = triplet.split("\t")
        rel_ls.append(rel)
    return "\t".join(rel_ls)


def generate_path_vocab():
    """
    If we need to generate a vocab to notes the indices of the positive pairs of each sample,
    so that we may mask them out during in-batch negative sampling.
    """
    ...
