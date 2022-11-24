import argparse
import pickle
from tqdm import tqdm


def rearrange_example_id(examples):
    id_set = {}
    for exp in tqdm(examples, total=len(examples)):
        orig_id = str(exp["id"])
        if orig_id not in id_set:
            id_set[orig_id] = 0

        exp["id"] = orig_id + "_" + str(id_set[orig_id])
        id_set[orig_id] = id_set[orig_id] + 1
    return examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    data = pickle.load(open(args.input_file, "rb"))
    examples = rearrange_example_id(data["examples"])
    data["examples"] = examples
    pickle.dump(data, open(args.output_file, "wb"))
