import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    examples, raw_texts = torch.load(args.input_data)
    predictions = torch.load(args.predictions)
    index, predictions = predictions["index"], predictions["predictions"]

    output_examples = []
    for i, pred in zip(index, predictions):
        if pred == 0:
            output_examples.append(examples[i])

    print(f"Number of examples: {len(output_examples)}")
    torch.save((output_examples, raw_texts), args.output_file)


if __name__ == '__main__':
    main()
