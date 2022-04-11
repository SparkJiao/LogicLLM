import argparse
import torch
import glob
import os
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--prob_threshold', type=float, default=0.25)

    args = parser.parse_args()

    all_examples, raw_texts = torch.load(args.input_file)
    if os.path.exists(args.pred_file):
        predictions = torch.load(args.pred_file)
        preds = predictions["preds"]
        index = predictions["index"]
        logits = predictions["logits"]
    else:
        pred_files = glob.glob(args.pred_file)
        preds = []
        index = []
        logits = []
        for pred_file in pred_files:
            predictions = torch.load(pred_file)
            preds.append(predictions["preds"])
            index.append(predictions["index"])
            logits.append(predictions["logits"])
        preds = torch.cat(preds, dim=0)
        index = torch.cat(index, dim=0)
        logits = torch.cat(logits, dim=0)

    correct_index = set(index[preds == 0].tolist())

    probs = logits.softmax(dim=-1)
    cor_probs = probs[preds == 0][:, 0]
    cor_probs_avg = cor_probs.mean()
    print(cor_probs_avg)

    above_num = 0
    for prob in tqdm(cor_probs, total=cor_probs.size(0)):
        if prob > args.prob_threshold:
            above_num += 1
    print("Above threshold: {}/{}".format(above_num, len(correct_index)))

    aug_num = 0
    for i, (example, raw_text) in enumerate(tqdm(zip(all_examples, raw_texts))):
        if i in correct_index:
            if "h" in example:
                aug_num += 1
    print("Augmented: {}/{}".format(aug_num, len(correct_index)))

    # filtered_examples = []
    # filtered_texts = []
    # for i, (example, raw_text) in tqdm(enumerate(zip(all_examples, raw_texts))):
    #     if i not in correct_index:
    #         filtered_examples.append(example)
    #         filtered_texts.append(raw_text)
    #
    # torch.save((filtered_examples, filtered_texts), args.output_file)
    # print("Filtered {} examples".format(len(filtered_examples)))
    # print("Saved to {}".format(args.output_file))
    # print("Done")
