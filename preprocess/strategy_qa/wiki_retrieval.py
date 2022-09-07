import json
from nltk import word_tokenize
from multiprocessing import Pool
from tqdm import tqdm
from typing import List, Set
import sys
import argparse

sys.path.append("../../")

from data.bm25 import BM25Model


def read_wiki_corpus(file_path: str):
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, total=len(lines)):
            data.append(json.loads(line))

    del lines

    print(f"{len(data)} paragraphs loaded.")

    return data


def _tokenize(item):
    words = word_tokenize(item["para"])
    res = {
        "title": item["title"],
        "para_id": item["para_id"],
        "words": words
    }
    del item
    return res


def build_bm25_model(data, num_workers: int = 8):
    with Pool(num_workers) as p:
        documents = list(tqdm(
            p.imap(_tokenize, data, chunksize=32),
            total=len(data),
            desc="word tokenization"
        ))

    del data

    id_list = [f"{doc['title']}-{doc['para_id']}" for doc in documents]

    bm25_model = BM25Model(documents)
    return bm25_model, id_list


def load_strategy_qa(file_path: str):
    data = json.load(open(file_path))

    questions = []
    para_ids = []
    for item in data:
        questions.append(item["question"])

        item_para_ids = set()
        for evidence in item["evidence"]:
            for annotation in evidence:
                for evi_item in annotation:
                    if isinstance(evi_item, list):
                        for para_id in evi_item:
                            item_para_ids.add(para_id)
                    else:
                        assert evi_item in ["operation", "no_evidence"], evi_item
        para_ids.append(item_para_ids)

    return questions, para_ids


def bm25_retrieval(questions: List[str], para_ids: List[Set[str]], doc_id_list: List[str], bm25_model: BM25Model, top_k: int = 50):
    ques_recall_list = []
    for ques, q_para_ids in tqdm(zip(questions, para_ids), total=len(questions), desc="Calculating recall."):
        ques_words = word_tokenize(ques)
        doc_scores = bm25_model.get_documents_score(ques_words)
        doc_scores = [(i, score) for i, score in enumerate(doc_scores)]
        sorted_doc_scores = sorted(doc_scores, key=lambda x: x[1], reverse=False)
        top_k_doc_ids = [doc_id_list[i] for i, _ in sorted_doc_scores[:top_k]]
        q_recall = len(q_para_ids & set(top_k_doc_ids)) * 1.0 / len(q_para_ids)
        ques_recall_list.append(q_recall)

    return sum(ques_recall_list) * 1.0 / len(ques_recall_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file", type=str)
    parser.add_argument("--qa_file", type=str)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=100)
    args = parser.parse_args()

    corpus_data = read_wiki_corpus(args.corpus_file)
    bm25_model, doc_id_list = build_bm25_model(corpus_data, args.num_workers)
    qa_data, para_ids = load_strategy_qa(args.qa_file)
    recall = bm25_retrieval(qa_data, para_ids, doc_id_list, bm25_model, args.top_k)
    print(recall)

if __name__ == '__main__':
    main()
