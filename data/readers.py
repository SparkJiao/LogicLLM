import json
from typing import List


class LSATReader:
    label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    def __call__(self, file: str):
        data = json.load(open(file, 'r'))
        all_context = []
        all_question = []
        all_option_list = []
        all_label = []

        for item in data:
            passage = item['passage']
            for q in item['questions']:
                ques = q['question']
                all_context.append(passage)
                all_question.append(ques)
                all_option_list.append(q['options'])
                all_label.append(self.label2id[q['answer']])

        return all_context, all_question, all_option_list, all_label


class LogicNLILangReader:
    label2id = {
        'contradiction': 0,
        'self_contradiction': 1,
        'neutral': 2,
        'entailment': 3
    }

    def __call__(self, file):
        data = json.load(open(file, 'r'))
        all_facts = []
        all_rules = []
        all_statements = []
        all_labels = []
        for item in data.values():
            fact_ls: List[str] = item['facts']
            rule_ls: List[str] = item['rules']
            for statement, label in zip(item['statements'], item['labels']):
                all_facts.append(fact_ls)
                all_rules.append(rule_ls)
                all_statements.append(statement)
                all_labels.append(self.label2id[label])

        return all_facts, all_rules, all_statements, all_labels


class ReClorReader:
    def __call__(self, file):
        data = json.load(open(file, 'r'))

        all_context = []
        all_question = []
        all_option_list = []
        all_label = []
        for sample in data:
            all_context.append(sample["context"])
            all_question.append(sample["question"])
            if "label" not in sample:
                all_label.append(-1)
            else:
                all_label.append(sample["label"])
            all_option_list.append(sample["answers"])

        return all_context, all_question, all_option_list, all_label
