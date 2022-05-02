import json
from typing import List
from data.data_utils import dfs_enumerate_all_assign


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


class LSATProofWriterReader:
    label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    def __call__(self, file: str):
        data = json.load(open(file, 'r'))
        all_context = []
        all_question = []
        all_option_list = []
        all_label = []

        for item in data:
            passage = item['passage']
            p_deductions = []
            iter_id = 0
            while f"prediction_{iter_id}" in item:
                p_deductions.append(item[f"prediction_{iter_id}"])
                iter_id += 1
            for q in item['questions']:
                ques = q['question']
                q_deductions = []
                iter_id = 0
                while f"prediction_{iter_id}" in q:
                    q_deductions.append(q[f"prediction_{iter_id}"])
                    iter_id += 1
                all_context.append(' '.join([passage] + p_deductions))
                all_question.append(' '.join([ques] + q_deductions))
                all_option_list.append(q['options'])
                all_label.append(self.label2id[q['answer']])

        return all_context, all_question, all_option_list, all_label


class LSATReaderWPrompt:
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
                deduction = q['prediction']
                all_context.append(passage + ' ' + deduction)
                all_question.append(ques)
                all_option_list.append(q['options'])
                all_label.append(self.label2id[q['answer']])

        return all_context, all_question, all_option_list, all_label


class LSATReaderTrigger(LSATReader):
    relational_trigger_words = {
        'before': ['before', 'above', 'precede', 'earlier'],
        'after': ['after', 'larger', 'higher', 'bigger', 'older'],
        'last': ['immediately before', 'last'],
        'next': ['immediately after', 'next'],
        'adjacent': ['neighboring', 'adjacent'],
        'different': ['different'],
        'same': ['same', 'also'],
        'before_equal': ['no later'],
        'after_equal': ['no earlier'],
        'to': ['to', 'on', 'given', 'in']
    }
    relational_prompt = {
        'before': 'participant #1 is in the position before participant #2.',
        'after': 'participant #1 is in the position after participant #2.',
        'last': 'participant #1 is in the last position of participant #2.',
        'next': 'participant #1 is next to participant #2.',
        'adjacent': 'participant #1 is neighbouring to participant #2.',
        'different': 'participant #1 is in the different position with participant #2.',
        'same': 'participant #1 is in the same position with participant #2.',
        'before_equal': 'participant #1 is before or equals to the position of participant #2.',
        'after_equal': 'participant #1 is after or equals to the position of participant #2.',
        'to': 'participant #1 is assigned to the position #2.'
    }

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

                prompts = []
                template = "The operation {} means that {}"
                for trigger, trigger_words in self.relational_trigger_words.items():
                    for _word in trigger_words:
                        if _word in ques or _word in passage:
                            prompts.append(template.format(_word, self.relational_prompt[trigger]))
                            break
                passage = ' '.join(prompts) + ' ' + passage

                all_context.append(passage)
                all_question.append(ques)
                all_option_list.append(q['options'])
                all_label.append(self.label2id[q['answer']])

        return all_context, all_question, all_option_list, all_label


class LSATReaderTriggerV2(LSATReader):
    relational_trigger_words = {
        'before': ['before', 'above', 'precede', 'earlier'],
        'after': ['after', 'larger', 'higher', 'bigger', 'older'],
        'last': ['immediately before', 'last'],
        'next': ['immediately after', 'next'],
        'adjacent': ['neighboring', 'adjacent'],
        'different': ['different'],
        'same': ['same', 'also'],
        'before_equal': ['no later'],
        'after_equal': ['no earlier'],
        'to': ['to', 'on', 'given', 'in']
    }
    relational_prompt = {
        # 'before': 'participant #1 is in the position before participant #2.',
        'before': 'If A does something {} B, then A is in the position before B.',
        # 'after': 'participant #1 is in the position after participant #2.',
        'after': 'If A does something {} B, then A is in the position after B.',
        # 'last': 'participant #1 is in the last position of participant #2.',
        'last': 'If A does something {} B, then A is in the last position of B.',
        # 'next': 'participant #1 is next to participant #2.',
        'next': 'If A does something {} B, then A is next to B.',
        # 'adjacent': 'participant #1 is neighbouring to participant #2.',
        'adjacent': 'If A does something {} to B, then A is neighbouring to B.',
        # 'different': 'participant #1 is in the different position with participant #2.',
        'different': 'If A does something {} to B, then A is in the different position with B.',
        # 'same': 'participant #1 is in the same position with participant #2.',
        'same': 'If A does something {} to B, then A is in the same position with B.',
        # 'before_equal': 'participant #1 is before or equals to the position of participant #2.',
        'before_equal': 'If A does something {} than B, then A is before or equals to the position of B.',
        # 'after_equal': 'participant #1 is after or equals to the position of participant #2.',
        'after_equal': 'If A does something {} than B, then A is after or equals to the position of B.',
        # 'to': 'participant #1 is assigned to the position #2.'
        'to': 'If A does something {} B, then A is assigned to the position of B.'
    }

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

                prompts = []
                for trigger, trigger_words in self.relational_trigger_words.items():
                    for _word in trigger_words:
                        if _word in ques or _word in passage:
                            prompts.append(self.relational_prompt[trigger].format(_word))
                            break
                passage = ' '.join(prompts) + ' ' + passage

                all_context.append(passage)
                all_question.append(ques)
                all_option_list.append(q['options'])
                all_label.append(self.label2id[q['answer']])

        return all_context, all_question, all_option_list, all_label


class LSATAssignmentEnumerationReader(LSATReader):
    def __call__(self, file: str):
        data = json.load(open(file, 'r'))
        all_context = []
        all_question = []
        all_option_list = []
        all_label = []
        all_assignment = []

        for item in data:
            passage = item['passage']

            ent_group1 = item['group1']
            ent_group2 = item['group2']
            relation = item['relation']
            assignments = []
            dfs_enumerate_all_assign(ent_group1, ent_group2, relation, assignments, '', set(range(len(ent_group1))))
            item['all_assignment'] = assignments

            for q in item['questions']:
                ques = q['question']
                all_context.append(passage)
                all_question.append(ques)
                all_option_list.append(q['options'])
                all_label.append(self.label2id[q['answer']])
                all_assignment.append(assignments)

        return all_context, all_question, all_option_list, all_label, all_assignment


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
