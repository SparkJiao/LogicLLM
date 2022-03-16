import json


class LSATReader:
    def __call__(self, file: str):
        data = json.load(open(file, 'r'))
        all_context = []
        all_question = []
        all_option_list = []
        all_label = []
        label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        for item in data:
            passage = item['passage']
            for q in item['questions']:
                ques = q['question']
                all_context.append(passage)
                all_question.append(ques)
                all_option_list.append(q['options'])
                all_label.append(label2id[q['answer']])

        return all_context, all_question, all_option_list, all_label
