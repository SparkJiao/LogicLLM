import json
from nltk import sent_tokenize


def process_item(item):
    context = item["context"]
    mention2entity = item["mention2entity"]
    entity2type = item["entity2type"]

    sentences = sent_tokenize(context)






