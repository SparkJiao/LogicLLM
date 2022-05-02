import json
import argparse
from typing import List

annotations = {
    "199106_2-G_1": {
        "group1": ["Knosnik", "Londi", "Manley", "Neri", "Osata", "Poirier"],
        "group2": ["chair 1", "chair 2", "chair 3", "chair 4", "chair 5", "chair 6"],
        "relation": "sits on"
    },
    "199106_2-G_2": {
        "group1": ["the computer in office 1", "the computer in office 2", "the computer in office 3", "the computer in office 4",
                   "the printer in office 1", "the printer in office 2", "the printer in office 3", "the printer in office 4"],
        "group2": ["1987", "1988", "1989"],
        "relation": "was bought in"
    },
    "199106_2-G_3": {
        "group1": ["Gregg", "Hodges", "Ivan", "James", "King", "MacNeil", "Nader", "Owens"],
        "group2": ["1961", "1962", "1963", "1964", "1965", "1966", "1967", "1968"],
        "relation": "joined at"
    },
    "199106_2-G_4": {
        "group1": ["January tickets for line 1", "January tickets for line 2", "January tickets for line 3",
                   "February tickets for line 1", "February tickets for line 2", "February tickets for line 3"],
        "group2": ["green", "purple", "red", "yellow"],
        "relation": "are"
    },
    "199110_3-G_1": {
        "group1": ["Brandt", "Calva", "Duvall", "Eberle", "Fu", "Garcia", "Haga", "Irving", "Jessup"],
        "group2": ["Brandt\'s day", "Calva\'s day", "Duvall\'s day", "Eberle\'s day", "Fu\'s day", "Garcia\'s day", "Haga\'s day",
                   "Irving\'s day", "Jessup\'s day"],
        "relation": "was hired on"
    }
}


def clean_entities(entities: List[str]):
    entities = list(map(lambda x: x.replace(".", "").strip(), entities))
    return entities


def parsing(pred_str):
    group1_s = pred_str.find("#Group1:")
    group2_s = pred_str.find("#Group2:")
    relation_s = pred_str.find("#Relation:")

    if group1_s == -1 or group2_s == -1 or relation_s == -1:
        return None

    group1 = clean_entities(pred_str[group1_s + 8:group2_s - 1].split(","))
    group2 = clean_entities(pred_str[group2_s + 8:relation_s - 1].split(","))
    relation = pred_str[relation_s + 10:].strip().lower()

    return group1, group2, relation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)

    args = parser.parse_args()

    data = json.load(open(args.input_file))
    for item in data:
        if item["id"] in annotations:
            item.pop("prediction")
            item.update(annotations[item["id"]])
            continue

        prediction = item.pop("prediction")
        _res = parsing(prediction)
        if _res is not None:
            group1, group2, relation = _res
            item["group1"] = group1
            item["group2"] = group2
            item["relation"] = relation

    json.dump(data, open(args.input_file.replace(".json", ".parsed.json"), "w"), indent=2)
