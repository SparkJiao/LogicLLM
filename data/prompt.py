import json

from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from data.collators.dict2dict import DictTensorDataset
from general_util.logger import get_child_logger

logger = get_child_logger("Prompt")

"""
Passage 1:
Exactly six trade representatives negotiate a treaty: Klosnik, Londi, Manley, Neri, Osata, Poirier. 
There are exactly six chairs evenly spaced around a circular table. The chairs are numbered 1 through 6, 
with successively numbered chairs next to each other and chair number 1 next to chair number 6. 
Each chair is occupied by exactly one of the representatives. The following conditions apply: Poirier sits immediately next to Neri. 
Londi sits immediately next to Manley, Neri, or both. Klosnik does not sit immediately next to Manley. 
If Osata sits immediately next to Poirier, Osata does not sit immediately next to Manley.

Passage 2:
A small software firm has four offices, numbered 1, 2, 3, and 4. Each of its offices has exactly one computer and exactly one printer. 
Each of these eight machines was bought in either 1987, 1988, or 1989. 
The eight machines were bought in a manner consistent with the following conditions: 
The computer in each office was bought either in an earlier year than or in the same year as the printer in that office. 
The computer in office 2 and the printer in office 1 were bought in the same year. 
The computer in office 3 and the printer in office 4 were bought in the same year. 
The computer in office 2 and the computer in office 3 were bought in different years. 
The computer in office 1 and the printer in office 3 were bought in 1988.

Passage 3:
The eight partners of a law firm are Gregg, Hodges, Ivan, James, King, MacNeil, Nader, and Owens. 
In each of the years 1961 through 1968, exactly one of the partners joined the firm. 
Hodges joined the firm before Nader. King joined the firm before James. Nader and James joined the firm before Gregg. 
Nader joined the firm before Owens. James joined the firm before MacNeil. Gregg joined the firm before Ivan.

Passage 4:
A railway company has exactly three lines: line 1, line 2, and line 3. 
The company prints three sets of tickets for January and three sets of tickets for February: 
one set for each of its lines for each of the two months. 
The company's tickets are printed in a manner consistent with the following conditions: 
Each of the six sets of tickets is exactly one of the following colors: green, purple, red, yellow. 
For each line, the January tickets are a different color than the February tickets. 
For each month, tickets for different lines are in different colors. 
Exactly one set of January tickets is red. For line 3, either the January tickets or the February tickets, but not both, are green. 
The January tickets for line 2 are purple. No February tickets are purple.

Passage 5:
The Mammoth Corporation has just completed hiring nine new workers: Brandt, Calva, Duvall, Eberle, Fu, Garcia, Haga, Irving, and Jessup. 
Fu and Irving were hired on the same day as each other, and no one else was hired that day. 
Calva and Garcia were hired on the same day as each other, and no one else was hired that day. 
On each of the other days of hiring, exactly one worker was hired. Eberle was hired before Brandt. 
Haga was hired before Duvall. Duvall was hired after Irving but before Eberle. 
Garcia was hired after both Jessup and Brandt. Brandt was hired before Jessup.

Passage 6: (ID: 199110_3-G_4)
Exactly six dogs - P, Q, R, S, T, and U - are entered in a dog show. The judge of the show awards exactly four ribbons, 
one for each of first, second, third, and fourth places, to four of the dogs. 
The information that follows is all that is available about the six dogs: Each dog is either a greyhound or a labrador, but not both. 
Two of the six dogs are female and four are male. The judge awards ribbons to both female dogs, exactly one of which is a labrador. 
Exactly one labrador wins a ribbon. Dogs P and R place ahead of dog S, and dog S places ahead of dogs Q and T. Dogs P and R are greyhounds. 
Dogs S and U are labradors.

"""


def prompt_for_entity_extraction(file_path: str, tokenizer: PreTrainedTokenizer, max_input_length: int):
    # prefix = 'Extract the entities in the text: '
    # prompts = 'Input: A music store carries exactly ten types of CDs - both new and used of each of jazz, opera, pop, rap, and soul. ' \
    #           'Output: jazz, opera, pop, rap, soul ' \
    #           'Input: During a single week, from Monday through Friday, tours will be conducted of a company\'s three divisions\u2014' \
    #           'Operations, Production, and Sales. ' \
    #           'Output: Monday, Tuesday, Wednesday, Thursday, Friday, Operations, Production, Sales ' \
    #           'Input:  A closet contains exactly six hangers - 1, 2, 3, 4, 5, and 6 - hanging, in that order, from left to right. ' \
    #           'It also contains exactly six dresses - one gauze, one linen, one polyester, one rayon, one silk, and one wool. ' \
    #           'Output: 1, 2, 3, 4, 5, 6, one gauze, one linen, one polyester, one rayon, one silk, one wool'

    # prefix = 'Extract the two groups of participants and their relation: '
    # Original passages can be found above.
    # prompts = 'Input: Exactly six trade representatives negotiate a treaty: Klosnik, Londi, Manley, Neri, Osata, Poirier. ' \
    #           'There are exactly six chairs evenly spaced around a circular table. ' \
    #           'The chairs are numbered 1 through 6, ' \
    #           'with successively numbered chairs next to each other and chair number 1 next to chair number 6. ' \
    #           'Each chair is occupied by exactly one of the representatives. ' \
    #           'The following conditions apply: Poirier sits immediately next to Neri. ' \
    #           '' \
    #           'Output: #Group1: Knosnik, Londi, Manley, Neri, Osata, Poirier ' \
    #           '#Group2: chair 1, chair 2, chair 3, chair 4, chair 5, chair 6 ' \
    #           '#Relation: sits on ' \
    #           '' \
    #           '' \
    #           'Input: A small software firm has four offices, numbered 1, 2, 3, and 4. ' \
    #           'Each of its offices has exactly one computer and exactly one printer. ' \
    #           'Each of these eight machines was bought in either 1987, 1988, or 1989. ' \
    #           'The eight machines were bought in a manner consistent with the following conditions: ' \
    #           'The computer in office 2 and the printer in office 1 were bought in the same year. ' \
    #           '' \
    #           'Output: #Group1: the computer in office 1, the computer in office 2, the computer in office 3, the computer in office 4, ' \
    #           'the printer in office 1, the printer in office 2, the printer in office 3, the printer in office 4 ' \
    #           '#Group2: 1987, 1988, 1989 ' \
    #           '#Relation: were bought in ' \
    #           '' \
    #           '' \
    #           'Input: The eight partners of a law firm are Gregg, Hodges, Ivan, James, King, MacNeil, Nader, and Owens. ' \
    #           'In each of the years 1961 through 1968, exactly one of the partners joined the firm. ' \
    #           'Hodges joined the firm before Nader. ' \
    #           '' \
    #           'Output: #Group1: Gregg, Hodges, Ivan, James, King, MacNeil, Nader, Owens ' \
    #           '#Group2: 1961, 1962, 1963, 1964, 1965, 1966, 1964, 1968 ' \
    #           '#Relation: joined at ' \
    #           '' \
    #           '' \
    #           'Input: A railway company has exactly three lines: line 1, line 2, and line 3. ' \
    #           'The company prints three sets of tickets for January and three sets of tickets for February: ' \
    #           'one set for each of its lines for each of the two months. ' \
    #           'The company\'s tickets are printed in a manner consistent with the following conditions: ' \
    #           'Each of the six sets of tickets is exactly one of the following colors: green, purple, red, yellow. ' \
    #           'For each line, the January tickets are a different color than the February tickets. ' \
    #           'Exactly one set of January tickets is red. ' \
    #           'For line 3, either the January tickets or the February tickets, but not both, are green. ' \
    #           '' \
    #           'Output: #Group1: January tickets for line 1, January tickets for line 2, January tickets for line 3, ' \
    #           'February tickets for line 1, February tickets for line 2, February tickets for line 3 ' \
    #           '#Group2: green, purple, red, yellow ' \
    #           '#Relation: are ' \
    #           '' \
    #           '' \
    #           'Input: The Mammoth Corporation has just completed hiring nine new workers: ' \
    #           'Brandt, Calva, Duvall, Eberle, Fu, Garcia, Haga, Irving, and Jessup. ' \
    #           'Fu and Irving were hired on the same day as each other, and no one else was hired that day. ' \
    #           'On each of the other days of hiring, exactly one worker was hired. ' \
    #           'Eberle was hired before Brandt. ' \
    #           '' \
    #           '' \
    #           'Output: #Group1: Brandt, Calva, Duvall, Eberle, Fu, Garcia, Haga, Irving, Jessup' \
    #           '#Group2: Brandt\'s day, Calva\'s day, Duvall\'s day, Eberle\'s day, Fu\'s day, Garcia\'s day, Haga\'s day, ' \
    #           'Irving\'s day, Jessup\'s day ' \
    #           '#Relation: was hired ' \
    #           ''

    ### v1.3
    prefix = 'Extract the two groups of participants and their relation. Examples: '
    prompts = 'Input: Exactly six trade representatives negotiate a treaty: Klosnik, Londi, Manley, Neri, Osata, Poirier. ' \
              'There are exactly six chairs evenly spaced around a circular table. ' \
              'The chairs are numbered 1 through 6, ' \
              'with successively numbered chairs next to each other and chair number 1 next to chair number 6. ' \
              'Each chair is occupied by exactly one of the representatives. ' \
              'The following conditions apply: Poirier sits immediately next to Neri. ' \
              '' \
              'Output: #Group1: Knosnik, Londi, Manley, Neri, Osata, Poirier. ' \
              '#Group2: chair 1, chair 2, chair 3, chair 4, chair 5, chair 6. ' \
              '#Relation: sits on. ' \
              '' \
              '' \
              'Input: A small software firm has four offices, numbered 1, 2, 3, and 4. ' \
              'Each of its offices has exactly one computer and exactly one printer. ' \
              'Each of these eight machines was bought in either 1987, 1988, or 1989. ' \
              'The eight machines were bought in a manner consistent with the following conditions: ' \
              'The computer in office 2 and the printer in office 1 were bought in the same year. ' \
              '' \
              'Output: #Group1: the computer in office 1, the computer in office 2, the computer in office 3, the computer in office 4, ' \
              'the printer in office 1, the printer in office 2, the printer in office 3, the printer in office 4. ' \
              '#Group2: 1987, 1988, 1989. ' \
              '#Relation: were bought in. ' \
              '' \
              '' \
              'Input: The eight partners of a law firm are Gregg, Hodges, Ivan, James, King, MacNeil, Nader, and Owens. ' \
              'In each of the years 1961 through 1968, exactly one of the partners joined the firm. ' \
              'Hodges joined the firm before Nader. ' \
              '' \
              'Output: #Group1: Gregg, Hodges, Ivan, James, King, MacNeil, Nader, Owens. ' \
              '#Group2: 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968. ' \
              '#Relation: joined at. ' \
              '' \
              '' \
              'Input: A railway company has exactly three lines: line 1, line 2, and line 3. ' \
              'The company prints three sets of tickets for January and three sets of tickets for February: ' \
              'one set for each of its lines for each of the two months. ' \
              'The company\'s tickets are printed in a manner consistent with the following conditions: ' \
              'Each of the six sets of tickets is exactly one of the following colors: green, purple, red, yellow. ' \
              'For each line, the January tickets are a different color than the February tickets. ' \
              'Exactly one set of January tickets is red. ' \
              'For line 3, either the January tickets or the February tickets, but not both, are green. ' \
              '' \
              'Output: #Group1: January tickets for line 1, January tickets for line 2, January tickets for line 3, ' \
              'February tickets for line 1, February tickets for line 2, February tickets for line 3. ' \
              '#Group2: green, purple, red, yellow. ' \
              '#Relation: are. ' \
              '' \
              '' \
              'Input: The Mammoth Corporation has just completed hiring nine new workers: ' \
              'Brandt, Calva, Duvall, Eberle, Fu, Garcia, Haga, Irving, and Jessup. ' \
              'Fu and Irving were hired on the same day as each other, and no one else was hired that day. ' \
              'On each of the other days of hiring, exactly one worker was hired. ' \
              'Eberle was hired before Brandt. ' \
              '' \
              'Output: #Group1: Brandt, Calva, Duvall, Eberle, Fu, Garcia, Haga, Irving, Jessup. ' \
              '#Group2: Brandt\'s day, Calva\'s day, Duvall\'s day, Eberle\'s day, Fu\'s day, Garcia\'s day, Haga\'s day, ' \
              'Irving\'s day, Jessup\'s day. ' \
              '#Relation: was hired. ' \
              ''

    # prefix = 'Extract the two groups of participants and their relation. Examples: '
    # prompts = 'Input: Exactly six trade representatives negotiate a treaty: Klosnik, Londi, Manley, Neri, Osata, Poirier. ' \
    #           'There are exactly six chairs evenly spaced around a circular table. ' \
    #           'The chairs are numbered 1 through 6, ' \
    #           'with successively numbered chairs next to each other and chair number 1 next to chair number 6. ' \
    #           'Each chair is occupied by exactly one of the representatives. ' \
    #           'The following conditions apply: Poirier sits immediately next to Neri. ' \
    #           '' \
    #           'Output: #Group1: Knosnik, Londi, Manley, Neri, Osata, Poirier. ' \
    #           '#Group2: chair 1, chair 2, chair 3, chair 4, chair 5, chair 6. ' \
    #           '#Relation: sits on. ' \
    #           '' \
    #           '' \
    #           'Input: A small software firm has four offices, numbered 1, 2, 3, and 4. ' \
    #           'Each of its offices has exactly one computer and exactly one printer. ' \
    #           'Each of these eight machines was bought in either 1987, 1988, or 1989. ' \
    #           'The eight machines were bought in a manner consistent with the following conditions: ' \
    #           'The computer in office 2 and the printer in office 1 were bought in the same year. ' \
    #           '' \
    #           'Output: #Group1: the computer in office 1, the computer in office 2, the computer in office 3, the computer in office 4, ' \
    #           'the printer in office 1, the printer in office 2, the printer in office 3, the printer in office 4. ' \
    #           '#Group2: 1987, 1988, 1989. ' \
    #           '#Relation: were bought in. ' \
    #           '' \
    #           '' \
    #           'Input: The eight partners of a law firm are Gregg, Hodges, Ivan, James, King, MacNeil, Nader, and Owens. ' \
    #           'In each of the years 1961 through 1968, exactly one of the partners joined the firm. ' \
    #           'Hodges joined the firm before Nader. ' \
    #           '' \
    #           'Output: #Group1: Gregg, Hodges, Ivan, James, King, MacNeil, Nader, Owens. ' \
    #           '#Group2: 1961, 1962, 1963, 1964, 1965, 1966, 1964, 1968. ' \
    #           '#Relation: joined at. ' \
    #           '' \
    #           '' \
    #           'Input: A railway company has exactly three lines: line 1, line 2, and line 3. ' \
    #           'The company prints three sets of tickets for January and three sets of tickets for February: ' \
    #           'one set for each of its lines for each of the two months. ' \
    #           'The company\'s tickets are printed in a manner consistent with the following conditions: ' \
    #           'Each of the six sets of tickets is exactly one of the following colors: green, purple, red, yellow. ' \
    #           'For each line, the January tickets are a different color than the February tickets. ' \
    #           'Exactly one set of January tickets is red. ' \
    #           'For line 3, either the January tickets or the February tickets, but not both, are green. ' \
    #           '' \
    #           'Output: #Group1: January tickets for line 1, January tickets for line 2, January tickets for line 3, ' \
    #           'February tickets for line 1, February tickets for line 2, February tickets for line 3. ' \
    #           '#Group2: green, purple, red, yellow. ' \
    #           '#Relation: are. ' \
    #           '' \
    #           '' \
    #           'Input: The Mammoth Corporation has just completed hiring nine new workers: ' \
    #           'Brandt, Calva, Duvall, Eberle, Fu, Garcia, Haga, Irving, and Jessup. ' \
    #           'Fu and Irving were hired on the same day as each other, and no one else was hired that day. ' \
    #           'On each of the other days of hiring, exactly one worker was hired. ' \
    #           'Eberle was hired before Brandt. ' \
    #           '' \
    #           'Output: #Group1: Brandt, Calva, Duvall, Eberle, Fu, Garcia, Haga, Irving, Jessup. ' \
    #           '#Group2: Brandt\'s day, Calva\'s day, Duvall\'s day, Eberle\'s day, Fu\'s day, Garcia\'s day, Haga\'s day, ' \
    #           'Irving\'s day, Jessup\'s day. ' \
    #           '#Relation: was hired. ' \
    #           '' \
    #           '' \
    #           'Input: Exactly six dogs - P, Q, R, S, T, and U - are entered in a dog show. ' \
    #           'The judge of the show awards exactly four ribbons, ' \
    #           'one for each of first, second, third, and fourth places, to four of the dogs. ' \
    #           'The information that follows is all that is available about the six dogs: ' \
    #           'Each dog is either a greyhound or a labrador, but not both.' \
    #           'Two of the six dogs are female and four are male. ' \
    #           'The judge awards ribbons to both female dogs, exactly one of which is a labrador. ' \
    #           'Exactly one labrador wins a ribbon. ' \
    #           'Dogs P and R place ahead of dog S, and dog S places ahead of dogs Q and T. Dogs P and R are greyhounds. ' \
    #           'Dogs S and U are labradors.' \
    #           'Output: #Group1: P, Q, R, S, T, U. ' \
    #           '#Group2: greyhound and female, greyhound and male, labrador and female, labrador and female.' \
    #           '#Relation: are. ' \

    prompt_len = len(tokenizer.tokenize(prefix + prompts))
    logger.info(f"Length of the prompts: {prompt_len}")

    data = json.load(open(file_path, "r"))
    inputs = []
    for item in data:
        passage = item["passage"]
        seq = prefix + ' ' + prompts + ' Input: ' + passage
        seq = tokenizer.tokenize(seq)[:(max_input_length - 4)]
        inputs.append(tokenizer.convert_tokens_to_string(seq) + ' Output:')

    model_inputs = tokenizer(inputs,
                             padding=PaddingStrategy.LONGEST,
                             truncation=True,
                             max_length=max_input_length,
                             return_tensors="pt")
    logger.info(f"Input length: {model_inputs['input_ids'].size(1)}")

    return DictTensorDataset(model_inputs)


def prompt_for_rule_parsing(file_path: str, tokenizer: PreTrainedTokenizer):
    # prefix = 'Parse the sentence into logical forms using the functions: ' \
    #          'Before, After, Last, Next, Adjacent, Different, Same, BeforeEqual, AfterEqual, To, ' \
    #          'IfThen, IFF, And, Or, Unless, Neither, FirstPos, LastPos. ' \
    #          'Examples: '
    prefix = 'Parse the sentence into logical forms. Examples: '
    prompts = 'Input: Participants: H, J, K, M, N, O, P, X, Y, Z. ' \
              'Positions: 1, 2, 3, 4, 5, 6; Giuliani, Rodrigo, Vivaldi. ' \
              'Sentence: Two concertos will be selected from among three concertos by Giuliani—H, J, and K; ' \
              'two from among four concertos by Rodrigo—M, N, O, and P; ' \
              'and two from among three concertos by Vivaldi—X, Y, and Z. ' \
              'Output: count(select(any,Giuliani))=2 AND ' \
              'group(H,J,K,dim=1)=Giuliani AND count(select(any,Rodrigo))=2 AND ' \
              'group(M,N,O,P,dim=1)=Rodrigo AND count(select(any,Vivaldi))=2 AND group(X,Y,Z,dim=1)=Vivaldi, {1,2,3,4,5,6}.  ' \
              '' \
              'Input: Participants: H, J, K, M, N, O, P, X, Y, Z. ' \
              'Positions: 1, 2, 3, 4, 5, 6; Giuliani, Rodrigo, Vivaldi. ' \
              'Sentence: The following conditions apply without exception: If N is selected, then J is also selected. ' \
              'Output: value(N)>=1 ENTAIL value(J)>=1. ' \
              '' \
              'Input: Participants: H, J, K, M, N, O, P, X, Y, Z. ' \
              'Positions: 1, 2, 3, 4, 5, 6; Giuliani, Rodrigo, Vivaldi. ' \
              'Sentence: If both J and O are selected, then J is played at some time before O. ' \
              'X `cannot be played on the fifth Sunday unless one of Rodrigo\'s concertos is played on the first Sunday. ' \
              'Output: (group(J,O)>=1 ENTAIL value(J)<value(O)) AND ((NOT value(any)=1, {M,N,O,P}) ENTAIL NOT value(X)=5). ' \
              '' \
              '' \
              '' \
              'Input: Participants: H, J, K, M, N, O, P, X, Y, Z. ' \
              'Positions: 1, 2, 3, 4, 5, 6; Giuliani, Rodrigo, Vivaldi. ' \
              'Sentence: H Z M N Y K. ' \
              'Output: value(H)=1 AND value(Z)=2 AND value(M)=3 AND value(N)=4 AND value(Y)=5 AND value(K)=6. ' \
              '' \
              'Input: Participants: H, J, K, M, N, O, P, X, Y, Z. ' \
              'Positions: 1, 2, 3, 4, 5, 6; Giuliani, Rodrigo, Vivaldi. ' \
              'Sentence: If the six concertos to be played are J, K, N, O, Y, and Z and if N is to be played on the first Sunday, ' \
              'then which one of the following concertos CANNOT be played on the second Sunday? ' \
              'Output: value(any)>=1 AND value(N)=1, {J,K,N,O,Y,Z}, NOT. ' \
              '' \
              'Input: ' \
              'Participants: Fran, George, Henry, Joan, Kathy, Lewis, Nathan, Olga. ' \
              'Positions: swimming, tennis, volleyball. ' \
              'Sentence: The assignment of counselors must conform to the following conditions: ' \
              'Each activity is supervised by at least two, but not more than three, of the eight counselors. ' \
              'Output: count(select(any))>=2 AND count(select(any))<=3, {swimming, tennis, volleyball}. ' \
              '' \
              'Input: ' \
              'Participants: Fran, George, Henry, Joan, Kathy, Lewis, Nathan, Olga. ' \
              'Positions: swimming, tennis, volleyball. ' \
              'Sentence: Henry supervises swimming. ' \
              'Output: value(Henry)=swimming. ' \
              '' \
              'Input: ' \
              'Participants: Fran, George, Henry, Joan, Kathy, Lewis, Nathan, Olga. ' \
              'Positions: swimming, tennis, volleyball. ' \
              'Sentence: Neither Kathy nor Olga supervises tennis. ' \
              'Output: NOT (value(Kathy)=tennis) AND NOT (value(Olga)=tennis). ' \
              '' \
              'Input: ' \
              'Participants: Fran, George, Henry, Joan, Kathy, Lewis, Nathan, Olga. ' \
              'Positions: swimming, tennis, volleyball. ' \
              'Sentence: Neither Kathy nor Nathan supervises the same activity as Joan. ' \
              'Output: NOT (value(Kathy)=value(Joan)) AND NOT (value(Nathan)=value(Joan)). ' \
              '' \
              # 'Input: ' \
              # 'Participants: Fran, George, Henry, Joan, Kathy, Lewis, Nathan, Olga. ' \
              # 'Positions: swimming, tennis, volleyball. ' \
              # 'Sentence: Swimming: Fran, George, Henry; Tennis: Joan, Lewis; Volleyball: Kathy, Nathan, Olga. ' \
              # 'Output: swimming=group(Fran, George, Henry) AND tennis=group(Joan, Lewis) AND volleyball=group(Kathy, Nathan, Olga). ' \
              # ''

    prompt_len = len(tokenizer.tokenize(prefix + prompts))
    logger.info('prompt_len: %d', prompt_len)

    data = json.load(open(file_path, "r"))
    inputs = []
    for item in data:
        context_sentences = list(item["context"].values())
        participants = ";".join(item["participants"])
        positions = ";".join(item["positions"])

        for sent in context_sentences:
            inputs.append(prefix + prompts + f"Input: Participants: {participants}. Positions: {positions}. Sentence: {sent} Output:")

        for qa in item["qa"]:
            question = qa["question"]
            options = qa["option"]

            inputs.append(prefix + prompts + f"Input: Participants: {participants}. Positions: {positions}. Sentence: {question} Output:")
            for option in options:
                inputs.append(prefix + prompts + f"Input: Participants: {participants}. Positions: {positions}. Sentence: {option} Output:")

    model_inputs = tokenizer(inputs, padding=PaddingStrategy.LONGEST, truncation=TruncationStrategy.DO_NOT_TRUNCATE, return_tensors="pt")

    logger.info(f"Input length: {model_inputs['input_ids'].size(1)}")

    return DictTensorDataset(model_inputs)
