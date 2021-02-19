# -*- coding: utf-8 -*-
import json
import sys
import os
import re
import numpy as np
np.set_printoptions(precision=3)
np.random.seed(2)
input_dir, output_dir = sys.argv[1], sys.argv[2]
# GLOBAL VARIABLES
DICT_SIZE = 400
MAX_LENGTH = 50
IGNORE_KEYS_IN_GOAL = ['eod', 'topic', 'messageLen', 'message']

replacements = []
with open(os.path.join(input_dir, 'mapping.pair'), 'r') as fin:
    for line in fin.readlines():
        tok_from, tok_to = line.replace('\n', '').split('\t')
        replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def insert_space(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text


def normalize(text):
    # lower case every word
    text = text.lower()
    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)
    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)
    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)
    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')
    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\"\<>@\(\)]', '', text) # remove
    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insert_space(token, text)
    # insert white space for 's
    text = insert_space('\'s', text)
    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]
    # remove multiple spaces
    text = re.sub(' +', ' ', text)
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)
    return text


def get_summary_bstate(bstate, get_domain=False):
    """Based on the mturk annotations we form multi-domain belief state"""
    domains = [u'taxi',u'restaurant',  u'hospital', u'hotel',u'attraction', u'train', u'police']
    summary_bstate = []
    summary_bvalue = []
    active_domain = []
    for domain in domains:
        domain_active = False

        booking = []
        #print(domain,len(bstate[domain]['book'].keys()))
        for slot in sorted(bstate[domain]['book'].keys()):
            if slot == 'booked':
                if len(bstate[domain]['book']['booked'])!=0:
                    booking.append(1)
                    # summary_bvalue.append("book {} {}:{}".format(domain, slot, "Yes"))
                else:
                    booking.append(0)
            else:
                if bstate[domain]['book'][slot] != "":
                    booking.append(1)
                    summary_bvalue.append(["{}-book {}".format(domain, slot.strip().lower()), normalize(bstate[domain]['book'][slot].strip())])
                else:
                    booking.append(0)
        if domain == 'train':
            if 'people' not in bstate[domain]['book'].keys():
                booking.append(0)
            if 'ticket' not in bstate[domain]['book'].keys():
                booking.append(0)
        summary_bstate += booking

        for slot in bstate[domain]['semi']:
            slot_enc = [0, 0, 0]  # not mentioned, dontcare, filled
            if bstate[domain]['semi'][slot] == 'not mentioned':
                slot_enc[0] = 1
            elif bstate[domain]['semi'][slot] in ['dont care', 'dontcare', "don't care", "do not care"]:
                slot_enc[1] = 1
                summary_bvalue.append(["{}-{}".format(domain, slot.strip().lower()), "dontcare"])
            elif bstate[domain]['semi'][slot]:
                summary_bvalue.append(["{}-{}".format(domain, slot.strip().lower()), normalize(bstate[domain]['semi'][slot].strip())])
            if slot_enc != [0, 0, 0]:
                domain_active = True
            summary_bstate += slot_enc

        # quasi domain-tracker
        if domain_active:
            summary_bstate += [1]
            active_domain.append(domain)
        else:
            summary_bstate += [0]

    assert len(summary_bstate) == 94
    if get_domain:
        return active_domain
    else:
        return summary_bstate, summary_bvalue


def analyze_dialogue(dialogue, maxlen):
    """Cleaning procedure for all kinds of errors in text and annotation."""
    # do all the necessary postprocessing
    if len(dialogue['log']) % 2 != 0:
        #print path
        print('odd # of turns')
        return None  # odd number of turns, wrong dialogue
    d_pp = {}
    d_pp['goal'] = dialogue['goal']  # for now we just copy the goal
    usr_turns = []
    sys_turns = []
    # last_bvs = []
    for i in range(len(dialogue['log'])):
        if len(dialogue['log'][i]['text'].split()) > maxlen:
            # print('too long')
            return None  # too long sentence, wrong dialogue
        if i % 2 == 0:  # usr turn
            text = dialogue['log'][i]['text']
            if not is_ascii(text):
                # print('not ascii')
                return None
            usr_turns.append(dialogue['log'][i])
        else:  # sys turn
            text = dialogue['log'][i]['text']
            if not is_ascii(text):
                # print('not ascii')
                return None
            belief_summary, belief_value_summary = get_summary_bstate(dialogue['log'][i]['metadata'])
            dialogue['log'][i]['belief_summary'] = str(belief_summary)
            dialogue['log'][i]['belief_value_summary'] = belief_value_summary
            sys_turns.append(dialogue['log'][i])
    d_pp['usr_log'] = usr_turns
    d_pp['sys_log'] = sys_turns
    return d_pp


def getDomain(idx, log, domains, last_domain):
    if idx == 1:
        active_domains = get_summary_bstate(log[idx]["metadata"], True) 
        crnt_doms = active_domains[0] if len(active_domains)!=0 else domains[0]
        return crnt_doms
    else:
        ds_diff = get_ds_diff(log[idx-2]["metadata"], log[idx]["metadata"])
        if len(ds_diff.keys()) == 0: # no clues from dialog states
            crnt_doms = last_domain
        else:
            crnt_doms = list(ds_diff.keys())
        return crnt_doms[0] # How about multiple domains in one sentence senario ?


def get_ds_diff(prev_d, crnt_d):
    diff = {}
    # Sometimes, metadata is an empty dictionary, bug?
    if not prev_d or not crnt_d:
        return diff

    for ((k1, v1), (k2, v2)) in zip(prev_d.items(), crnt_d.items()):
        assert k1 == k2
        if v1 != v2: # updated
            diff[k2] = v2
    return diff


def fix_general_label_error(label_dict, slots):
    GENERAL_TYPO = {
        # type
        "guesthouse": "guest house", "guesthouses": "guest house", "guest": "guest house", "mutiple sports": "multiple sports", 
        "sports": "multiple sports", "mutliple sports": "multiple sports", "swimmingpool": "swimming pool", "concerthall": "concert hall", 
        "concert": "concert hall", "pool": "swimming pool", "night club": "nightclub", "mus": "museum", "ol": "architecture", 
        "colleges": "college", "coll": "college", "architectural": "architecture", "musuem": "museum", "churches": "church",
        # area
        "center": "centre", "center of town": "centre", "near city center": "centre", "in the north": "north", "cen": "centre", "east side": "east", 
        "east area": "east", "west part of town": "west", "ce": "centre", "town center": "centre", "centre of cambridge": "centre", 
        "city center": "centre", "the south": "south", "scentre": "centre", "town centre": "centre", "in town": "centre", "north part of town": "north", 
        "centre of town": "centre", "cb30aq": "none",
        # price
        "mode": "moderate", "moderate -ly": "moderate", "mo": "moderate", 
        # day
        "next friday": "friday", "monda": "monday", 
        # parking
        "free parking": "free",
        # internet
        "free internet": "yes",
        # star
        "4 star": "4", "4 stars": "4", "0 star rarting": "none",
        # others 
        "y": "yes", "any": "dontcare", "n": "no", "does not care": "dontcare", "not men": "none", "not": "none", "not mentioned": "none",
        '': "none", "not mendtioned": "none", "3 .": "3", "does not": "no", "fun": "none", "art": "none",  
    }

    for slot in slots:
        if slot in label_dict.keys():
            # general typos
            if label_dict[slot] in GENERAL_TYPO.keys():
                label_dict[slot] = label_dict[slot].replace(label_dict[slot], GENERAL_TYPO[label_dict[slot]])

            # miss match slot and value 
            if  slot == "hotel-type" and label_dict[slot] in ["nigh", "moderate -ly priced", "bed and breakfast", "centre", "venetian", "intern", "a cheap -er hotel"] or \
                slot == "hotel-internet" and label_dict[slot] == "4" or \
                slot == "hotel-pricerange" and label_dict[slot] == "2" or \
                slot == "attraction-type" and label_dict[slot] in ["gastropub", "la raza", "galleria", "gallery", "science", "m"] or \
                "area" in slot and label_dict[slot] in ["moderate"] or \
                "day" in slot and label_dict[slot] == "t":
                label_dict[slot] = "none"
            elif slot == "hotel-type" and label_dict[slot] in ["hotel with free parking and free wifi", "4", "3 star hotel"]:
                label_dict[slot] = "hotel"
            elif slot == "hotel-star" and label_dict[slot] == "3 star hotel":
                label_dict[slot] = "3"
            elif "area" in slot:
                if label_dict[slot] == "no": label_dict[slot] = "north"
                elif label_dict[slot] == "we": label_dict[slot] = "west"
                elif label_dict[slot] == "cent": label_dict[slot] = "centre"
            elif "day" in slot:
                if label_dict[slot] == "we": label_dict[slot] = "wednesday"
                elif label_dict[slot] == "no": label_dict[slot] = "none"
            elif "price" in slot and label_dict[slot] == "ch":
                label_dict[slot] = "cheap"
            elif "internet" in slot and label_dict[slot] == "free":
                label_dict[slot] = "yes"

            # some out-of-define classification slot values
            if  slot == "restaurant-area" and label_dict[slot] in ["stansted airport", "cambridge", "silver street"] or \
                slot == "attraction-area" and label_dict[slot] in ["norwich", "ely", "museum", "same area as hotel"]:
                label_dict[slot] = "none"
    return label_dict


def create_data(): 
    # create dictionary of delexicalied values that then we will search against, order matters here!
    # dic = delexicalize.prepareSlotValuesIndependent()
    delex_data = {}

    with open(os.path.join(input_dir, 'data.json'), 'r') as f:
        data = json.load(f)

    for dialogue_name, dialogue in data.items():
        domains = []
        for dom_k, dom_v in dialogue['goal'].items():
            if dom_v and dom_k not in IGNORE_KEYS_IN_GOAL: # check whether contains some goal entities
                domains.append(dom_k)
        last_domain  = ""
        for idx, turn in enumerate(dialogue['log']):
            origin_text = normalize(turn['text'])
            dialogue['log'][idx]['text'] = origin_text
            # FIXING delexicalization:
            if 'dialog_act' in dialogue['log'][idx] and not isinstance(dialogue['log'][idx]['dialog_act'], str):
                for k in dialogue['log'][idx]['dialog_act'].keys():
                    if 'Attraction' in k:
                        if 'restaurant_' in dialogue['log'][idx]['text']:
                            dialogue['log'][idx]['text'] = dialogue['log'][idx]['text'].replace("restaurant", "attraction")
                        if 'hotel_' in dialogue['log'][idx]['text']:
                            dialogue['log'][idx]['text'] = dialogue['log'][idx]['text'].replace("hotel", "attraction")
                    if 'Hotel' in k:
                        if 'attraction_' in dialogue['log'][idx]['text']:
                            dialogue['log'][idx]['text'] = dialogue['log'][idx]['text'].replace("attraction", "hotel")
                        if 'restaurant_' in dialogue['log'][idx]['text']:
                            dialogue['log'][idx]['text'] = dialogue['log'][idx]['text'].replace("restaurant", "hotel")
                    if 'Restaurant' in k:
                        if 'attraction_' in dialogue['log'][idx]['text']:
                            dialogue['log'][idx]['text'] = dialogue['log'][idx]['text'].replace("attraction", "restaurant")
                        if 'hotel_' in dialogue['log'][idx]['text']:
                            dialogue['log'][idx]['text'] = dialogue['log'][idx]['text'].replace("hotel", "restaurant")

            if idx % 2 == 1:  # if it's a system turn
                cur_domain = getDomain(idx, dialogue['log'], domains, last_domain)
                last_domain = [cur_domain]
                dialogue['log'][idx - 1]['domain'] = cur_domain
                acts = []
                if 'dialog_act' in dialogue['log'][idx] and not isinstance(dialogue['log'][idx]['dialog_act'], str):
                    for k in dialogue['log'][idx]['dialog_act'].keys():
                        if k.split('-')[1].lower() == 'request':
                            for a in dialogue['log'][idx]['dialog_act'][k]:
                                acts.append(a[0].lower())
                        elif k.split('-')[1].lower() == 'inform':
                            for a in dialogue['log'][idx]['dialog_act'][k]:
                                acts.append([a[0].lower(), normalize(a[1])])
                dialogue['log'][idx]['dialog_act'] = acts        
        delex_data[dialogue_name] = dialogue
    return delex_data


def main():
    with open(os.path.join(input_dir, "ontology.json"), "r") as f:
        ontology = json.load(f)
    slot_map = {}
    for k in ontology.keys():
        d, m, s = k.lower().split("-")
        if m == "semi":
            ds = d + "-" + s
        else:
            ds = d + "-" + m + " " + s
        slot_map[ds] = {"domain": "[" + d + "]", "slot": "[" + s + "]"}
    with open(os.path.join(output_dir, "slot_map.json"), mode="w") as f:
        json.dump(slot_map, f, indent=2, ensure_ascii=False)

    data = create_data()
    print("finish load data")
    with open(os.path.join(input_dir, 'testListFile.txt'), 'r') as f:
        testListFile = [line.strip() for line in f]
    with open(os.path.join(input_dir, 'valListFile.txt'), 'r') as f:
        valListFile = [line.strip() for line in f]

    test_dials = []
    val_dials = []
    train_dials = []
    count_train, count_val, count_test = 0, 0, 0
    
    for dialogue_name, dial_item in data.items():
        # print dialogue_name
        domains = []
        for dom_k, dom_v in dial_item['goal'].items():
            if dom_v and dom_k not in IGNORE_KEYS_IN_GOAL: # check whether contains some goal entities
                domains.append(dom_k)
        dial = []
        d_orig = analyze_dialogue(dial_item, MAX_LENGTH)  # max turn len is 50 words
        if d_orig is None:
            continue
        usr = [t['text'] for t in d_orig['usr_log']]
        sys = [t['text'] for t in d_orig['sys_log']]
        sys_acts = [t['dialog_act'] for t in d_orig['sys_log']]
        bs_dict = [{each[0]: each[1] for each in t['belief_value_summary']} for t in d_orig['sys_log']]
        domain = [t['domain'] for t in d_orig['usr_log']]
        for item in zip(usr, sys, sys_acts, domain, bs_dict):
            dial.append({'usr':item[0],'sys':item[1], 'sys_a':item[2], 'domain':item[3], 'bvs':item[4]})

        dialogue = {}
        dialogue['dialogue_idx'] = dialogue_name
        dialogue['domains'] = list(set(domains)) #list(set([d['domain'] for d in dial]))
        last_bs = {}
        dialogue['dialogue'] = []

        for turn_i, turn in enumerate(dial):
            # usr, usr_o, sys, sys_o, sys_a, domain
            turn_dialog = {}
            turn_dialog['system_transcript'] = dial[turn_i-1]['sys'] if turn_i > 0 else ""
            turn_dialog['turn_idx'] = turn_i
            turn_dialog['belief_state'] = fix_general_label_error(turn['bvs'], slot_map.keys())
            turn_dialog['turn_label'] = {bs: turn_dialog['belief_state'][bs] for bs in turn_dialog['belief_state'] if bs not in last_bs}
            turn_dialog['transcript'] = turn['usr']
            turn_dialog['system_acts'] = dial[turn_i-1]['sys_a'] if turn_i > 0 else []
            turn_dialog['domain'] = turn['domain']
            last_bs = turn_dialog['belief_state']
            dialogue['dialogue'].append(turn_dialog)
        
        if dialogue_name in testListFile:
            test_dials.append(dialogue)
            count_test += 1
        elif dialogue_name in valListFile:
            val_dials.append(dialogue)
            count_val += 1
        else:
            train_dials.append(dialogue)
            count_train += 1

    print("# of dialogues: Train {}, Val {}, Test {}".format(count_train, count_val, count_test))
    # save all dialogues
    with open(os.path.join(output_dir, 'dev_dials.json'), 'w') as f:
        json.dump(val_dials, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, 'test_dials.json'), 'w') as f:
        json.dump(test_dials, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, 'train_dials.json'), 'w') as f:
        json.dump(train_dials, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
