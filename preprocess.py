import json
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
import os
from tqdm import tqdm
import random
import copy
import argparse
from functools import partial
from collections import OrderedDict
from transformers import (
    AutoTokenizer,
    MBartTokenizer,
    MBartTokenizerFast
)

MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast]
zh2en_SLOT_MAP = {"名字": "name", "位置":"location", "评分":"rating", 
"类别":"type", "地址":"address", "电话":"phone_number", 
"可用选项":"available_options", "参考编号":"ref_number", 
"每晚价格":"price_per_night", "用户名":"user_name", "房间数":"number_of_rooms", 
"预订月":"start_month", "预订日":"start_day", "预订天数":"number_of_nights", 
"价格范围":"price_level", "星级数":"stars", "菜品":"cuisine", 
"饮食限制":"dietary_restrictions", "日期":"day", "城市":"city", 
"最高温度":"max_temp", "最低温度":"min_temp", "天气":"weather", 
"描述":"description", "出发地":"departure", "目的地":"destination", 
"人数":"number_of_people", "时间":"time"}
zh2en_API_MAP = {"餐馆查询": "restaurants_zh_CN_search", "餐馆预订":"restaurants_zh_CN_booking", 
"宾馆查询": "hotels_zh_CN_search", "宾馆预订": "hotels_zh_CN_booking", 
"景点查询":"attractions_zh_CN_search", "天气查询":"weathers_zh_CN_search", 
"香港地铁":"HKMTR_zh"}
en2zh_RELATION_MAP = {"equal_to":"等于", "not":"非", "less_than":"少于", "at_least":"至少", "one_of":"其中之一"}

API_MAP = {'chat':'chat', "restaurants_en_US_search": "restaurants search", "restaurants_en_US_booking":"restaurants booking", 
"hotels_en_US_search": "hotels search", "hotels_en_US_booking": "hotels booking", 
"attractions_en_US_search": "attractions search", "weathers_en_US_search": "weathers search", 
"HKMTR_en": "HKMTR en"}
API_MAP.update({v:k for k, v in zh2en_API_MAP.items()})
API_MAP.update({k:k for k, v in zh2en_API_MAP.items()})
en_API_MAP = {'chat':'chat', "restaurants_en_US_search": "restaurants search", "restaurants_en_US_booking":"restaurants booking", 
"hotels_en_US_search": "hotels search", "hotels_en_US_booking": "hotels booking", 
"attractions_en_US_search": "attractions search", "weathers_en_US_search": "weathers search", 
"HKMTR_en": "HKMTR en"}

# for cross lingual transfer
with open("data/dict_en_zh.json") as f:
    en_zh_value_MAP = json.load(f)
zh_en_value_MAP = {v:k for k, v in en_zh_value_MAP.items()}
zh_en_API_MAP = {"餐馆查询": "restaurants_en_US_search", "餐馆预订":"restaurants_en_US_booking", 
"宾馆查询": "hotels_en_US_search", "宾馆预订": "hotels_en_US_booking", 
"景点查询":"attractions_en_US_search", "天气查询":"weathers_en_US_search", 
"香港地铁":"HKMTR_en"}

random.seed(577)


def read_require_slots():
    require_slots = {}
    for fn in os.listdir("knowledgebase/apis"):
        api_name = fn.replace(".json", "")
        with open(os.path.join("knowledgebase/apis", fn)) as f:
            ontology = json.load(f)
            require_slots[api_name] = ontology["required"]
    return require_slots


def state2span(state, required_slots):
    
    span = ""
    for intent in state:
        span += f"<API> {intent}"
        # check the required slots
        if len(required_slots[intent])>0:
            for slot in required_slots[intent]:
                if slot in state[intent]:
                    relation = state[intent][slot]["relation"]
                    values = [str(value) for value in state[intent][slot]["value"]]
                    values = "<value> ".join(values)
                    span += f"<slot> {slot}<relation> {relation}<value> {values}"
                else:
                    span += f"<slot> {slot}<unknow>"
        else:
            for slot in state[intent]:
                relation = state[intent][slot]["relation"]
                values = [str(value) for value in state[intent][slot]["value"]]
                values = "<value> ".join(values)
                span += f"<slot> {slot}<relation> {relation}<value> {values}"
    return span


def compute_lev_span(previous_state, new_state, intent):
    Lev = f"<API> {intent}"
    if intent=="chat":
        return "<API>"
    old_state = copy.deepcopy(previous_state)
    if intent not in old_state:
        old_state[intent] = {}
    for slot in new_state[intent]:
        if old_state[intent].get(slot) != new_state[intent].get(slot):
            relation = new_state[intent][slot]["relation"]
            values = [str(value) for value in new_state[intent][slot]["value"]]
            values = "<value> ".join(values)
            Lev += f"<slot> {slot}<relation> {relation}<value> {values}"
    for slot in old_state[intent]:
        if slot not in new_state[intent]:
            print(intent, old_state[intent][slot])
            Lev += f"<slot> {slot}<unknow>"
    return Lev

def knowledge2span(knowledge):
    knowledge_text = "<knowledge>" 
    for domain, item in knowledge.items():
        knowledge_text += f" [{domain}]"
        for slot, values in item.items():
            if slot not in ["type", "description", "类别", "描述"]:
                if isinstance(values, list):
                    values_text = "<value> ".join(values)
                else:
                    values_text = str(values)
                knowledge_text += f"<slot> {slot}<value> {values_text}"
    return knowledge_text


def read_data(args, path_names, tokenizer, max_history=3):
    print(("Reading all files from {}".format(path_names)))
    data = []
    # dst_data = []
    required_slots = read_require_slots()

    required_slots = {API_MAP[k]:v for k, v in required_slots.items()}
    
    # read files
    for path_name in path_names:
        with open(path_name) as f:
            dials = json.load(f)
            # cross lingual adaptation setting
            # use only 10% data from target lang
            if "2" in args.setting:
                _, target_lang = args.setting.split("2")
                if f"{target_lang}_train" in path_name:
                    if not os.path.exists(f"data/{target_lang}_fewshot_dials.json"):
                        dial_ids = list(dials.keys())
                        dial_ids = dial_ids[:(len(dial_ids)//10)]
                        print(f"few shot for {target_lang}, dialogue number: {len(dial_ids)}")
                        with open(f"data/{target_lang}_fewshot_dials.json", "w") as f:
                            json.dump({"fewshot_dials":dial_ids}, f, indent=True)
                    else:
                        with open(f"data/{target_lang}_fewshot_dials.json") as f:
                            dial_ids = json.load(f)["fewshot_dials"]
                    dials = {dial_id:dials[dial_id] for dial_id in dial_ids}
            

            for dial_id, dial in dials.items():
                dialog_history = []
                knowledge = {}
                knowledge_text = "<knowledge>"
                API_flag = False
                turn_id = 0
                last_dialogue_state = {}

                for turn in dial["Events"]:
                    
                    if turn["Agent"] == "KnowledgeBase":
                        domain = turn["Topic"].split("_")[0]

                        if domain not in knowledge:
                            knowledge[domain] = {}
                        
                        if int(turn["TotalItems"]) == 0:
                            knowledge_text = f"<knowledge> [{domain}] Message = No item avaiable." 
                        else:
                            knowledge[domain].update(turn["Item"])
                            knowledge_text = knowledge2span(knowledge)
                    
                    if turn["Agent"] == "User":
                        turn_id+=1
                        # accumulate dialogue utterances
                        dialog_history.append("<user> " + turn["Text"])
                        dialog_history_text = "".join(dialog_history[-max_history:])
                        intent = turn["active_intent"]
                        intent = API_MAP[intent]
                        # compute the Levenshtein belief spans
                        current_state = {API_MAP[k]:v for k, v in turn["state"].items()}
                        Lev = compute_lev_span(last_dialogue_state, current_state, intent)

                        state_text = state2span(last_dialogue_state,required_slots)
                        
                        input_text = "Track Dialogue State:"+ knowledge_text + "<dialogue_state> " + state_text + dialog_history_text

                        # for cross lingual transfer task
                        if args.pretraining_prefix == "en2zh_trainsfer":
                            for k,v in en_API_MAP.items():
                                input_text = input_text.replace(f" {v}",f" {k}")
                                Lev = Lev.replace(f" {v}",f" {k}")
                            for k,v in zh_en_API_MAP.items():
                                input_text = input_text.replace(f" {v}",f" {k}")
                                Lev = Lev.replace(f" {v}",f" {k}")
                            for k,v in zh2en_SLOT_MAP.items():
                                input_text = input_text.replace(f" {v}",f" {k}")
                                Lev = Lev.replace(f" {v}",f" {k}")
                            for k,v in en2zh_RELATION_MAP.items():
                                input_text = input_text.replace(f" {k}",f" {v}")
                                Lev = Lev.replace(f" {k}",f" {v}")
                            for k,v in en_zh_value_MAP.items():
                                input_text = input_text.replace(f" {k}",f" {v}")
                                Lev = Lev.replace(f" {k}",f" {v}")
                        elif args.pretraining_prefix == "zh2en_trainsfer":
                            for k,v in en_API_MAP.items():
                                input_text = input_text.replace(f" {k}",f" {v}")
                                Lev = Lev.replace(f" {k}",f" {v}")
                            for k,v in zh_en_API_MAP.items():
                                input_text = input_text.replace(f" {k}",f" {v}")
                                Lev = Lev.replace(f" {k}",f" {v}")
                            for k,v in zh2en_SLOT_MAP.items():
                                input_text = input_text.replace(f" {k}",f" {v}")
                                Lev = Lev.replace(f" {k}",f" {v}")
                            for k,v in en2zh_RELATION_MAP.items():
                                input_text = input_text.replace(f" {v}",f" {k}")
                                Lev = Lev.replace(f" {v}",f" {k}")
                            for k,v in zh_en_value_MAP.items():
                                input_text = input_text.replace(f" {k}",f" {v}")
                                Lev = Lev.replace(f" {k}",f" {v}")

                        dst_data_detail = {
                                "dial_id":dial_id,
                                "task":intent,
                                "turn_id":turn_id,
                                "dialog_history":dialog_history_text,
                                # "knowledge": copy.deepcopy(knowledge),
                                "input_text":input_text,
                                "output_text":Lev,
                                "train_target": "DST"
                                }
                        data.append(dst_data_detail)
                        last_dialogue_state = current_state

                    if turn["Agent"] == "Wizard":
                        if turn["Actions"] == "query":
                            API_flag = True
                            
                            target = f"<API> {intent}" 
                            last_API_call = target
                            API_call = ""
                        else:
                            turn_id+=1
                            # if last event is an API call
                            if API_flag:
                                API_call = last_API_call
                            else:
                                API_call = ""
                            target = turn["Text"]
                            API_flag = False
                            
                        state_text = state2span(last_dialogue_state, required_slots)
                        input_text = "Generate Response:" + knowledge_text + "<dialogue_state> " + state_text + dialog_history_text + API_call
                        input_text = input_text.strip()

                        target = target.strip()

                        if args.pretraining_prefix == "en2zh_trainsfer":
                            for k,v in en_API_MAP.items():
                                input_text = input_text.replace(f" {v}",f" {k}")
                                target = target.replace(f" {v}",f" {k}")
                            for k,v in zh_en_API_MAP.items():
                                input_text = input_text.replace(f" {v}",f" {k}")
                                target = target.replace(f" {v}",f" {k}")
                            for k,v in zh2en_SLOT_MAP.items():
                                input_text = input_text.replace(f" {v}",f" {k}")
                                target = target.replace(f" {v}",f" {k}")
                            for k,v in en2zh_RELATION_MAP.items():
                                input_text = input_text.replace(f" {k}",f" {v}")
                                target = target.replace(f" {k}",f" {v}")
                            for k,v in en_zh_value_MAP.items():
                                input_text = input_text.replace(f" {k}",f" {v}")
                                target = target.replace(f" {k}",f" {v}")
                        elif args.pretraining_prefix == "zh2en_trainsfer":
                            for k,v in en_API_MAP.items():
                                input_text = input_text.replace(f" {k}",f" {v}")
                                target = target.replace(f" {k}",f" {v}")
                            for k,v in zh_en_API_MAP.items():
                                input_text = input_text.replace(f" {k}",f" {v}")
                                target = target.replace(f" {k}",f" {v}")
                            for k,v in zh2en_SLOT_MAP.items():
                                input_text = input_text.replace(f" {k}",f" {v}")
                                target = target.replace(f" {k}",f" {v}")
                            for k,v in en2zh_RELATION_MAP.items():
                                input_text = input_text.replace(f" {v}",f" {k}")
                                target = target.replace(f" {v}",f" {k}")
                            for k,v in zh_en_value_MAP.items():
                                input_text = input_text.replace(f" {k}",f" {v}")
                                target = target.replace(f" {k}",f" {v}")
                        data_detail = {
                                "dial_id":dial_id,
                                "task":intent,
                                "turn_id":turn_id,
                                "dialog_history":dialog_history_text,
                                # "knowledge": copy.deepcopy(knowledge),
                                "input_text":input_text,
                                "output_text":target,
                                "train_target": "response"
                                }


                        data.append(data_detail)
                        # accumulate dialogue utterances
                        if not API_flag:
                            dialog_history.append("<system> " + target)
                    
    print(data[:5])
    return data




def prepare_data(args, tokenizer, max_history=3, test_only=False):
    # "en, zh, en&zh, en2zh, zh2en"

    if args.setting in ["en", "zh2en"]:
        path_train = ["data/en_train.json"]
        path_dev = ["data/en_valid.json"]
        path_test = ["data/en_test.json"]
    elif args.setting in ["zh", "en2zh"]:
        path_train = ["data/zh_train.json"]
        path_dev = ["data/zh_valid.json"]
        path_test = ["data/zh_test.json"]
    else:
        path_train = ["data/zh_train.json", "data/en_train.json"]
        path_dev = ["data/zh_valid.json", "data/en_valid.json"]
        path_test = ["data/zh_test.json", "data/en_test.json"]

    if test_only:
        data_test = read_data(args, path_test, tokenizer, max_history)
        return data_test
    else:
        data_train = read_data(args, path_train, tokenizer, max_history)
        data_dev = read_data(args, path_dev, tokenizer, max_history)
        if args.setting == "en_zh":
            random.shuffle(data_train)
            random.shuffle(data_dev)
        return data_train, data_dev



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="data/preprocessed", help="path to save prerpocessed data for training")
    parser.add_argument("--tokenizer_name", type=str, default="google/mt5-small", help="tokenizer name")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True)
    parser.add_argument("--setting", type=str, default="en", help="en, zh, en_zh, en2zh, zh2en")
    parser.add_argument("--pretraining_prefix", type=str, default="", help="for cross lingual pretrainings: [en2zh_trainsfer, zh2en_trainsfer]")
    parser.add_argument("--max_history", type=int, default=3)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=args.use_fast_tokenizer
    )
    # test
    num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': ["<user>", "<system>", "<API>", "<knowledge>", "<slot>", "<relation>", "<value>", "<sep>", "<unknow>", "<dialogue_state>"]})

    # model_inputs = tokenizer("hotels search<user> Hey, can you help me with hotel booking?")
    # print(model_inputs["input_ids"])
    # input_text = tokenizer.batch_decode([model_inputs["input_ids"]])[0]
    # print(input_text)
    # exit(0)

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        tokenizer.src_lang = "en_XX"

    data_train, data_dev = prepare_data(args, tokenizer, max_history=args.max_history)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open(os.path.join(args.save_dir, f"{args.pretraining_prefix}{args.setting}_train.json"), "w") as f:
        json.dump({"version":1.0, "data":data_train}, f, indent=True, ensure_ascii=False)

    with open(os.path.join(args.save_dir,f"{args.pretraining_prefix}{args.setting}_valid.json"), "w") as f:
        json.dump({"version":1.0, "data":data_dev}, f, indent=True, ensure_ascii=False)
