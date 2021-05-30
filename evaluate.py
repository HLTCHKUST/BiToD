import os, random
from preprocess import knowledge2span, prepare_data, read_require_slots, state2span, zh2en_API_MAP, en2zh_RELATION_MAP, en_API_MAP, API_MAP
from knowledgebase import api
from datasets import load_metric
from collections import defaultdict
import numpy as np
import argparse
import json
import torch
import copy
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

r_en_API_MAP = {v:k for k, v in en_API_MAP.items()}

def is_int(val):
    try:
        num = int(val)
    except ValueError:
        return False
    return True

def span2dict(api_span, api_names):
    # convert text span to state dict
    
    state = {}
    api_list = api_span.split("<") # add dummy

    for word in api_list:
        if word.startswith("API"):
            api_name = word.replace("API> ", "")
            if api_name in api_names:
                state[api_name] = {}
            slot = None
            relation = None
            value = None
        elif word.startswith("slot"):
            if value:
                value[0] = int(value[0]) if is_int(value[0]) else value[0]
                try:
                    state[api_name][slot] = {"relation":relation, "value":value}
                except:
                    pass
            slot = word.replace("slot> ", "")
        elif word.startswith("relation"):
            relation = word.replace("relation> ", "")
            value = []
        elif word.startswith("value"):
            v = word.replace("value> ", "")
            value.append(v)
    # last one
    try:
        if value:
            value[0] = int(value[0]) if is_int(value[0]) else value[0]
            state[api_name][slot] = {"relation":relation, "value":value}
    except:
        print(f"FAILED: api:{api_name}, slot:{slot}, relation:{relation}, value:{value}")

    return state


def state2api(dict_data):
    # convert the dictionary in the data to api
    constraints = {}
    for slot, r_v in dict_data.items():
        if r_v["value"]==["don't care"] or r_v["value"]==["不在乎"]:
            continue
        relation = r_v["relation"]
        values = r_v["value"]
        if relation != "one_of" and relation != en2zh_RELATION_MAP["one_of"]:
            values = values[0]
        if relation=="one_of" or relation==en2zh_RELATION_MAP["one_of"]:
            constraints[slot] = api.is_one_of(values)
        elif relation=="at_least" or relation==en2zh_RELATION_MAP["at_least"]:
            constraints[slot] = api.is_at_least(values)
        elif relation=="not" or relation==en2zh_RELATION_MAP["not"]:
            constraints[slot] = api.is_not(values)
        elif relation=="less_than" or relation==en2zh_RELATION_MAP["less_than"]:
            constraints[slot] = api.is_less_than(values)
        else:
            constraints[slot] = api.is_equal_to(values)
    return constraints

def dict2api(dict_data):
    # convert the dictionary in the data to api
    constraints = {}
    for const in dict_data:
        for slot, values in const.items():
            relation = values[values.find(".")+1:values.find("(")]
            values = values[values.find("(")+1:-1]
            # for one of
            
            if relation=="one_of" or relation==en2zh_RELATION_MAP["one_of"]:
                # check this
                values = values.split(" , ")
            else:
                values = int(values) if is_int(values) else values
            if relation=="one_of" or relation==en2zh_RELATION_MAP["one_of"]:
                constraints[slot] = api.is_one_of(values)
            elif relation=="at_least" or relation==en2zh_RELATION_MAP["at_least"]:
                constraints[slot] = api.is_at_least(values)
            elif relation=="not" or relation==en2zh_RELATION_MAP["not"]:
                constraints[slot] = api.is_not(values)
            elif relation=="less_than" or relation==en2zh_RELATION_MAP["less_than"]:
                constraints[slot] = api.is_less_than(values)
            else:
                constraints[slot] = api.is_equal_to(values)
    return constraints


def generate_dst(args, tokenizer, test_data, dst_model=None):
    # knowledge_text + "<dialogue_state> " + state_text + dialog_history_text
    # knowledge_text + "<dialogue_state> " + state_text + dialog_history_text + API_call
    predictions = {}
    required_slots = read_require_slots()
    required_slots = {API_MAP[k]:v for k, v in required_slots.items()}
    api_names = list(required_slots.keys())
    API_CALL = False

    for sample in test_data:
        if sample["dial_id"] not in predictions:
            predictions[sample["dial_id"]]={"turns":{}, "API":{}}
            knowledge = {}
            knowledge_text = "<knowledge>"
            dialogue_state = {}
        
        # DST
        if sample["turn_id"]%2==0 and sample["train_target"]=="response":
            try:
                state_text = state2span(dialogue_state,required_slots)
            except:
                print(lev)
                print(state_update)
                print(dialogue_state)
                
            input_text = knowledge_text + "<dialogue_state> " + state_text + sample["dialog_history"]
            input_batch = tokenizer(input_text, return_tensors="pt", verbose=False)
            input_ids = input_batch["input_ids"]

            lev_outputs = dst_model.generate(input_ids=input_ids.cuda() if torch.cuda.is_available() else input_ids,
                                    # eos_token_id=tokenizer.eos_token_id,
                                    max_length=200,
                                    num_beams=args.num_beams
                                    )
            
            lev_batch = tokenizer.batch_decode(lev_outputs)
            lev = lev_batch[0]

            lev = lev.replace("<pad>", "")
            lev = lev.replace("</s>", "")
            # print(input_text)
            # print(lev)
            state_update = span2dict(lev, api_names)
            # print(state_update)
            for api_name in state_update:
                if api_name not in dialogue_state:
                    dialogue_state[api_name] = state_update[api_name]
                else:
                    dialogue_state[api_name].update(state_update[api_name])

            predictions[sample["dial_id"]]["turns"][str(sample["turn_id"])] = {}
            to_record = copy.deepcopy(dialogue_state)
            to_record = {r_en_API_MAP.get(k, k):v for k, v in to_record.items()}
            predictions[sample["dial_id"]]["turns"][str(sample["turn_id"])]["state"] = to_record

            if API_CALL:
                api_name = sample["task"]
                # print(dialogue_state)
                if api_name in dialogue_state:
                    constraints= state2api(dialogue_state[api_name])
                    try:
                        msg = api.call_api(
                            r_en_API_MAP.get(api_name,api_name),
                            constraints=[constraints],
                        )
                    except:
                        print("failed API call: ", constraints)
                        msg = [0,0]
                    # print()
                    # print(constraints)
                    # print(msg)
                    domain = api_name.split("_")[0]
                    if domain not in knowledge:
                        knowledge[domain] = {}
                    
                    if int(msg[1]) <= 0:
                        knowledge_text = f"<knowledge> [{domain}] Message = No item avaiable."
                    else:
                        knowledge[domain].update(msg[0])
                        knowledge_text = knowledge2span(knowledge)
                
                API_CALL = False
        else:
            API_CALL = True


    return predictions


def generate_e2e(args, tokenizer, test_data, r_model=None, dst_model=None):
    # knowledge_text + "<dialogue_state> " + state_text + dialog_history_text
    # knowledge_text + "<dialogue_state> " + state_text + dialog_history_text + API_call
    predictions = {}
    required_slots = read_require_slots()
    required_slots = {API_MAP[k]:v for k, v in required_slots.items()}
    api_names = list(required_slots.keys())
    for sample in test_data:
        if sample["dial_id"] not in predictions:
            predictions[sample["dial_id"]]={"turns":{}, "API":{}}
            knowledge = {}
            knowledge_text = "<knowledge>"
            dialogue_state = {}

        # We only evaluate System turn
        if sample["turn_id"]%2==0 and sample["train_target"]=="response":
            # DST
            if args.eval_task in ["end2end", "dst"]:
                try:
                    state_text = state2span(dialogue_state,required_slots)
                except:
                    print(lev)
                    print(state_update)
                    print(dialogue_state)
                    
                input_text = "Track Dialogue State:"+ knowledge_text + "<dialogue_state> " + state_text + sample["dialog_history"]
                input_batch = tokenizer(input_text, return_tensors="pt", verbose=False)
                input_ids = input_batch["input_ids"]

                lev_outputs = dst_model.generate(input_ids=input_ids.cuda() if torch.cuda.is_available() else input_ids,
                                        # eos_token_id=tokenizer.eos_token_id,
                                        max_length=200,
                                        num_beams=args.num_beams
                                        )
                
                lev_batch = tokenizer.batch_decode(lev_outputs)
                lev = lev_batch[0]
                lev = lev.replace("<s>", "")
                lev = lev.replace("<pad>", "")
                lev = lev.replace("</s>", "")
                lev = lev.strip()
                # print(sample["dialog_history"])
                # print(lev)
                try:
                    state_update = span2dict(lev, api_names)
                except:
                    print(f"Invalid Lev span:{lev}")
                    state_update = {}
                     
                for api_name in state_update:
                    active_api = api_name
                    if api_name not in dialogue_state:
                        dialogue_state[api_name] = state_update[api_name]
                    else:
                        dialogue_state[api_name].update(state_update[api_name])

                predictions[sample["dial_id"]]["turns"][str(sample["turn_id"])] = {}
                to_record = copy.deepcopy(dialogue_state)
                to_record = {r_en_API_MAP.get(k, k):v for k, v in to_record.items()}
                predictions[sample["dial_id"]]["turns"][str(sample["turn_id"])]["state"] = to_record
                                

            if args.eval_task in ["end2end", "response"]:
                # response
                state_text = state2span(dialogue_state,required_slots)
                input_text = "Generate Response:" + knowledge_text + "<dialogue_state> " + state_text + sample["dialog_history"]

                input_batch = tokenizer(input_text, return_tensors="pt", verbose=False)
                input_ids = input_batch["input_ids"]

                response_outputs = r_model.generate(input_ids=input_ids.cuda() if torch.cuda.is_available() else input_ids,
                                        # eos_token_id=tokenizer.eos_token_id,
                                        max_length=200,
                                        num_beams=args.num_beams
                                        )
                
                response_batch = tokenizer.batch_decode(response_outputs)
                response = response_batch[0]
                response = response.replace("<s>", "")
                response = response.replace("<pad>", "")
                response = response.replace("</s>", "")
                response = response.strip()
                # print(sample["dialog_history"])
                # print(response)
                # print(input_text)
                # print(response)
                # if api call
                if response.startswith('<API>'):
                    try:
                        API_call = '<API> ' + active_api
                        api_name = active_api
                    except:
                        active_api = response.replace('<API> ', "")
                        API_call = '<API> ' + active_api
                        api_name = active_api
                    # response.replace('<API> ', "")
                    if api_name in dialogue_state:
                        constraints= state2api(dialogue_state[api_name])

                        predictions[sample["dial_id"]]["API"][r_en_API_MAP.get(api_name, api_name)] = copy.deepcopy(constraints)

                        # print(constraints, api_name)
                        try:
                            msg = api.call_api(
                                r_en_API_MAP.get(api_name, api_name),
                                constraints=[constraints],
                            )
                        except:
                            print("API_name: ", api_name)
                            print("failed API call: ", constraints)
                            msg = [0,0]

                        domain = api_name.split(" ")[0]
                        if domain not in knowledge:
                            knowledge[domain] = {}
                        
                        if int(msg[1]) <= 0:
                            knowledge_text = f"<knowledge> [{domain}] Message = No item avaiable."
                        else:
                            knowledge[domain].update(msg[0])
                            knowledge_text = knowledge2span(knowledge)
                    # print(msg)
                    # print(knowledge_text)
                    # exit(0)
                    # except:
                    #     print("Failed to parse API.")
                    input_text = "Generate Response:" + knowledge_text + "<dialogue_state> " + state_text + sample["dialog_history"] + API_call
                    input_text = input_text.strip()
                    input_batch = tokenizer(input_text, return_tensors="pt", verbose=False)
                    input_ids = input_batch["input_ids"]

                    outputs = r_model.generate(input_ids=input_ids.cuda() if torch.cuda.is_available() else input_ids,
                                            # eos_token_id=tokenizer.eos_token_id,
                                            max_length=200,
                                            num_beams=args.num_beams
                                            )

                    response_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    response = response_batch[0]
                    response = response.replace("<pad>", "")
                    response = response.replace("</s>", "")
                    # print(input_text)
                    # print(response)


                if str(sample["turn_id"]) not in predictions[sample["dial_id"]]["turns"]:
                    predictions[sample["dial_id"]]["turns"][str(sample["turn_id"])] = {}
                predictions[sample["dial_id"]]["turns"][str(sample["turn_id"])]["response"] = response
                predictions[sample["dial_id"]]["turns"][str(sample["turn_id"])]["dialog_history"] = sample["dialog_history"].split("<user> ")[-1]
        
    return predictions



metric = load_metric("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_bleu(eval_preds):
    """
    input: (preds, labels)
    preds = [pred1, pred2,...]
    labels = [label1, label2,...]
    """
    preds, labels = eval_preds

    # Some simple post-processing
    preds, labels = postprocess_text(preds, labels)
    
    result = metric.compute(predictions=preds, references=labels)
    result = {"bleu": result["score"]}

    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    # result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def compute_success_rate(predictions, references):
    """
    Success: 
    The system is able to offer the correct entities (e.g., restaurant name), provide the correct information (e.g., restaurant address),
    and confirm the booking information with the user before booking.

    Api call Accuracy:
    The predicted api call match the annotated api call.
    """

    total_dial = 0
    total_api_call = 0
    success_dial = 0
    correct_api_call = 0
    task_info = {}

    for dial_id in references:
        responses = ""
        total_dial += 1
        # api accuracy
        for api_name, constraints in references[dial_id]["API"].items():
            total_api_call += 1
            if predictions[dial_id]["API"].get(api_name) == constraints:
                correct_api_call += 1
            # else:
            #     print("pred: ", predictions[dial_id]["API"].get(api_name))
            #     print("gold: ", constraints)

        # success
        dial_success_flag = True
        for response in predictions[dial_id]["turns"].values():
            responses += response["response"] + " "
        
        for intent in references[dial_id]["tasks"]:
            if intent not in task_info:
                task_info[intent] = {"total":0, "hit":0, "success_rate":0}
            task_success_flag = True
            task_info[intent]["total"]+=1
            
            for entity in references[dial_id]["tasks"][intent]["inform+offer"]:
                if str(entity) not in responses:
                    # mt5 cannot generate chinese comma
                    if str(entity).replace("，", ",") in responses:
                        continue
                    # print(str(entity).replace("，", ","))
                    # print(responses)
                    task_success_flag = False
                    break
            for entity in references[dial_id]["tasks"][intent]["confirmation"]:
                if str(entity) not in responses:
                    # mt5 cannot generate chinese comma
                    if str(entity).replace("，", ",") in responses:
                        continue
                    # print(str(entity).replace("，", ","))
                    # print(responses)
                    task_success_flag = False
                    break
            if task_success_flag:
                task_info[intent]["hit"]+=1
            else:
                dial_success_flag = False

        # for entity in references[dial_id]["inform+offer"]:
        #     if str(entity) not in responses:
        #         success_flag = False
        #         break
        # for entity in references[dial_id]["confirmation"]:
        #     if str(entity) not in responses:
        #         success_flag = False
        #         break
        if dial_success_flag:
            success_dial += 1

    total_tasks = 0
    success_tasks = 0
    for task in task_info:
        task_info[task]["success_rate"] = task_info[task]["hit"]/task_info[task]["total"]
        total_tasks +=task_info[task]["total"]
        success_tasks += task_info[task]["hit"]
    task_info["Averaged_task_success"] = success_tasks/total_tasks
    success_rate = success_dial/total_dial
    api_acc = correct_api_call/total_api_call
    return success_rate, api_acc, task_info

def compute_result(args, predictions, reference_data):
    bleu, success_rate, api_acc, JGA, task_info = 0, 0, 0, 0, 0

    if args.eval_task in ["dst","end2end"]:
        total = 0
        hit = 0
        for dial_id in reference_data:
            turn_id = 0
            for turn in reference_data[dial_id]["Events"]:
                if turn["Agent"] == "User":
                    turn_id+=2
                    total += 1
                    predictions[dial_id]["turns"][str(turn_id)]["gold_state"] = turn["state"]
                    if predictions[dial_id]["turns"][str(turn_id)]["state"] == turn["state"]:
                        hit += 1
                    # else:
                    #     print("predict:")
                    #     print(predictions[dial_id]["turns"][str(turn_id)]["state"])
                    #     print("gold:")
                    #     print(turn["state"])
        JGA = hit/total
        print(f"JGA: {JGA}")
        # update with goal label
        with open(os.path.join(args.result_path, f"{args.save_prefix}{args.setting}_{args.eval_task}_predictions.json"), "w") as f:
            json.dump(predictions, f, indent=4, ensure_ascii=False)
        
    if args.eval_task in ["end2end", "response"]:
        reference_task_success = defaultdict(dict)
        reference_response = []
        predicted_response = []
        for dial_id in reference_data:
            if dial_id not in reference_task_success:
                reference_task_success[dial_id]["tasks"]={task["Task"]:{"inform+offer":[], "confirmation":[]} for task in reference_data[dial_id]["Scenario"]["WizardCapabilities"]}
                reference_task_success[dial_id]["API"] = {}
            turn_id = 0
            user_requested_info = defaultdict(dict)
            confirm_info = defaultdict(dict)
            for turn in reference_data[dial_id]["Events"]:
                if turn["Agent"] == "User":
                    intent = turn["active_intent"]
                if turn["Agent"] == "Wizard":
                    if turn["Actions"] == "query":
                        reference_task_success[dial_id]["API"][turn["API"]] = dict2api(turn["Constraints"])
                    else:
                        turn_id+=2
                        reference_response.append(turn["Text"])
                        
                        if intent in zh2en_API_MAP.keys():
                            # mt5 cannot generate chinese comma
                            predictions[dial_id]["turns"][str(turn_id)]["response"] = predictions[dial_id]["turns"][str(turn_id)]["response"].replace(",", "，")
                        predicted_response.append(predictions[dial_id]["turns"][str(turn_id)]["response"])


                        # For each task, the last value for each slot are considered as final requested information from user
                        for action in turn["Actions"]:
                            if (action["act"] in ["inform", "offer"]) and (len(action["value"])>0) and action["slot"]!="available_options" and action["slot"]!="可用选项":
                                user_requested_info[intent][action["slot"]] = action["value"]
                            elif (action["act"] == "confirm") and (len(action["value"])>0):
                                confirm_info[intent][action["slot"]] = action["value"]
            for intent, slot_values in user_requested_info.items():
                for values in slot_values.values():
                    reference_task_success[dial_id]["tasks"][intent]["inform+offer"] += values
            for intent, slot_values in confirm_info.items():
                for values in slot_values.values():
                    reference_task_success[dial_id]["tasks"][intent]["confirmation"] += values
                        
        
        bleu = compute_bleu((predicted_response, reference_response))
        success_rate, api_acc, task_info = compute_success_rate(predictions, reference_task_success)

        print(f"BLEU: {bleu}, DIAL_SUCCESS_RATE: {success_rate}, API_ACC: {api_acc}, task_info: {task_info}")
        
    return bleu, success_rate, api_acc, JGA, task_info

def eval_model(args, reference_data):

    if args.model_path:
        config = AutoConfig.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=args.use_fast_tokenizer)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, config=config)
        if model.config.decoder_start_token_id is None:
            model.config.decoder_start_token_id = model.config.bos_token_id
        if torch.cuda.is_available():
            model = model.cuda()
    
    data_test = prepare_data(args, tokenizer, max_history=args.max_history, test_only=True)

    if args.eval_task=="end2end":
        predictions = generate_e2e(args, tokenizer, data_test, model, model)
    
    if args.eval_task=="response":
        predictions = generate_e2e(args, tokenizer, data_test, r_model = model)
    
    if args.eval_task=="dst":
        predictions = generate_dst(args, tokenizer, data_test, dst_model = model)

    with open(os.path.join(args.result_path, f"{args.save_prefix}{args.setting}_{args.eval_task}_predictions.json"), "w") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

    bleu, success_rate, api_acc, JGA, task_info = compute_result(args, predictions, reference_data)
    
    

    return bleu, success_rate, api_acc, JGA, task_info

def eval_file(args, reference_data):
    with open(args.prediction_file_path) as f:
        predictions = json.load(f)
    
    bleu, success_rate, api_acc, JGA, task_info = compute_result(args, predictions, reference_data)

    return bleu, success_rate, api_acc, JGA, task_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_file_path", type=str, default="data/test.json", help="path of reference")
    parser.add_argument("--prediction_file_path", type=str, default="data/test.json", help="path of prediction")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True)
    parser.add_argument("--pretraining_prefix", type=str, default="", help="for cross lingual pretrainings: [en2zh_trainsfer, zh2en_trainsfer]")
    parser.add_argument("--lang", type=str, default="en_XX")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--max_history", type=int, default=3)
    parser.add_argument("--eval_mode", type=str, default="eval_model", help="eval_model or eval_file?")
    parser.add_argument("--eval_task", type=str, default="end2end", help="end2end, dst, response")
    parser.add_argument("--setting", type=str, default="en", help="en, zh, en&zh, en2zh, zh2en")
    parser.add_argument("--result_path", type=str, default="result", help="eval_model or eval_file?")
    parser.add_argument("--num_beams", type=int, default=1, help="use greedy is num_beams==1")
    parser.add_argument("--save_prefix", type=str, default="", help="prefix of save file name")
    
    args = parser.parse_args()

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    reference_data = {}
    for reference_file_path in args.reference_file_path.split("__"):
        with open(reference_file_path) as f:
            reference_data.update(json.load(f))
    if args.eval_mode=="eval_model":
        bleu, success_rate, api_acc, JGA, task_info = eval_model(args, reference_data)
    else:
        bleu, success_rate, api_acc, JGA, task_info = eval_file(args, reference_data)
    
    with open(os.path.join(args.result_path, f"{args.save_prefix}{args.setting}_{args.eval_task}_result.json"), "w") as f:
        json.dump({"BLEU":bleu, "dialogue_success_rate":success_rate, "API_ACC":api_acc, "JGA":JGA, "task_info": task_info}, f, indent=4, ensure_ascii=False)
        # f.write(f"BLEU: {bleu}, DIAL_SUCCESS_RATE: {success_rate}, API_ACC: {api_acc}, JGA: {JGA}")
        # json.dump(f"BLEU: {bleu}, SUCCESS_RATE: {success_rate}, API_ACC: {api_acc}", f)


if __name__ == "__main__":
    # args = get_args()

    main()


