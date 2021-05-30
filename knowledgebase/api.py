import json
import os
import random
import re

from dateutil.parser import parse
# noinspection PyUnresolvedReferences
from itertools import combinations
from typing import Dict, Text, Any, List, Optional, Tuple, Union
from pymongo import MongoClient
from parlai.mturk.tasks.woz.knowledgebase.hk_mtr import MTR
import pymongo
client = MongoClient()
dblist = client.list_database_names()
mydb = client["bilingual_tod"]

zh2en_SLOT_MAP = {"名字": "name", "位置":"location", "评分":"rating", "类别":"type", "地址":"address", "电话":"phone_number", "可用选项":"available_options", "参考编号":"ref_number", "每晚价格":"price_per_night", "用户名":"user_name", "房间数":"number_of_rooms", "预订月":"start_month", "预订日":"start_day", "预订日期":"start_date", "预订天数":"number_of_nights", "价格范围":"price_level", "星级数":"stars", "菜品":"cuisine", "饮食限制":"dietary_restrictions", "日期":"day", "城市":"city", "最高温度":"max_temp", "最低温度":"min_temp", "天气":"weather", "描述":"description", "出发地":"departure", "目的地":"destination", "人数":"number_of_people", "时间":"time", "预订日期":"date", "预估时间":"estimated_time", "最短路线":"shortest_path", "价格":"price"} 
zh2en_API_MAP = {"餐馆查询": "restaurants_zh_CN_search", "餐馆预订":"restaurants_zh_CN_booking", "宾馆查询": "hotels_zh_CN_search", "宾馆预订": "hotels_zh_CN_booking", "景点查询":"attractions_zh_CN_search", "天气查询":"weathers_zh_CN_search", "香港地铁":"HKMTR_zh"}
en2zh_SLOT_MAP = {v:k for k, v in zh2en_SLOT_MAP.items()}
en2zh_API_MAP = {v:k for k, v in zh2en_API_MAP.items()}
entity_map = {}
with open("knowledgebase/zh_entity_map.json") as f:
    entity_map.update(json.load(f))
with open("knowledgebase/en_entity_map.json") as f:
    entity_map.update(json.load(f))

def is_equal_to(value):
    if(is_mongo):
        return value #
    else:
        return lambda x: x == value
        
def is_not(value):
    if(is_mongo):
        return {"$ne": value} #
    else:
        # return lambda x: x == value
        return lambda x: x != value


def contains_none_of(value):
    if(is_mongo):
        return {"$nin": value} #
    else:
        return lambda x: not any([e in x for e in value])


def is_one_of(value):
    if(is_mongo):
        return {"$in":value} #
    else:
        return lambda x: x in value


def is_at_least(value):
    if(is_mongo):
        return {"$gte": value} #
    else:
        return lambda x: x >= value


def is_less_than(value):
    if(is_mongo):
        return {"$lt": value} #
    else:
        return lambda x: x < value

def is_at_most(value):
    if(is_mongo):
        return {"$lte": value} #
    else:
        return lambda x: x <= value

def contain_all_of(value):
    return lambda x: all([e in x for e in value])


def contain_at_least_one_of(value):
    return lambda x: any([e in x for e in value])



dbs = {"null": None}
dbs["restaurants_en_US_booking"] = mydb["restaurants_en_US"]
dbs["restaurants_en_US_search"] = mydb["restaurants_en_US"]
dbs["hotels_en_US_search"] = mydb["hotels_en_US"]
dbs["hotels_en_US_booking"] = mydb["hotels_en_US"]
dbs["attractions_en_US_search"] = mydb["attractions_en_US"]
dbs["weathers_en_US_search"] = mydb["weathers_en_US"]

dbs["restaurants_zh_CN_booking"] = mydb["restaurants_zh_CN"]
dbs["restaurants_zh_CN_search"] = mydb["restaurants_zh_CN"]
dbs["hotels_zh_CN_search"] = mydb["hotels_zh_CN"]
dbs["hotels_zh_CN_booking"] = mydb["hotels_zh_CN"]
dbs["attractions_zh_CN_search"] = mydb["attractions_zh_CN"]
dbs["weathers_zh_CN_search"] = mydb["weathers_zh_CN"]


is_mongo = True


def constraint_and(constraint1, constraint2):
    return lambda x: constraint1(x) and constraint2(x)


def constraint_list_to_dict(constraints: List[Dict[Text, Any]]) -> Dict[Text, Any]:
    result = {}
    for constraint in constraints:
        for name, constraint_function in constraint.items():
            # print(name, callable(constraint_function))
            if name not in result:
                result[name] = constraint_function
            else:
                result[name] = constraint_and(result[name], constraint_function)
    return result
    
def restaurants_en_US_booking(api_name,db,query,api_out_list=None):
    pre_api_return = {"user_name": query["user_name"], "number_of_people": query["number_of_people"], "time": query["time"]}
    api_out_list.remove("number_of_people")
    api_out_list.remove("time")
    api_out_list.remove("date")
    if "date" in query:
        pre_api_return["date"] = query["date"]
        del query["date"]
    del query["user_name"]
    query["max_num_people_book"] = {"$gte": query["number_of_people"]}
    del query["number_of_people"]
    
    # print("before: {}".format(query["time"]))
    temp = int(query["time"].split(":")[0])
    temp %= 12
    if "pm" in query["time"]:
        temp +=12
    mins = float(re.findall(r"[0-9][0-9]", query["time"].split(":")[1])[0])/60
    temp +=mins
    # print(f"after: {temp}")
    query["time"] = temp
    query["open_time"] = {"$lte": query["time"]}
    query["close_time"] = {"$gte": query["time"]}
    del query["time"]
    res = list(db.find(query))

    results = []
    for r in res:
        r["_id"] = str(r["_id"])
        results.append(r)
    
    if len(results)==0:

        return dict(Message="Sorry, the restaurant is not available given the booking time and number of people. The booking is failed."), len(results)
    else:
        api_return = {k: results[-1][k] for k in api_out_list}
        api_return.update(pre_api_return)
        return api_return, len(results)

def restaurants_zh_CN_booking(api_name,db,query,api_out_list=None):
    pre_api_return = {"user_name": query["user_name"], "number_of_people": query["number_of_people"], "time": query["time"], "date": query["date"]}
    api_out_list.remove("number_of_people")
    api_out_list.remove("time")
    api_out_list.remove("date")
    del query["date"]
    del query["user_name"]
    query["max_num_people_book"] = {"$gte": query["number_of_people"]}
    del query["number_of_people"]
    

    if "上午" in query["time"]:
        temp = query["time"].replace("上午", "")

    if "下午" in query["time"]:
        temp = query["time"].replace("下午", "")

    temp , m = temp.split(":")
    temp = int(temp)
    m = float(m)

    temp %= 12
    if "下午" in query["time"]:
        temp +=12
    
    mins = m/60
    temp +=mins
    # print(f"after: {temp}")
    query["time"] = temp
    query["open_time"] = {"$lte": query["time"]}
    query["close_time"] = {"$gte": query["time"]}
    del query["time"]
    res = list(db.find(query))

    results = []
    for r in res:
        r["_id"] = str(r["_id"])
        results.append(r)
    
    if len(results)==0:
        return dict(Message="对不起，预约失败。"), len(results)
    else:
        api_return = {k: results[-1][k] for k in api_out_list}
        api_return.update(pre_api_return)
        api_return = {en2zh_SLOT_MAP[k]: v for k, v in api_return.items()}
        return api_return, len(results)

def hotels_en_US_booking(api_name,db,query,api_out_list=None):
    pre_api_return = {"user_name": query["user_name"], "number_of_rooms": query["number_of_rooms"], "start_month": query["start_month"], "start_day": query["start_day"], "number_of_nights": query["number_of_nights"]}
    api_out_list.remove("number_of_rooms")
    del query["user_name"]
    query["num_of_rooms"] = {"$gte": query["number_of_rooms"]}
    del query["number_of_rooms"]
    del query["start_month"]
    del query["start_day"]
    del query["number_of_nights"]
    
    res = list(db.find(query))

    results = []
    for r in res:
        r["_id"] = str(r["_id"])
        results.append(r)
    
    if len(results)==0:

        return dict(Message="Sorry, the hotel is not available given the number of rooms. The booking is failed."), len(results)
    else:
        api_return = {k: results[-1][k] for k in api_out_list}
        api_return.update(pre_api_return)
        return api_return, len(results)

def hotels_zh_CN_booking(api_name,db,query,api_out_list=None):
    pre_api_return = {"user_name": query["user_name"], "number_of_rooms": query["number_of_rooms"], "start_month": query["start_month"], "start_day": query["start_day"], "number_of_nights": query["number_of_nights"]}
    api_out_list.remove("number_of_rooms")
    del query["user_name"]
    query["num_of_rooms"] = {"$gte": query["number_of_rooms"]}
    del query["number_of_rooms"]
    del query["start_month"]
    del query["start_day"]
    del query["number_of_nights"]
    
    res = list(db.find(query))

    results = []
    for r in res:
        r["_id"] = str(r["_id"])
        results.append(r)
    
    if len(results)==0:

        return dict(Message="对不起，预约失败。"), len(results)
    else:
        api_return = {k: results[-1][k] for k in api_out_list}
        api_return.update(pre_api_return)
        api_return = {en2zh_SLOT_MAP[k]: v for k, v in api_return.items()}
        return api_return, len(results)


def general_search_en_US(api_name,db,query,api_out_list=None):
    res = list(db.find(query).sort([("rating", pymongo.ASCENDING), ("_id", pymongo.DESCENDING)]))
    results = []
    for r in res:
        r["_id"] = str(r["_id"])
        results.append(r)
    if len(results)==0:
        return {}, len(results)
    else:
        # print(res)
        api_return = {k: results[-1][k] for k in api_out_list}
        api_return["available_options"] = len(results)
        
        if "price_per_night" in api_return:
            api_return["price_per_night"] = str(api_return["price_per_night"]) + " HKD"
        return api_return, len(results)

def general_search_zh_CN(api_name,db,query,api_out_list=None):
    res = list(db.find(query).sort([("rating", pymongo.ASCENDING), ("_id", pymongo.DESCENDING)]))
    results = []
    for r in res:
        r["_id"] = str(r["_id"])
        results.append(r)
    if len(results)==0:
        return {}, len(results)
    else:
        # print(res)
        api_return = {k: results[-1][k] for k in api_out_list}
        api_return["available_options"] = len(results)
        
        if "price_per_night" in api_return:
            api_return["price_per_night"] = str(api_return["price_per_night"]) + "港币"
        api_return = {en2zh_SLOT_MAP[k]: v for k, v in api_return.items()}
        return api_return, len(results)

def query_mongo(api_name,db,query,api_out_list=None):
    if api_name == "restaurants_en_US_booking":
        res, count = restaurants_en_US_booking(api_name,db,query,api_out_list)
    elif api_name == "hotels_en_US_booking":
        res, count = hotels_en_US_booking(api_name,db,query,api_out_list)
    elif api_name == "restaurants_zh_CN_booking":
        res, count = restaurants_zh_CN_booking(api_name,db,query,api_out_list)
    elif api_name == "hotels_zh_CN_booking":
        res, count = hotels_zh_CN_booking(api_name,db,query,api_out_list)
    elif "zh" in api_name:
        res, count = general_search_zh_CN(api_name,db,query,api_out_list)
    else:
        res, count = general_search_en_US(api_name,db,query,api_out_list)
    return res, count
    



def call_api(api_name, constraints: List[Dict[Text, Any]]) -> Tuple[Dict[Text, Any], int]:
    global is_mongo
    print(api_name)

    # Canonicalization
    for slot, value in constraints[0].items():
        if isinstance(value, str) and (value in entity_map):
            constraints[0][slot] = entity_map[value]
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, str) and (v in entity_map):
                    constraints[0][slot][k] = entity_map[v]
                if isinstance(v, list):
                    constraints[0][slot][k] = [entity_map[v_v] if v_v in entity_map else v_v for v_v in v]

    if api_name in ["restaurants_en_US_search","restaurants_en_US_booking", "hotels_en_US_search", "hotels_en_US_booking", "attractions_en_US_search", "weathers_en_US_search"]:
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "apis", api_name + ".json"
            ),
            "r"
            ) as file:
            api_schema = json.load(file)
        is_mongo = True
        if constraints:
            all_provided_parameters = set.union(*[set(c) for c in constraints])
        else:
            all_provided_parameters = set()

        for parameter in api_schema["required"]:
            if parameter not in all_provided_parameters:
                raise ValueError(
                    f"Parameter '{parameter}' is required but was not provided."
                )
                
        api_out_list = [slot["Name"] for slot in api_schema["output"]]
        api_obj = dbs[api_name]
        res, count = query_mongo(api_name, dbs[api_name], constraint_list_to_dict(constraints),api_out_list)
        # res, count = query_mongo(dbs[api_name], constraints)
        return res, count
    elif api_name in ["餐馆查询","餐馆预订", "宾馆查询", "宾馆预订", "景点查询", "天气查询"]:
        api_name = zh2en_API_MAP[api_name]
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "apis", api_name + ".json"
            ),
            "r"
            ) as file:
            api_schema = json.load(file)

        constraints = [{zh2en_SLOT_MAP[k]: v for k, v in constraints[0].items()}]
        
        is_mongo = True
        if constraints:
            all_provided_parameters = set.union(*[set(c) for c in constraints])
        else:
            all_provided_parameters = set()

        for parameter in api_schema["required"]:
            if parameter not in all_provided_parameters:
                raise ValueError(
                    f"Parameter '{parameter}' is required but was not provided."
                )
                
        api_out_list = [slot["Name"] for slot in api_schema["output"]]
        api_obj = dbs[api_name]
        res, count = query_mongo(api_name, dbs[api_name], constraint_list_to_dict(constraints),api_out_list)
        # res, count = query_mongo(dbs[api_name], constraints)
        return res, count

    elif api_name in ["HKMTR_en", "香港地铁"]:
        if api_name=="香港地铁":
            api_name = zh2en_API_MAP[api_name]
            constraints = [{zh2en_SLOT_MAP[k]: v for k, v in constraints[0].items()}]
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "apis", api_name + ".json"
            ),
            "r"
            ) as file:
            api_schema = json.load(file)

        source = constraint_list_to_dict(constraints)["departure"]
        target = constraint_list_to_dict(constraints)["destination"]
        lang = api_name.split("_")[1]
        try:
            mtr_dict = MTR(source=source,target=target, lang= lang)
        except Exception:
            return None, -1

        return mtr_dict, 1 
    else:
        raise ValueError(
                f"API'{api_name}' is not available."
            )
        
