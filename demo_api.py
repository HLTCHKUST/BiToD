from knowledgebase import api
import json



msg = api.call_api(
    "HKMTR_en",
    constraints=[{
        "departure": api.is_equal_to("University"),
        "destination": api.is_equal_to("Kowloon Bay"),
    }],
)
print(json.dumps(msg))

msg = api.call_api(
    "香港地铁",
    constraints=[{
        "出发地": api.is_equal_to("銅鑼灣"),
        "目的地": api.is_equal_to("香港仔"),
    }],
)
print(json.dumps(msg,ensure_ascii=False))


msg = api.call_api(
    "restaurants_en_US_booking",
    constraints=[{
        "name": api.is_equal_to("Chilli Fagara"),
        "number_of_people": api.is_equal_to(3),
        "user_name": api.is_equal_to("Ruth"),
        "time": api.is_equal_to("4:00 pm"),
        "date": api.is_equal_to("today")
    }],
)

print(json.dumps(msg))


msg = api.call_api(
    "餐馆预订",
    constraints=[{
        "名字": api.is_equal_to("Brick Lane"),
        "人数": api.is_equal_to(3),
        "用户名": api.is_equal_to("Ruth"),
        "时间": api.is_equal_to("上午 4:00"),
        "预订日期": api.is_equal_to("today")
    }],
)

print(json.dumps(msg,ensure_ascii=False))
