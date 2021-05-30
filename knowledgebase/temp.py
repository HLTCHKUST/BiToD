import json

with open("apis/HKMTR_en.json") as f:
    api_en = json.load(f)
with open("apis/HKMTR_zh.json") as f:
    api_zh = json.load(f)
with open("dbs/HKMTR_en.json") as f:
    dbs_en = json.load(f)
with open("dbs/HKMTR_zh.json") as f:
    dbs_zh = json.load(f)

for x in api_en["input"]:
    x["Categories"].sort()
for x in api_zh["input"]:
    x["Categories"].sort()
for x in dbs_en:
    x["Categories"].sort()
for x in dbs_zh:
    x["Categories"].sort()

with open("apis/HKMTR_en.json", "w") as f:
    json.dump(api_en, f, ensure_ascii=False, indent=4)

with open("apis/HKMTR_zh.json", "w") as f:
    json.dump(api_zh, f, ensure_ascii=False, indent=4)
with open("dbs/HKMTR_en.json", "w") as f:
    json.dump(dbs_en, f, ensure_ascii=False, indent=4)
with open("dbs/HKMTR_zh.json", "w") as f:
    json.dump(dbs_zh, f, ensure_ascii=False, indent=4)

