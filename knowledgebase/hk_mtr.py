import networkx as nx
import pydot
import os
import json
fi = os.path.join(
            "knowledgebase",
            "hongkong-mtr_with_eng.dot",
        )

MTR_map_file = os.path.join(
            "knowledgebase",
            "mtr_mapping.json",
        )

with open(MTR_map_file) as f:
    MTR_map = json.load(f)



graph = pydot.graph_from_dot_file(fi)[0]
nx_graph = nx.drawing.nx_pydot.from_pydot(graph)

color = {"AE": "turquoise",
         "DR": "pink",
         "EK": "tan",
         "EW": "brown",
         "I": "blue",
         "KT": "green",
         "N": "black",
         "NS": "cyan",
         "SIE": "yellowgreen",
         "SIW": "mediumpurple",
         "TKO": "purple",
         "TW": "red",
         "TC": "orange",
         "NP":"orange"}
 
color_to_zh = {
        "turquoise":"綠松石",
        "pink":"粉色的",
        "tan":"棕褐色",
        "brown":"棕色的",
        "blue":"藍色",
        "green":"綠色",
        "black":"黑色的",
        "cyan":"青色",
        "yellowgreen":"黃綠色",
        "mediumpurple":"中紫色",
        "purple":"紫色的",
        "red":"紅色的",
        "orange":"橘子",
            }

name_line = {"AE": "Airport Express Line",
         "DR": "Disneyland Resort Line",
         "EK": "East Kowloon Line",
         "EW": "East West Line",
         "I": "Island Line",
         "KT": "Kwan Tong Line",
         "N": "Northern Line",
         "NS": "North South Line",
         "SIE": "South Island Line (East)",
         "SIW": "South Island Line (West)",
         "TKO": "Tseung Kwan O Line",
         "TW": "Tsuen Wan Line",
         "TC": "Tung Chung Line",
         "NP":"Tung Chung Line"}

name_missing = {"AE1": {"label":"博覽館\rAsiaWorld-Expo"},
                "AE2": {"label":"機場\rAirport"},
                "AE3_TC5": {"label":"青衣\rTsing Yi"},
                "AE5_TC20": {"label":"香港\rHong Kong"},
                "DR1_TC4": {"label":"欣澳\rSunny Bay"},
                "DR2": {"label":"迪士尼\rDisneyland Resort"},
                "DR2": {"label":"迪士尼\rDisneyland Resort"},
                "I1": {"label":"監尼地城\rKennedy Town"},
                "I2_SIW1":{"label":"香港大學\rHKU"},
                "I3": {"label":"西營盤\rSai Ying Pun"},
                "I4": {"label":"上環\rSheung Wan"},
                "I5_TW16":{"label":"中環\rCentral"},
                "I6_NS1_SIE1_TW15":{"label":"金鐘\rAdmirality"},
                "I7": {"label":"灣仔\rWan Chai"},
                "I8": {"label":"銅鑼灣\rCauseway Bay"},
                "I9": {"label":"天后\rTin Hau"},
                "I10":{ "label":"炮台山\rFortress Hill"},
                "I11_TKO4":{"label":"北角\rNorth Point"},
                "I12_TKO5":{"label":"鰂魚涌\rQuarry Bay"},
                "I13":{ "label":"太古\rTai Koo"},
                "I13":{ "label":"西灣河\rSai Wan Ho"},
                "I14":{ "label":"筲箕灣\rShau Kei Wan"},
                "I15":{ "label":"杏花邨\rHeng Fa Chuen"},
                "I16":{ "label":"柴灣\rChai Wan"},
                "SIE2": {"label":"海洋公園\rOcean Park"},
                "SIE3_SIW7": {"label":"黃竹坑\rWong Chuk Hang"},
                "SIE4": {"label":"利東\rLei Tung"},
                "SIE5": {"label":"海怡半島\rSouth Horizons"},
                "SIW2": {"label":"瑪麗醫院\rQueen Mary Hospital"},
                "SIW3": {"label":"數碼港\rCyberport"},
                "SIW4": {"label":"華富\rWah Fu"},
                "SIW5": {"label":"田灣\rTin Wan"},
                "SIW6": {"label":"香港仔\rAberdeen"},
                "TC21_TKO1": {"label":"添馬\rTamar"},
                "NS2_TKO2": {"label":"會展\rExhibition"},
                "TKO3": {"label":"維園\rVictoria Park"},
                "TC1": {"label":"東涌\rTung Chung West"},
                "TC2": {"label":"東涌\rTung Chung"},
                "TC3": {"label":"東涌\rTung Chung East"},
                "NP": {"label":"昂坪\rNgong Ping"}
                }

name_to_code_en = {}
name_to_code_zh = {}
for n in nx_graph.nodes():
    temp = {"name_en":"","name_zh":"", "color":[], "line":[]}
    if "label" in nx_graph.nodes[n]:
        temp["name_zh"] = nx_graph.nodes[n]["label"].split("\\r")[0].replace('"',"")
        temp["name_en"] = nx_graph.nodes[n]["label"].split("\\r")[1].replace('"',"")
    else:
        if n in name_missing:
            temp["name_zh"] = name_missing[n]["label"].split("\r")[0]
            temp["name_en"] = name_missing[n]["label"].split("\r")[1]
    name_to_code_en[temp["name_en"]] = n
    name_to_code_zh[temp["name_zh"]] = n
    temp_c = []
    for id_ in n.split("_"):
        t = [x.isdigit() for x in id_]
        i = len(t)
        if True in t:
            i = [x.isdigit() for x in id_].index(True)
        id_ = id_[:i]

        temp["color"].append(color[id_])
        temp["line"].append(name_line[id_])

    nx_graph.nodes[n].update(temp)
    # print(n, nx_graph.nodes[n])

G = nx.Graph()
G_ZH = nx.Graph()

for n in nx_graph.nodes():
    name = nx_graph.nodes[n]["name_en"]
    name_zh = nx_graph.nodes[n]["name_zh"]

    colors = nx_graph.nodes[n]["color"]
    G.add_node(f'{name}')
    G_ZH.add_node(f'{name_zh}')
    for c in colors:
        G.add_node(f'{name}_{c}')
        G_ZH.add_node(f'{name_zh}_{color_to_zh[c]}')
        G.add_edge(f'{name}', f'{name}_{c}',weight=0) 
        G_ZH.add_edge(f'{name_zh}', f'{name_zh}_{color_to_zh[c]}',weight=0) 
    for i in range(len(colors)):
        for j in range(i, len(colors)):
            G.add_edge(f'{name}_{colors[i]}', f'{name}_{colors[j]}',weight=1) 
            G_ZH.add_edge(f'{name_zh}_{color_to_zh[colors[i]]}', f'{name_zh}_{color_to_zh[colors[j]]}',weight=1) 

for s,t in nx_graph.edges():
    s_name = nx_graph.nodes[s]["name_en"]
    s_name_zh = nx_graph.nodes[s]["name_zh"]
    s_colors = nx_graph.nodes[s]["color"]
    t_name = nx_graph.nodes[t]["name_en"]
    t_name_zh = nx_graph.nodes[t]["name_zh"]
    t_colors = nx_graph.nodes[t]["color"]
    for c in list(set(s_colors).intersection(set(t_colors))):
        G.add_edge(f'{s_name}_{c}', f'{t_name}_{c}',weight=1)
        G_ZH.add_edge(f'{s_name_zh}_{color_to_zh[c]}', f'{t_name_zh}_{color_to_zh[c]}',weight=1)

# for n in G.nodes():
#     if "_" not in n:
#         print(f'"{n}",')

# print()
# for n in G_ZH.nodes():
#     if "_" not in n:
#         print(f'"{n}",')

def MTR(source = "HKU",target = "Kowloon Tong", lang="en"):
    source = MTR_map[source]
    target = MTR_map[target]
    if lang == "en":
        shortest = nx.shortest_path(G, source=source, target=target)
        price = round((len(shortest)-2)*0.88+3.4, 2)
        time = (len(shortest)-2)*3
        str_ret = ""
        str_ret += f"Take the {shortest[1].split('_')[1]} line of the {shortest[0]} station."
        for i in range(2,len(shortest)-1):
            name, color = shortest[i].split("_")
            name_prev, color_prev = shortest[i-1].split("_")
            if color != color_prev and name == name_prev:
                str_ret += f"Then change at {name} station from {color_prev} line to {color} line."
        str_ret += f"Get off the train at {shortest[-1]} station."
        return {"shortest_path":str_ret,"price":f"{price} HKD","estimated_time":f"{time} mins"}
    else:
        shortest = nx.shortest_path(G_ZH, source=source, target=target)
        price = round((len(shortest)-2)*0.88+3.4, 2)
        time = (len(shortest)-2)*3
        str_ret = ""
        str_ret += f"请在{shortest[0]}站乘坐{shortest[1].split('_')[1]}线，"
        for i in range(2,len(shortest)-1):
            name, color = shortest[i].split("_")
            name_prev, color_prev = shortest[i-1].split("_")
            if color != color_prev and name == name_prev:
                str_ret += f"然后在{name}站换乘{color}线，"
        str_ret += f"最后在{shortest[-1]}站下车"
        return {"最短路线":str_ret,"价格":f"{price}港币","预估时间":f"{time}分钟"}

# print(MTR("銅鑼灣","香港仔"))