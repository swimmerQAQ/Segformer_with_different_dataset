import os
import json
with open("./namespace.txt", "r") as f:  # 打开文件
    types = f.read().splitlines()  # 读取文件
mseg_json = {}
mseg_json["mseg"] = {}
num = 0
################ txt to json
for one_type in types :
    print("one type = ",one_type)
    mseg_json["mseg"][one_type] = num
    num += 1
################  write mseg.json
temp_json = json.dumps(mseg_json)
with open("./mseg.json","w") as f :
    f.write(temp_json)
###################  read goal
with open("./cityscapes.json",'r', encoding='UTF-8') as f:
    goal_json = json.load(f)
###################### strategy
mseg2city_json = {}
mseg2city_json["mseg2city"] = {}
mseg2city_json["mseg2city"]["origin"] = "mseg"
for mseg_key in mseg_json["mseg"] :
    if mseg_key in goal_json["cityscapes"] :
        mseg2city_json["mseg2city"][mseg_key] = mseg_key
    else :
        mseg2city_json["mseg2city"][mseg_key] = "None"
temp_json = json.dumps(mseg2city_json)
with open("./mseg2city.json","w") as f :
    f.write(temp_json)