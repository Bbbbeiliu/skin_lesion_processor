import requests
import json

url = "http://127.0.0.1:5000/submit"
# 若连接同一局域网下的电脑，则修改url 为 http://被控端IP:5000/submit
# url = "http://10.180.239.92:5000/submit"
file_path = r"D:/Appdevelop/.venv/skin_lesion_processor/resources/test_id4.dxf"   # 待加工文件

# 构造模板参数（根据实际需求修改）
params = {
    "mode": "template",                          # 模板模式（可选 "direct"）
    "template_path": "D:/Appdevelop/.venv/skin_lesion_processor/resources/template/template.ezd", # 模板文件绝对路径
    # "placeholder": "PLACEHOLDER"                 # 模板中占位对象的名称
}

# 可选：如果不使用模板，可以传空字典或省略 params
# params = {}

with open(file_path, "rb") as f:
    files = {"file": f}
    data = {"params": json.dumps(params)}   # 将字典转为 JSON 字符串
    response = requests.post(url, files=files, data=data)

print(response.json())