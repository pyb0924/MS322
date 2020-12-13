import json
import pandas as pd


if __name__ == '__main__':
    with open('result.json','r')as f:
        records=json.load(f)

    output=[]
    for record in records:
        if not record['problem-type']=='binary':
            output.append(record)

    with open('result1.json','w+',encoding='utf-8')as f1:
        json.dump(output,fp=f1,ensure_ascii=False,indent=4,sort_keys=True)