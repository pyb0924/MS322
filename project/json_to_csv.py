import json
import pandas as pd
import numpy as np


if __name__ == '__main__':
    with open('result.json','r')as f:
        data=json.load(f)
    df=pd.DataFrame(data)

    df=df.groupby([df['model'],df['problem-type']]).mean()

    col = ['valid_loss', 'iou', 'dice']
    df = df[col]

    df.to_csv('result.csv')
    print('done!')