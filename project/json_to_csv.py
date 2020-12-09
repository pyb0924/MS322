import json
import pandas as pd


if __name__ == '__main__':
    with open('result.json','r')as f:
        data=json.load(f)
    df=pd.DataFrame(data)

    df=df.groupby([df['model'],df['problem-type']]).mean()
    dice_list = ['dice_' + str(i) for i in range(1, 8)]
    df['dice'] = df[dice_list].mean(axis=1)

    col = ['jaccard_loss', 'valid_loss', 'iou', 'dice']
    df = df[col]

    df.to_csv('result.csv')
    print('done!')