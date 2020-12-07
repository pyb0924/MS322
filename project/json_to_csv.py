import json
import pandas as pd





if __name__ == '__main__':
    with open('test.json','r') as f:
        data=json.load(f)

    df=pd.DataFrame(data)
    df.to_csv('test.csv')
    print('done')

