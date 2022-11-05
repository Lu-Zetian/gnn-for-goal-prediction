from statsbombpy import sb
import pandas as pd
import multiprocessing as mp
import pickle
import os
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# heads=100
df_EURO = sb.matches(competition_id=55,season_id=43)
# print(df_EURO.columns)

print(df_EURO[['match_id','match_date','home_team','away_team','home_score','away_score','competition_stage','stadium']])

DataLocation = r'C:\Users\scs20\OneDrive - HKUST Connect\2022-23_Fall\COMP4222\Project\comp4222-project\DataDownload'

from DataParsing import DataParsing 

def mp_construct(match_id):
    e = pd.read_pickle(os.path.join(DataLocation,f'{match_id}_events.pkl'))
    f = pd.read_pickle(os.path.join(DataLocation,f'{match_id}_frames.pkl'))
    m = DataParsing.Match(match_id,match_df=df_EURO,event_df=e,frame_df=f)
    return m
def parsing():
    print('start')
    print(len(df_EURO['match_id']))
    with mp.Pool() as pool:
        Matches = pool.map(mp_construct,df_EURO['match_id'])
    # Matches.get()
    with open("data_2.pickle",'wb') as f:
        pickle.dump(Matches,f,protocol=pickle.HIGHEST_PROTOCOL)
    print(Matches)
    print('done')

# if __name__=='__main__':

import pickle,time
st = time.time()
# objects = []
os.chdir(os.path.dirname(os.path.abspath(__name__)))
with (open('data_2.pickle','rb')) as openfile:  # read the parsed data
    print('reading')
    while True:
        try:
            data=pickle.load(openfile)
        except EOFError:
            break
    print('loaded')
et = time.time()
print(et-st) #loading time

print('match:\t',data[0].match_id)
print('Label:\t',data[0].gamestates[0].label)
print('Graph:\t',data[0].gamestates[0].graph)
print('Meta: \t',data[0].gamestates[0].metadata)        

print('match:\t',data[1].match_id)
print('Label:\t',data[1].gamestates[10].label)
print('Graph:\t',data[1].gamestates[10].graph)
print('Meta: \t',data[1].gamestates[10].metadata)        
    