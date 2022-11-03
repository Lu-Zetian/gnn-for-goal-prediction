from asyncio import current_task
from multiprocessing import Event
import pandas as pd
from statsbombpy import sb
import torch
from torch_geometric.data import Data
# import numpy as np
import time
import os
# import numba

class Match:
    def __init__(self,match_id,match_df,event_df,frame_df):
        print(f'Parsing {match_id}')
        self.match_id = match_id
        self.home_team = match_df.loc[match_df['match_id']==match_id,'home_team'].values[0]
        self.away_team = match_df.loc[match_df['match_id']==match_id,'away_team'].values[0]

        # self.events = sb.events(match_id=match_id)
        self.events = event_df
        self.events.sort_values(['period','timestamp'],inplace=True)
        
        # self.frames = sb.frames(match_id=match_id)
        self.frames = frame_df
        self.gamestates = []
        self.result = (
            match_df.loc[match_df['match_id']==match_id,'home_score'].values[0],match_df.loc[match_df['match_id']==match_id,'away_score'].values[0]
        )
        # self.get_gamestates[1,self.events.shape[0]]()
            
        self.get_gamestates()
        
    
    def get_gamestates(self):
        current_score = [0,0]
        # print(self.events['id'])
        # for i in range(len(self.events['id'])):
        for id in self.events['id']:
            # id = self.events.at[i,'id']
            if self.events.loc[self.events['id']==id,'type'].values[0] in [
                'Starting XI','Half Start','Duel'
                ]:continue

            #   +1 for goal
            
            if self.events.loc[self.events['id']==id,'shot_outcome'].values[0]=='Goal':
                #   identify scroing team, 0 for home and 1 for away team.
                scoring_team = self.events.loc[self.events['id']==id,'team'].values[0]

                # print(scoring_team)
                current_score[int(scoring_team==self.away_team)]+=1
                    
            self.gamestates.append(GameState(id,self.events,self.frames,current_score=current_score))
            # print(self.gamestates[-1].metadata['period'],self.gamestates[-1].metadata['timestamp'])

def Match_Test():
    os.chdir(os.path.dirname(__file__))
    M_df = pd.read_pickle(r'Matches_EURO.pkl')
    print('Macth columns\n',M_df.columns)
    M = Match(3788765,M_df)
    # M.get_gamestates()    
    # print(M.gamestates)

class GameState:
    def get_inverse_distance(self,x1,y1,x2,y2):    
        x  = ((x1-x2)**2+(y1-y2)**2)**0.5
        try:
            x=1/x
        except:
            x=x
        return x

    def __init__(self,event_id,Events,Frame,current_score=(0,0),device='cpu'):
        self.metadata = Events.loc[
            Events['id']==event_id,
            ['id','period','timestamp']
            ].to_dict('records')[0]
        try:
            self.metadata['visible_area'] = Frame.loc[Frame['id']==event_id,'visible_area'].to_list()[0]
        except:
            # print(Frame.loc[Frame['id']==event_id,'visible_area'])
            # print(f'Event ID:{event_id:50s}\tNo visible area')
            return None
        self.metadata['actor_team']= int(Events.loc[Events['id']==event_id,'team']==Events.loc[Events['id']==event_id,'possession_team'])
        #1 for attacking
        #0 for defensing

        self.metadata['current_score'] = current_score

        #   self.metadata is a dictionary storing different metadata/global data:
        #   'id': event id, used as the key matching the frames

        self.actor = torch.tensor(list(Frame.loc[
            (Frame['id']==event_id)&(Frame['actor']==True),
            'location'
            ]),device=device)
        self.teammates = torch.tensor(list(Frame.loc[
            (Frame['id']==event_id)&(Frame['teammate']),
            'location'
            ]),device=device)
        self.enemy = torch.tensor(list(Frame.loc[
            (Frame['id']==event_id)&(Frame['teammate']==False),
            'location'
            ]),device=device)

        self.locations = torch.cat((
                self.actor,
                self.teammates,
                self.enemy
            ))

        self.node_features = torch.cat((
            self.locations,
            torch.cat((
                torch.ones(self.actor.shape[0]+self.teammates.shape[0],1,device=device)*self.metadata['actor_team'],
                torch.ones(self.enemy.shape[0],1,device=device)*(1-self.metadata['actor_team'])
            ))
        ),axis=1)

        _ = torch.arange(self.node_features.shape[0],device=device)

        edge_index = torch.cat([
                torch.repeat_interleave(_,self.node_features.shape[0]),
                _.repeat(self.node_features.shape[0])
            ])
        
        #   The indexing order is first considering all edges going from the first node to other nodes (self-included), which will have n edges created. Then indexing the edges from the second nodes to other nodes, which also have n edges. Noted that for each pair of nodes, we have 2 edges to represent the bi-/un-directed graph. 

        # print('event id:',event_id)
        # print(self.node_features)
        try:
            x1,x2 = torch.meshgrid(self.node_features[:,0],self.node_features[:,0])        
            y1,y2 = torch.meshgrid(self.node_features[:,1],self.node_features[:,1])        
            inverse_distance = self.get_inverse_distance(x1,x2,y1,y2)
            # print(inverse_distance)

            t1,t2 =torch.meshgrid(self.node_features[:,2],self.node_features[:,2])
            same_team = 1-torch.abs(t1-t2)
            # print(same_team)

            self.edge_attr = torch.cat([inverse_distance.reshape(-1,1),same_team.reshape(-1,1)],axis=1)
            self.graph = Data(self.node_features,edge_index=edge_index,device=device)
        except:
            print(f'Error\nevent id : {event_id}\nself.node_feature : {self.node_features}')
def GameState_Test(device='cpu',event_id='04ba5d38-1d5a-4486-9f10-71ef08cc9351'):
    heads=20
    # Events = sb.events(match_id=3788765).sort_values(['possession','timestamp'])
    # Events.to_pickle(r"C:\Users\scs20\OneDrive - HKUST Connect\2022-23_Fall\COMP4222\Project\Main\DataParsing\Events_3788765.pkl")
    # Frame = sb.frames(match_id=3788765)
    # Frame.to_pickle(r"C:\Users\brian\OneDrive - HKUST Connect\2022-23_Fall\COMP4222\Project\Main\DataParsing\Frames_3788765.pkl")
    os.chdir(os.path.dirname(__file__))
    Events = pd.read_pickle("Events_3788765.pkl")
    Frame = pd.read_pickle("Frames_3788765.pkl")

    st = time.time()

    # event_id = 'e42b7a9f-694c-4ff3-ab08-3455ad35f6e3'   #   Attacking E.g.
    # event_id = '2d2508b7-e9b6-4cfc-9c8b-2f6df4bacaaa'   #   Defencing E.g.

    # print(Events.columns)
    # print(
    #     Events[
    #         ['id','team','possession_team','team','player','period','timestamp','location','possession','play_pattern','type','pass_recipient']].sort_values(
    #             ['possession','period','timestamp']).head(heads))
    # # print(Frame.loc[Frame['id']==event_id,['teammate','keeper','location']])
    # print(Frame.loc[Frame['keeper']==True])

    g=GameState(
    event_id=event_id,
    Events=Events,
    Frame=Frame,
    device=device
    )

    # print('-'*60)

    # print(g.metadata)
    # print(g.node_features)
    # print(g.edge_attri)
    # # print(g.locations.shape)
    print(device,time.time()-st)

def read_events(**kwargs):
    os.chdir(os.path.dirname(__file__))
    pd.set_option('display.max_rows',None)
    Events = pd.read_pickle("Events_3788765.pkl")
    # print(Events.columns)
    
    if kwargs=={}:
        print(Events[['team','period','timestamp','type','shot_outcome']])
    
    if 'event_id' in kwargs:
        print(Events.loc[Events['id']==kwargs['event_id'],['team','period','timestamp','type','shot_outcome']])
    
    if 'type' in kwargs:
        print(Events.loc[Events['type']==kwargs['type'],['team','period','timestamp','type','shot_outcome']])

if __name__=='__main__':    #   Testing
    # GameState_Test(event_id='04ba5d38-1d5a-4486-9f10-71ef08cc9351')
    Match_Test()
    # read_events()
    # read_events(event_id='84703116-40f7-4ef0-aca2-f60769395800')
    # st =time.time()
    # GameState_Test(device='cuda:0')
    # print('cuda:0',time.time()-st)
    # st =time.time()
    # GameState_Test(device='cpu')
    # print('cpu',time.time()-st)

    # os.chdir(os.path.dirname(__file__))
    # pd.set_option('display.max_rows',None)
    # EURO = pd.read_pickle("Matches_EURO.pkl")
    # print(EURO.columns)
    # print(EURO[['home_team','away_team','home_score','away_score']])