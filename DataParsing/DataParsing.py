import pandas as pd
from pyparsing import disable_diag
from statsbombpy import sb
import torch
import numpy as np
from torch_geometric.data import Data
import time


class GameState:
    def get_distance(node1,node2):    
        # node1,node2 are the 2 xy coordinates 
        # node1=[x1,y1,<other features>],pos2=[x2,y2,<other features>]
        return ((node1[0]-node2[0])**2+(node1[1]-node2[1])**2)**0.5

    def __init__(self,event_id,Match,Frame):
        self.metadata = Match.loc[
            Match['id']==event_id,
            ['id','period','timestamp']
            ].to_dict('records')[0]
        self.metadata['visible_area'] = Frame.loc[Frame['id']==event_id,'visible_area'].to_list()[0]
        self.metadata['actor_team']= int(Match.loc[Match['id']==event_id,'team']==Match.loc[Match['id']==event_id,'possession_team'])
        #1 for attacking
        #0 for defensing

        self.actor = torch.tensor(list(Frame.loc[
            (Frame['id']==event_id)&(Frame['actor']==True),
            'location'
            ]))
        self.teammates = torch.tensor(list(Frame.loc[
            (Frame['id']==event_id)&(Frame['teammate']),
            'location'
            ]))
        self.enemy = torch.tensor(list(Frame.loc[
            (Frame['id']==event_id)&(Frame['teammate']==False),
            'location'
            ]))

        self.locations = torch.cat((
                self.actor,
                self.teammates,
                self.enemy
            ))

        self.node_features = torch.cat((
            self.locations,
            torch.cat((
                torch.ones(self.actor.shape[0]+self.teammates.shape[0],1)*self.metadata['actor_team'],
                torch.ones(self.enemy.shape[0],1)*(1-self.metadata['actor_team'])
            ))
        ),axis=1)

        _ = torch.arange(self.node_features.shape[0])

        edge_index = torch.tensor(
            [
                _.repeat_interleave(self.node_features.shape[0]),
                _.repeat(self.node_features.shape[0])
            ]
        )  
        #   The indexing order is first considering all edges going from the first node to other nodes (self-included), which will have n edges created. Then indexing the edges from the second nodes to other nodes, which also have n edges. Noted that for each pair of nodes, we have 2 edges to represent the bi-/un-directed graph. 

        distance = torch.zeros(self.node_features.shape[0],self.node_features.shape[0])

        for i in range(self.node_features.shape[0]):
            for j in range(i+1,self.node_features.shape[0]):
                distance[i,j] = self.get_distance(self.node_features[i],self.node_features[j])

        distance += distance.T

        self.edge_attri = torch.tensor(distance.reshape(-1,1))
        self.graph = Data(self.node_features,edge_index=edge_index)


if __name__=='__main__':    #   Testing
    heads=20
    # Match = sb.events(match_id=3788765)
    # Match.to_pickle(r"C:\Users\brian\OneDrive - HKUST Connect\2022-23_Fall\COMP4222\Project\Main\DataParsing\Match_3788765.pkl")
    # Frame = sb.frames(match_id=3788765)
    # Frame.to_pickle(r"C:\Users\brian\OneDrive - HKUST Connect\2022-23_Fall\COMP4222\Project\Main\DataParsing\Frame_3788765.pkl")
    Match = pd.read_pickle(r"C:\Users\brian\OneDrive - HKUST Connect\2022-23_Fall\COMP4222\Project\Main\DataParsing\Match_3788765.pkl")
    Frame = pd.read_pickle(r"C:\Users\brian\OneDrive - HKUST Connect\2022-23_Fall\COMP4222\Project\Main\DataParsing\Frame_3788765.pkl")

    st = time.time()

    event_id = 'e42b7a9f-694c-4ff3-ab08-3455ad35f6e3'   #   Attacking E.g.
    # event_id = '2d2508b7-e9b6-4cfc-9c8b-2f6df4bacaaa'   #   Defencing E.g.

    print(Match.columns)
    print(
        Match[
            ['id','team','possession_team','team','player','period','timestamp','location','possession','play_pattern','type','pass_recipient']].sort_values(
                ['possession','period','timestamp']).head(heads))
    # # print(Frame.loc[Frame['id']==event_id,['teammate','keeper','location']])
    # print(Frame.loc[Frame['keeper']==True])

    g=GameState(
    event_id=event_id,
    Match=Match,
    Frame=Frame
    )

    print('-'*60)

    print(g.metadata)
    print(g.node_features)
    # print(g.locations.shape)
    print(time.time()-st)