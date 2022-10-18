from statsbombpy import sb
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

heads=100
# pd.set_option('large_repr','info')

# df_comp = sb.competitions()[['competition_id','competition_name','season_id','season_name']]
# print(df_comp)

# df_EURO = sb.matches(competition_id=55,season_id=43)
# print(df_EURO.columns)

# print('-'*100)

# df_EURO_info = df_EURO#[['match_id','competition_stage','home_team','away_team','home_score','away_score']]
# print(df_EURO_info['match_status_360'])

Match = sb.events(match_id=3788765)
# print(Match.columns)
print(
    # Match[['id','team','player','period','timestamp','location','possession','play_pattern','type','duration','ball_receipt_outcome','pass_recipient','shot_outcome']].sort_values(['possession','period','timestamp']).head(heads)
    Match[['id','team','player','period','timestamp','location','possession','play_pattern','type','pass_recipient']].sort_values(['possession','period','timestamp']).head(heads)
)
Frame = sb.frames(match_id=3788765)
# print(Frame.columns)
print(Frame)

from DataParsing import DataParsing

# for event_id in Match['id']:
g=DataParsing.GameState(
    event_id='e42b7a9f-694c-4ff3-ab08-3455ad35f6e3',
    Match=Match,
    Frame=Frame
    )
    
print(g.period_timestamp)
print(g.actor)
print(g.teammate)
print(g.enemy)


# EURO_final = sb.events(match_id=3795506)
# print(EURO_final.columns)
# print(EURO_final.shape)
# print(EURO_final[['index','player','timestamp','duration','type']].sort_values(by='timestamp',ascending=False))