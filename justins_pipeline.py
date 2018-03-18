import pandas as pd

def w_avg(group, val_col, weight_col):
   w = group[weight_col].transpose() / group[weight_col].sum()
   v = group[val_col]
   return w.dot(v)


def get_seeds(seeds_path):
   # read in NCAATourneySeeds.csv
   seeds_raw = pd.read_csv(seeds_path)
   
   # prepare seeds index for merging
   seeds = seeds_raw.set_index(['Season', 'TeamID'])
   
   # convert seed data to int
   seeds['Seed'] = seeds['Seed'].apply(lambda x: x[1:3]).astype(int)

   return seeds


def get_ordinals(ordinal_path):
   # read in MasseyOrdinals.csv
   ordinals_raw = pd.read_csv(ordinal_path)
   
   # filter out rankings except for MOR
   ordinals_raw = ordinals_raw[ordinals_raw['SystemName'] == 'MOR']
      
   # instead of using ordinal ranks from different days as distinct features, would like to have one feature per ranking system
   # probably a bad idea to use a moving average because of missing data
   # will use a weighted average, so that rankings at later dates have greater importance, while avoiding unpredictability from missing data
      
   # get weighted average of each system ranking for each team
   ordinals = (pd.DataFrame(ordinals_raw.groupby(['Season', 'TeamID', 'SystemName'])
                                        .apply(w_avg, 'OrdinalRank', 'RankingDayNum'),   # return weighted average of each grouping
                            columns=['WeightedOrdinal'])
              .unstack(level=-1)['WeightedOrdinal'])    # each unique ranking system is a feature

   return ordinals


def get_team_stats(results_path, seeds_path, ordinal_path):
   # read in RegularSeasonDetailedResults.csv
   stats = pd.read_csv(results_path)
   
   # get rid of data we won't be using
   stats.drop(columns=['DayNum', 'WLoc'], inplace=True)
   
   # stats (grouped by team) summed up across all regular season wins
   w_stats = (stats.drop(columns=['LTeamID'])
                   .rename(columns={'WTeamID': 'TeamID'})
                   .rename(columns=lambda x: x[1:] + '_for'     if x[0] == 'W' else x)
                   .rename(columns=lambda x: x[1:] + '_against' if x[0] == 'L' else x)
                   .groupby(['Season', 'TeamID'])
                   .sum())
      
   # stats (grouped by team) summed up across all regular season losses
   l_stats = (stats.drop(columns=['WTeamID'])
                   .rename(columns={'LTeamID': 'TeamID'})
                   .rename(columns=lambda x: x[1:] + '_for'     if x[0] == 'L' else x)
                   .rename(columns=lambda x: x[1:] + '_against' if x[0] == 'W' else x)
                   .groupby(['Season', 'TeamID'])
                   .sum())
              
   # combine (add) stats from wins and losses
   team_stats = w_stats + l_stats
              
   # compute number of wins and losses
   team_stats['Wins']   = stats.groupby(['Season', 'WTeamID']).size().astype(int)
   team_stats['Losses'] = stats.groupby(['Season', 'LTeamID']).size().astype(int)
              
   # reorganize columns
   important_cols = ['Wins', 'Losses', 'Score_for', 'Score_against']
   team_stats = team_stats[important_cols + [col for col in list(team_stats.columns) if col not in important_cols]]
              
   team_stats_p_game = (team_stats.divide(team_stats['Wins'] + team_stats['Losses'], axis='index')
                                  .drop(columns=['Losses']))

   # collect supplemental data
   seeds = get_seeds(seeds_path)
   ordinals = get_ordinals(ordinal_path)

   # merge everything
   team_stats_all = (team_stats_p_game.merge(seeds,    how='left', left_index=True, right_index=True)
                                      .merge(ordinals, how='left', left_index=True, right_index=True))
              
   return team_stats_all


def results_processing(results_in, team_stats, fillna=True, thresh=0, save_name=None):
   # preprocessing
   results = results_in[['Season', 'WTeamID', 'LTeamID']].copy()
   results['y'] = (results['WTeamID'] < results['LTeamID']).astype(int)
   
   results['LowID']  = np.minimum(results['WTeamID'], results['LTeamID'])
   results['HighID'] = np.maximum(results['WTeamID'], results['LTeamID'])
   
   # these columns are no longer needed
   results.drop(columns=['WTeamID', 'LTeamID'], inplace=True)
   
   # concatenating team stats
   results = (results.merge(team_stats.rename(columns=lambda x: 'l_' + x), how='left', left_on=['Season', 'LowID'], right_index=True)
                     .merge(team_stats.rename(columns=lambda x: 'h_' + x), how='left', left_on=['Season', 'HighID'], right_index=True)
                     .drop(columns=['Season', 'LowID', 'HighID']))
      
   results.dropna(thresh=thresh, inplace=True)
              
   if fillna:
      results.fillna(results.mean(), inplace=True)

   if save_name:
      results.to_csv('dataframes/{}.csv'.format(save_name))
   
   return results


def individual_to_diff(results_df, save_name=None):
   # determine which columns belong to low and high teams
   l_features = [feat for feat in list(results_df.columns) if feat[0:2] == 'l_']
   h_features = [feat for feat in list(results_df.columns) if feat[0:2] == 'h_']
   
   # separate columns corresponding to stats from low and high teams
   l_df = results_df[l_features].rename(columns=lambda x: x[2:])
   h_df = results_df[h_features].rename(columns=lambda x: x[2:])
   
   # d_df represents difference between stats of lower and higher id teams
   d_df = l_df - h_df
   d_df.rename(columns=lambda x: 'd_' + x, inplace=True)
   
   # generate new dataframe by merging components
   results_diff = pd.merge(pd.DataFrame(results_df['y']),
                           d_df,
                           how='inner',
                           left_index=True,
                           right_index=True)
      
   if save_name:
      results_diff.to_csv('dataframes/{}.csv'.format(save_name))

   return results_diff


def full_pipeline(post_results, reg_results_path, seeds_path, ordinal_path, fillna=True):
   # get postseason results
   if isinstance(post_results, str):
      post_results = pd.read_csv(post_results)

   # get rid of results before 2003
   post_results = post_results[post_results['Season'] >= 2003]
   
   # generate team stats
   team_stats = get_team_stats(reg_results_path, seeds_path, ordinal_path)

   # use team stats as features for postseason results
   df = results_processing(post_results, team_stats, fillna)
   
   # difference based features
   df = individual_to_diff(df)

   return df

