
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from crop import crop_statements_until_t
from dataclasses import dataclass
from typing import List
from datetime import datetime, timedelta

from models import SimulateStatement, Model, PoliticianOpinion, PoliticiansOpinionInTime


# @dataclass
# class PoliticianOpinionHistogram:
#     """Class for identifying a single politician opinion"""
#     politician_id: int
#     opinions: list[int]
#     counts: list[int] 

def days_from_td(delta):
    total_seconds = delta.total_seconds()
    days = total_seconds / (24 * 3600)  
    return days


class ModelStats: 

    def __init__(self, path, deputados_path, simulate = False):

        if not simulate:
            self.df = pd.read_csv(path)
            self.df = self.df.sort_values(by=['time'])
            self.df.time = pd.to_datetime(self. df.time)

            self.N = len(set(self.df.Id_politico))
            self.deputados = pd.read_csv(deputados_path)
            
    def head(self):
        return self.df.head()
    
    def get_politicians(self):
        ids = list(self.df['Id_politico'].unique())
        return ids
    
    def get_rates(self):
        ids = self.get_politicians()
        from_id_to_df = {id : self.df[self.df['Id_politico'] == id] for id in ids}
        from_id_to_rate = {id: 1 / (days_from_td(np.mean(from_id_to_df[id].time.diff()))) for id in ids}
        from_id_to_rate = {id: 0 if np.isnan(from_id_to_rate[id]) else from_id_to_rate[id] for id in ids}
        return from_id_to_rate
    
    def get_window_size_fror_probability_estimation(self, days_to_reckoning):
        ids = self.get_politicians()
        from_id_to_rate = self.get_rates()
        from_id_to_expected_number_of_posts_until_reckoning =  {id: round(from_id_to_rate[id]*days_to_reckoning)  for id in ids}
    
        return from_id_to_expected_number_of_posts_until_reckoning

    def count_lags(start_datetime, end_datetime, lagsize):
        current_datetime = start_datetime
        count = 0
        while current_datetime < end_datetime:
            current_datetime += lagsize
            count += 1
        return count
    
    

    def get_opinions(self, l, delta, lag,  day_of_reckoning, score = 'exp', delta_method = 'dynamic'):

        # times = self.df.time[::lag] -- old formula

        total_delta =  (self.df.time.iloc[-1] - self.df.time.iloc[0]).total_seconds() 
        total_delta_to_reckoning = (day_of_reckoning - self.df.time.iloc[0]).total_seconds() 
        nlags =  round(total_delta/timedelta(days=lag).total_seconds())
        lags_to_reckoning = round(total_delta_to_reckoning/timedelta(days=lag).total_seconds()) # unit is lags

        self.nlags = nlags
        self.lags_to_reckoning = lags_to_reckoning

        times = pd.Series([self.df.time[0] + timedelta(days=lag)*i for i in range(nlags)] )
        self.times = times

        from_time_to_politician_opinion_list = {}
        id_politicos = [id_politico for statements, id_politico in crop_statements_until_t(self.df, times.iloc[-1])] 
        from_politician_to_opinion_history = {id_politico : [] for id_politico in id_politicos}

        for n, time_ in tqdm(enumerate(times)):

            politician_opinion_list = []

            for elem in crop_statements_until_t(self.df, time_): # de politico em politico

                statements, id_politico = elem

                if delta_method ==  'dynamic':
                    P = Model(statements).runlite_dynamic(l, delta, lags_to_reckoning - n, lags_to_reckoning)
                if delta_method ==  'static':
                    P = Model(statements).runlite(l, delta)

                politician_opinion = PoliticianOpinion(id_politico, P)
                from_politician_to_opinion_history[id_politico].append(P)
                politician_opinion_list.append(politician_opinion)
                statements, id_politico = elem

            politicians_opinion_in_time = PoliticiansOpinionInTime(politician_opinion_list, time_)
            from_time_to_politician_opinion_list[time_] = politicians_opinion_in_time

        self.from_time_to_politician_opinion_list = from_time_to_politician_opinion_list
        self.from_politician_to_opinion_history = from_politician_to_opinion_history

        return self
    
    
    def get_changes(self, silent_neutrality=None):
        """
        Counts changes of opinion in approval sets
        following model dynamic

        Args: 
        
        l - lambda parameter
        delta - delta parameter
        lag - time lag between measurement of system state

        """

        politicians_opinions_until_t = []
        total_sets = []

        changes = np.zeros(( len(self.times) , 3 ))

        A_t = []
        O_t = []
        K_t = []

        for ii, t in tqdm(enumerate(self.times)):

            politician_opinion_list = self.from_time_to_politician_opinion_list[t]
            opinions = [x.opinion for x in politician_opinion_list.politician_opinions]

            A = opinions.count(1)
            O = opinions.count(-1)
            K = opinions.count(0)  

            if silent_neutrality:
                K = K + self.deputados.NOME.count() - (A + O + K)     # silent neutrality assumption
                # K = K + self.df.NOME.count() - (A + O + K)     # silent neutrality assumption

            total_sets = total_sets + [[A,O,K]]

            A_t.append(A)
            O_t.append(O)
            K_t.append(K)

            if(ii>0):

                nA1 = int(A)
                nO1 = int(O)
                nK1 = int(K)

                changes[ii][0] = nA1 - nA
                changes[ii][1] = nO1 - nO
                changes[ii][2] = nK1 - nK

                nA = nA1
                nO = nO1
                nK = nK1

            elif(ii==0):

                nA = int(A)
                nO = int(O)
                nK = int(K)

            self.changes = changes
            self.A = A_t
            self.O = O_t
            self.K = K_t

        return self
    

    def get_fluxes(self):
        """
        CONVENÇÃO PARA OS FLUXOS
        FLUXES: A -> K ; K -> O ; A -> O; #
        """

        self.fluxes = np.zeros(( len(self.times) , 3 ))

        for ii, t in enumerate(tqdm(self.times)):

            time.sleep(1)

            if(ii>0):

                politician_opinion_list_0 = self.from_time_to_politician_opinion_list[self.times[ii-1]]
                politician_opinion_list_0 = [x.opinion for x in politician_opinion_list_0.politician_opinions]

                politician_opinion_list = self.from_time_to_politician_opinion_list[t]
                politician_opinion_list = [x.opinion for x in politician_opinion_list.politician_opinions]

                changing_opinions = pd.Series(politician_opinion_list_0) - pd.Series(politician_opinion_list[:len(politician_opinion_list_0)])
                non_zero_changing_opinions = np.where(changing_opinions != 0)[0]

                if len(non_zero_changing_opinions) > 0 :

                    po = np.array(politician_opinion_list_0)[non_zero_changing_opinions]
                    pf = np.array(politician_opinion_list)[non_zero_changing_opinions]

                    phi = np.transpose([po,pf])

                    for i in phi:

                        if (i.tolist() == [ 0. , 1.]):
                            self.fluxes[ii][0] += -1
                        if (i.tolist() == [ 1. , 0.]):
                            self.fluxes[ii][0] += 1
                        if (i.tolist() == [ 0., -1.]):
                            self.fluxes[ii][1] += 1
                        if (i.tolist() == [ -1. , 0.]):
                            self.fluxes[ii][1] += -1
                        if (i.tolist() == [ 1. ,-1.]):
                            self.fluxes[ii][2] += 1
                        if (i.tolist() == [ -1., 1.]):
                            self.fluxes[ii][2] += -1

        return self

    def get_fluxes_stats(self):
        # CONVENÇÃO PARA OS FLUXOS
        # FLUXES: A -> K ; K -> O ; A -> O; #
        #
        means = [np.mean(np.transpose(self.fluxes)[0]),np.mean(np.transpose(self.fluxes)[1]),np.mean(np.transpose(self.fluxes)[2])]
        stds = [np.std(np.transpose(self.fluxes)[0]),np.std(np.transpose(self.fluxes)[1]),np.std(np.transpose(self.fluxes)[2])]
        #print(means)
        return means, stds

    def get_P_stats(self, Plista):
        #pegamos a media do tamanho de cada conjunto

        for P in Plista:
            A = np.where(P==1)
            O = np.where(P==-1)
            K = np.where(P==0)


        means = [np.mean(np.transpose(self.fluxes)[0]),np.mean(np.transpose(self.fluxes)[1]),np.mean(np.transpose(self.fluxes)[2])]
        stds = [np.std(np.transpose(self.fluxes)[0]),np.std(np.transpose(self.fluxes)[1]),np.std(np.transpose(self.fluxes)[2])]
        #print(means)
        return means, stds

    
    def study_delta(self,delta,l,lag):

        m = []
        s = []

        for d in delta:

            fluxes, Plista = self.get_fluxes_df_interval(self.df, d, l, lag)
            #print('debug')
            meanss , stdss = self.get_fluxes_stats(fluxes)
            #print('debug 1')
            m.append(meanss)
            s.append(stdss)
        
        return m, s

    def study_l(self, df, delta, lambd, lag):

        m = []
        s = []

        for l in lambd:

            fluxes, Plista = self.get_fluxes_df_interval(df, delta, l, lag)
            means , stds = self.get_fluxes_stats(fluxes)
            m.append(means)
            s.append(stds)
        
        return m, s

    def map_l_delta(self, df,delta,lambd,lag):

        means3d=[]
        stds3d=[]

        for d in delta:
            print(d)
            m , s = self.study_l(df,d,lambd,lag)
            means3d.append(m)
            stds3d.append(s)
        return means3d, stds3d

    def create_visualization(Plista,t):

        mapa = np.zeros((18,18))
        
        k=0
        j=0
        for i in Plista[t]:
            
            if(k%18 ==0 and k!=0):
                k=0
                j+=1
            if i==0:
                mapa[k][j]=0.5
            else:
                mapa[k][j]=i
            k+=1
        
        return mapa

    def organize_politicalparty(self, t):

        #df = pd.read_csv('DEPUTADOS_FINAL.csv')

        df = self.deputados

        IdtoParty = {i:df[df.Id_politico == i]['Partido'].values[0] for i in df.Id_politico} 

        politician_opinions = self.from_time_to_politician_opinion_list[t]
        politician_opinion_list = [x.opinion for x in politician_opinions.politician_opinions]
        politician_id_list = [x.politician_id for x in politician_opinions.politician_opinions]

        parties = [IdtoParty[i] for i in politician_id_list]

        parties_participation = { i: parties.count(i) for i in np.unique(parties) }

        partytoopinions= {}

        for [i,j] in zip(politician_opinion_list, politician_id_list):
            #print(i,j)
            party = IdtoParty[j]
            if party in partytoopinions.keys():
                partytoopinions[party] = partytoopinions[party] + [i]
            else:
                partytoopinions[party] = [i]

        totalpartyopinion = {}

        for party in np.unique(self.deputados.Partido):
            totalpartyopinion[party] = {1: 0, 0: + self.deputados.Partido.value_counts()[party], -1: 0}

        for party in partytoopinions.keys():
            p_A = partytoopinions[party].count(1)
            p_K = partytoopinions[party].count(0) + self.deputados.Partido.value_counts()[party] - partytoopinions[party].count(1) - partytoopinions[party].count(-1)
            p_O = partytoopinions[party].count(-1)

            totalpartyopinion[party] = {1: p_A, 0: p_K, -1: p_O}

        return parties_participation, partytoopinions, totalpartyopinion

    def visualize_parties_evolution(self):  

        self.parties_opinion_evolution = []

        for t in tqdm(self.times):

            parties_participation, partytoopinions, totalpartyopinion = self.organize_politicalparty( t)

            self.parties_opinion_evolution = self.parties_opinion_evolution + [totalpartyopinion]

        return self.parties_opinion_evolution

    def serie_temporal_partido(self, party):

        serie_A =  [ self.parties_opinion_evolution[i][party][1] for i in range(len(self.parties_opinion_evolution)) ]
        serie_K =  [ self.parties_opinion_evolution[i][party][0] for i in range(len(self.parties_opinion_evolution)) ]
        serie_O =  [ self.parties_opinion_evolution[i][party][-1] for i in range(len(self.parties_opinion_evolution))]

        return serie_A, serie_K, serie_O
    

    def get_score_histogram(self, l, delta, lag, method = 'exp'):

        times = self.df.time[::lag]
        politician_opinion_list = []
        from_time_to_politician_opinion_list = {}
        from_politician_to_opinion_list = {}

        id_politicos = [id_politico for statements, id_politico in crop_statements_until_t(self.df, times.iloc[-1])] 

        for time_ in tqdm(times):
            for elem in crop_statements_until_t(self.df, time_): # de politico em politico

                statements, id_politico = elem
                P = Model(statements).runlite(l, delta,method)
                politician_opinion = PoliticianOpinion(id_politico, P)
                politician_opinion_list.append(politician_opinion)
                statements, id_politico = elem

            politicians_opinion_in_time = PoliticiansOpinionInTime(politician_opinion_list, time_)
            from_time_to_politician_opinion_list[time_] = politicians_opinion_in_time
        
        from_id_politico_to_opinion_list = {}
        politicians_opinion_histograms = []

        for id_politico in tqdm(id_politicos):
            opinion_in_that_time_list = []
            for time_ in times:
                opinion_in_that_time = [i for i in from_time_to_politician_opinion_list[time_].politicians_opinions if i.politician_id == id_politico][0]
                opinion_in_that_time_list += [opinion_in_that_time.opinion] 
            from_id_politico_to_opinion_list[id_politico] = opinion_in_that_time_list 
            values, counts = np.histogram(opinion_in_that_time_list)
            politicians_opinion_histograms.append(PoliticianOpinionHistogram(id_politico,values, counts))

        return politicians_opinion_histograms
    
    def get_politician_trajectories(opinions_in_time: List[PoliticiansOpinionInTime], politician_id: int):
        """
        Get all different trajectories of opinions for a single politician.

        Parameters:
        - opinions_in_time: List of PoliticiansOpinionInTime instances.
        - politician_id: integer associated with politician.

        Returns:
        - A list of trajectories for the specified politician.
        """
        politician_trajectories = []

        # Iterate through the list of opinions_in_time
        for opinion_in_time in opinions_in_time:
            datetime_point = opinion_in_time.datetime

            # Find the politician's opinion at this datetime_point
            politician_opinion = next((opinion.opinion_score for opinion in opinion_in_time.politician_opinions
                                    if opinion.politician_id == politician_id), None)

            if politician_opinion is not None:
                # Append the datetime_point and opinion to the trajectories
                politician_trajectories.append((datetime_point, politician_opinion))

        return politician_trajectories








#Modelo([[1,0,-1,1],[1,0,-1,1]]).teste()


#print("\n")
#Model([1,1,1,1]).teste()

#print("\n")
#print(Model([1,0,0,1]).h_exp_escalar(0.9))

# print("\n")
# print(Model([1,0,0,1]).h_exp(0.9))

# uou  = ModelStats('dataId.csv')

# # print(uou.headd())

# changes, Plista = uou.get_changes_df_interval(0.9,0.2,100)

# # print(a)

# print(Plista[1])

# changes, changesL, Plista = uou.get_changes_df_interval_L(0.9,0.2,100)

# print(Plista[1])
# print(b
