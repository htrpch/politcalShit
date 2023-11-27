
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from crop import crop_statements_until_t
from dataclasses import dataclass
from typing import List

from models import SimulateStatement, Model, PoliticianOpinion, PoliticiansOpinionInTime


@dataclass
class PoliticianOpinionHistogram:
    """Class for identifying a single politician opinion"""
    politician_id: int
    opinions: list[int]
    counts: list[int] 

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
        from_id_to_rate = {id: days_from_td(np.mean(from_id_to_df[id].time.diff())) for id in ids}
        return from_id_to_rate
    
    def get_window_size_fror_probability_estimation(self, days_to_reckoning):
        from_id_to_rate = self.get_rates()
        
        return from_id_to_rate

    def count_lags(start_datetime, end_datetime, lagsize):
        current_datetime = start_datetime
        count = 0
        while current_datetime < end_datetime:
            current_datetime += lagsize
            count += 1
        return count
    
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

    def get_opinion_trajectory_histogram(self, l, delta, lag, lags_from_reckoning, method = 'exp'):

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

        from_time_to_politician_opinion_list[::lags_from_reckoning]

        for time_ in tqdm(times[::lags_from_reckoning]): 
            trajectory_set = self.get_politician_trajectories(politicians_opinion_in_time, )
            from_time_to_politician_opinion_list
        
        return 
    
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
    
    def get_changes(self, l, delta, lag, method='exp'):
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

        tempo = self.df.time[1::lag]
        changes = np.zeros(( len(tempo) , 3 ))


        for ii, t in tqdm(enumerate(tempo)):

            time.sleep(.1)
            politician_opinion_list = []

            for elem in crop_statements_until_t(self.df, t): # de politico em politico

                statements, id_politico = elem
                P = Model(statements).runlite(l, delta,'exp')
                politician_opinion = PoliticianOpinion(id_politico, P)
                politician_opinion_list.append(politician_opinion)

            politicians_opinion_t = PoliticiansOpinionInTime(politician_opinion_list, t)
            politicians_opinions_until_t.append(politicians_opinion_t)

            A = [x.opinion for x in politician_opinion_list].count(1)
            O = [x.opinion for x in politician_opinion_list].count(-1)
            K = [x.opinion for x in politician_opinion_list].count(0)  

            K = K + self.deputados.NOME.count() - (A + O + K)     # presuncao de neutralidade dos calados

            total_sets = total_sets + [[A,O,K]]

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

            ii=ii+1

        self.time = [x.time for x in politicians_opinions_until_t]
        self.changes = changes
        self.politicians_opinions_until_t = politicians_opinions_until_t

        return self
    
    def get_changes_df_interval(self, l, delta, lag, method ='exp'):
        """
        Counts changes of opinion in approval sets
        following model dynamic

        Args: 
        
        l - lambda parameter
        delta - delta parameter
        lag - time lag between measurement of system state

        """

        Plista = []
        total_sets = []

        tempo = self.df.time[1::lag]
        changes = np.zeros(( len(tempo) , 3 ))

        ii = 0
        Nantes = 0

        for t in tqdm(tempo):

            time.sleep(.1)

            #gets statements
            
            p_intm = []

            for elem in crop_statements_until_t(self.df, t): # de politico em politico

                statements,id_politico = elem
                P = Model(statements).runlite(l, delta,'exp')
                p_intm.append([P,id_politico])

            Plista.append([p_intm,t])

            A = [x[0] for x in p_intm].count(1)
            O = [x[0] for x in p_intm].count(-1)
            K = [x[0] for x in p_intm].count(0)  

            K = K + self.deputados.NOME.count() - (A + O + K)     # presuncao de neutralidade dos calados
            total_sets = total_sets + [[A,O,K]]


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

            ii=ii+1

        self.time = [Plista[t][1] for t in range(len(Plista))]

        return changes,  Plista, total_sets

    def get_changes_df_interval_start_undecided(self, l, delta, lag, method='exp'):
        """
        Counts changes of opinion in approval sets
        following model dynamic

        Args: 
        
        l - lambda parameter

        delta - delta parameter

        lag - time lag between measurement of system state

        """

        Plista=[]

        tempo = self.df.time[1::lag]
        changes = np.zeros(( len(tempo) , 3 ))

        ii = 0

        Nantes = 0

        for t in tempo:

            print(ii, end='\r')
            time.sleep(.1)
            #gets statements
            p_intm = []

            for elem in crop_statements_until_t(self.df, t): # de politico em politico

                statements,id_politico = elem
                P = Model(statements).runlite(l, delta,'exp')
                p_intm.append([P,id_politico])


            Plista.append([p_intm,t])

            A = [x[0] for x in p_intm].count(1)

            O = [x[0] for x in p_intm].count(-1)

            K = [x[0] for x in p_intm].count(0)  

            K = K + self.deputados.NOME.count() - (A + O + K)   # presuncao de neutralidade dos calados

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

            ii=ii+1

        return changes,  Plista


    def get_changes_df_interval_L(self, l, delta, lag,  method='exp'):
        """
        Counts changes of opinion (changes within each set size) 
        following model dynamic

        Args: 
        
        l - lambda parameter

        delta - delta parameter

        lag - time lag between measurement of system state

        """

        #changes = np.zeros((T,3))

        Plista=[]

        #tempo = df.time

        tempo = self.df.time[1::lag]

        changes = np.zeros(( len(tempo) , 3 ))
        changesL = np.zeros(( len(tempo) , 3 ))

        ii = 0

        Nantes = 0

        for t in tempo:

            print(ii, end='\r')
            time.sleep(1)
            
            p_intm = []

            for elem in crop_statements_until_t(self.df, t): # de politico em politico

                statements,id_politico = elem
                P = Model(statements).runlite(l, delta,'exp')
                p_intm.append([P,id_politico])

            Plista.append([p_intm,t])

            A = [x[0] for x in p_intm].count(1)

            O = [x[0] for x in p_intm].count(-1)

            K = [x[0] for x in p_intm].count(-1)


            if(ii>1):

                nAL = list(np.transpose(np.array(Plista[-1][0])[[i in list(np.transpose(Plista[-2][0])[1]) for i in list(np.transpose(Plista[-1][0])[1])]])[0]).count(1)
                nOL = list(np.transpose(np.array(Plista[-1][0])[[i in list(np.transpose(Plista[-2][0])[1]) for i in list(np.transpose(Plista[-1][0])[1])]])[0]).count(-1)
                nKL = list(np.transpose(np.array(Plista[-1][0])[[i in list(np.transpose(Plista[-2][0])[1]) for i in list(np.transpose(Plista[-1][0])[1])]])[0]).count(0)

                changes[ii][0] = nA1 - nA
                changes[ii][1] = nO1 - nO
                changes[ii][2] = nK1 - nK

                changesL[ii][0] = nAL - nA
                changesL[ii][1] = nOL - nO
                changesL[ii][2] = nKL - nK

                nA = int(A)
                nO = int(O)
                nK = int(K)

            #elif(ii==0):
            elif(ii<=1):

                nA = int(A)
                nO = int(O)
                nK = int(K)

            ii=ii+1


        #return changes , Plista
        return changes, changesL , Plista

    def get_fluxes_df_interval(self, l, delta,lag, method = 'exp'):
        """
        CONVENÇÃO PARA OS FLUXOS
        FLUXES: A -> K ; K -> O ; A -> O; #
        """

        Plista = []
        Plista_flux = []

        tempo = self.df.time[1::lag]
        self.fluxes = np.zeros(( len(tempo) , 3 ))

        for ii, t in enumerate(tqdm(tempo)):

            time.sleep(1)
            p_intm = []
            all_p = []

            for elem in crop_statements_until_t(self.df, t): # de politico em politico

                statements, id_politico = elem
                P = Model(statements).runlite(l, delta,'exp')
                p_intm.append([P, id_politico])
                all_p.append(P)

            Plista.append([p_intm, t])
            Plista_flux.append(all_p)
            
            if(ii>0):

                changing_opinions = pd.Series(Plista_flux[ii-1]) - pd.Series(Plista_flux[ii][:len(Plista_flux[ii-1])])
                non_zero_changing_opinions = np.where(changing_opinions != 0)[0]

                if len(non_zero_changing_opinions) > 0 :

                    po = np.array(Plista_flux[ii-1])[non_zero_changing_opinions]
                    pf = np.array(Plista_flux[ii])[non_zero_changing_opinions]

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

    def organize_politicalparty(self, Plista, t):

        #df = pd.read_csv('DEPUTADOS_FINAL.csv')

        df = self.deputados

        IdtoParty = {i:df[df.Id_politico == i]['Partido'].values[0] for i in df.Id_politico} 

        #print(IdtoParty)

        parties = [IdtoParty[i] for i in np.transpose(Plista[t][0])[1]]

        parties_participation = { i:parties.count(i) for i in np.unique(parties) }

        partytoopinions= {}

        for [i,j] in Plista[t][0]:
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

    def visualize_parties_evolution(self, Plista):  

        self.parties_opinion_evolution = []

        for t in tqdm(range(len(Plista))):

            parties_participation, partytoopinions, totalpartyopinion = self.organize_politicalparty( Plista, t)

            self.parties_opinion_evolution = self.parties_opinion_evolution + [totalpartyopinion]

        return self.parties_opinion_evolution

    def serie_temporal_partido(self, party):

        serie_A =  [ self.parties_opinion_evolution[i][party][1] for i in range(len(self.parties_opinion_evolution)) ]
        serie_K =  [ self.parties_opinion_evolution[i][party][0] for i in range(len(self.parties_opinion_evolution)) ]
        serie_O =  [ self.parties_opinion_evolution[i][party][-1] for i in range(len(self.parties_opinion_evolution))]

        return serie_A, serie_K, serie_O








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
