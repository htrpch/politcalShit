
import numpy as np
import pandas as pd
from tests.crop import crop_statements_until_t

class SimulateStatement:
    def __init__(self, N, maxtweets):
        self.N = N
        self.maxtweets = maxtweets
    
    
    def np_continuous(self):
        """
        cria tweets um vetor com NxMaxTweets
        statements que podem assumir valor continuous
        """
        statements = np.zeros((self.N,self.maxtweets))
        for i in range(0,self.N):
            statements[i] =  np.random.uniform(-1,1,self.maxtweets)

        return statements
    
    def np_binary(self):
        """
        cria tweets um vetor com NxMaxTweets
        statements que podem assumir valor -1 ou 1
        """
        statements = np.zeros((self.N,self.maxtweets))
        for i in range(0,self.N):
            #statements[i] =  np.random.uniform(-1,1,T)
            statements[i] = np.random.randint(0,2,self.maxtweets)
            statements[i][np.where(statements[i]==0)]=-1
        return statements


    def list_continuous(self):
        """
        cria tweets um vetor com NxArbitrario (tamanho = # posts do politico)
        statements que podem assumir valor continuous
        """
        statements = []
        for i in range(0,self.N):
            maxt = np.random.randint(0,self.maxtweets)
            statementsi =  np.random.uniform(-1,1,maxt)
            statements.append(statementsi)
        return statements

    # cria tweets um vetor com NxArbitrario (tamanho = # posts do politico)
    # statements que podem assumir valor -1 ou 1

    def list_binary(self):
        """
        cria tweets um vetor com NxArbitrario (tamanho = # posts do politico)
        statements que podem assumir valor -1 ou 1
        """
        statements = []
        for i in range(0,self.N):
            maxt = np.random.randint(0,self.maxtweets)
            statementsi =  np.random.randint(0,2,maxt)
            statementsi[np.where(statementsi==0)]=-1
            statements.append(statementsi)

        return statements

class Model: 
    
    def __init__(self, tau):
        self.N = len(tau)
        self.tau = tau
    #self.maxtweets = maxtweets

    def lastOr0(obj):
        if len(obj)==0:
            return 0
        else:
            return obj[-1]

    def h_exp(self,l):

        h = np.zeros(len(self.tau))
        h[0] = self.tau[0]

        for i in range(1,len(self.tau)):
          h[i] = l*h[i-1] + (1-l)*self.tau[i]
        return h

    # End result Score

    def h_exp_escalar(self,l):
        return h_exp(l,delta)[-1]

    # Score as mean of posts

    def h_mean(self):
        return [np.mean(self.tau[:i]) for i in range(len(self.tau))]


    def classifier(self,scores,delta):
        h=[]
        for i in range(len(scores)):
            obj = scores[i]
            if obj<-delta:
                h.append(-1)
            if obj>delta:
                h.append(1)
            if obj<delta and obj>-delta:
                h.append(0)

        return h

    def run(self, l , delta, method='exp'): # t é n de enesimo tweet

        if method=='exp':
            function = self.h_exp
            scores = function(l)

        if method=='mean':
            function = self.h_mean
            scores = function()

        return self.classifier(scores,delta)

class ModelStats: 
    def __init__(self, path, simulate = False):

        if not simulate:
            self.df = pd.read_csv(path)
        #else:
         #   self


    
    def get_changes_df_interval(self, l, delta, lag, method='exp'):
    
        Plista=[]

        tempo = df.time[1::lag]
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
                P = Model(statements).run_model(l, delta,'exp')
                p_intm.append([P,id_politico])

            # funcao
            # se tau=[] retorna 0
            # caso contrario traz tau[-1] (ultimo tweet)

            Plista.append([p_intm,t])

            A = [Model.lastOr0(y) for y in [x[0] for x in p_intm]].count(1)
            #np.where([x[0] for x in P]==1)
            O = [Model.lastOr0(y) for y in [x[0] for x in p_intm]].count(-1)
            #np.where([x[0] for x in P]==-1)
            K = [Model.lastOr0(y) for y in [x[0] for x in p_intm]].count(0)
            #np.where([x[0] for x in P]==0)

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

        #return changes , Plista
        return changes,  Plista


    def get_changes_exp_df_interval_L(self, l, delta,lag):
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

        tempo = df.time[1::lag]

        changes = np.zeros(( len(tempo) , 3 ))
        changesL = np.zeros(( len(tempo) , 3 ))

        ii = 0

        Nantes = 0

        for t in tempo:

            print(ii, end='\r')
            time.sleep(1)

            #gets statements
            #P = run_model_exp_def(statements, l, delta)
            #P = Model(statements).run_model(l, delta,'exp')
            
            p_intm = []
            for elem in crop_statements_until_t(df, t): # de politico em politico

                statements,id_politico = elem
                P = Model(statements).run_model(l, delta,'exp')
                p_intm.append([P,id_politico])

            # funcao
            # se tau=[] retorna 0
            # caso contrario traz tau[-1] (ultimo tweet)

            Plista.append([p_intm,t])

            A = [Model.lastOr0(y) for y in [x[0] for x in p_intm]].count(1)
            #np.where([x[0] for x in P]==1)
            O = [Model.lastOr0(y) for y in [x[0] for x in p_intm]].count(-1)
            #np.where([x[0] for x in P]==-1)
            K = [Model.lastOr0(y) for y in [x[0] for x in p_intm]].count(0)
            #np.where([x[0] for x in P]==0)

            #Nnow = len(P)
            #Plista.append(P)

            #A = np.where(P==1)
            #O = np.where(P==-1)
            #K = np.where(P==0)

            if(ii>1):

                #Plista[len(Plista)-1] -> esse é o ultimo (P)

                P0 = Plista[len(Plista)-2]

                #A0 = np.where(P0 == 1)
                #O0 = np.where(P0 == -1)
                #K0 = np.where(P0 == 0)

                AL = A[0][A[0]<=len(P0)-1]
                OL = O[0][O[0]<=len(P0)-1]
                KL = K[0][K[0]<=len(P0)-1]

                nA1 = len(A[0])
                nO1 = len(O[0])
                nK1 = len(K[0])

                nAL = len(AL)
                nOL = len(OL)
                nKL = len(KL)

                changes[ii][0] = nA1 - nA
                changes[ii][1] = nO1 - nO
                changes[ii][2] = nK1 - nK

                changesL[ii][0] = nAL - nA
                changesL[ii][1] = nOL - nO
                changesL[ii][2] = nKL - nK

                nA = nA1
                nO = nO1
                nK = nK1

            #elif(ii==0):
            elif(ii<=1):

                nA = len(A[0])
                nO = len(O[0])
                nK = len(K[0])

            ii=ii+1


        #return changes , Plista
        return changes, changesL , Plista

    def get_fluxes_exp_df_interval(self, l, delta,lag):

        Plista=[]

        #philista=[]

        tempo = df.time[1::lag]

        fluxes = np.zeros(( len(tempo) , 3 ))

        # CONVENÇÃO PARA OS FLUXOS
        #
        # FLUXES: A -> K ; K -> O ; A -> O; #
        #
        #

        ii = 0

        Nantes = 0

        for t in tempo:

            print(ii, end='\r')
            time.sleep(1)
            statements = crop_statements_until_t(df, t)
            P = run_model_exp_def(statements, l, delta)
            Plista.append(P)

            if(ii>0):

                phi=np.transpose([Plista[ii-1][np.where(Plista[ii-1]-Plista[ii][0:len(Plista[ii-1])] != 0)],Plista[ii][np.where(Plista[ii-1]-Plista[ii][0:len(Plista[ii-1])] != 0)]])

                for i in phi:
                    #print(i)
                    if (i.tolist() == [ 0. , 1.]):
                        fluxes[ii][0] += -1
                    if (i.tolist() == [ 1. , 0.]):
                        fluxes[ii][0] += 1
                    if (i.tolist() == [ 0., -1.]):
                        fluxes[ii][1] += 1
                    if (i.tolist() == [ -1. , 0.]):
                        fluxes[ii][1] += -1
                    if (i.tolist() == [ 1. ,-1.]):
                        fluxes[ii][2] += 1
                    if (i.tolist() == [ -1., 1.]):
                        fluxes[ii][2] += -1


            ii = ii+1

        #return changes , Plista
        return fluxes, Plista

    def get_fluxes_stats(fluxes):
        # CONVENÇÃO PARA OS FLUXOS
        # FLUXES: A -> K ; K -> O ; A -> O; #
        #
        means = [np.mean(np.transpose(fluxes)[0]),np.mean(np.transpose(fluxes)[1]),np.mean(np.transpose(fluxes)[2])]
        stds = [np.std(np.transpose(fluxes)[0]),np.std(np.transpose(fluxes)[1]),np.std(np.transpose(fluxes)[2])]
        #print(means)
        return means, stds

    def get_P_stats(Plista):
        #pegamos a media do tamanho de cada conjunto
        means = [np.mean(np.transpose(fluxes)[0]),np.mean(np.transpose(fluxes)[1]),np.mean(np.transpose(fluxes)[2])]
        stds = [np.std(np.transpose(fluxes)[0]),np.std(np.transpose(fluxes)[1]),np.std(np.transpose(fluxes)[2])]
        #print(means)
        return means, stds

    
    def study_delta(df,delta,l,lag):

        m = []
        s = []

        for d in delta:

            fluxes, Plista = get_fluxes_exp_df_interval(df, d, l, lag)
            #print('debug')
            meanss , stdss = get_fluxes_stats(fluxes)
            #print('debug 1')
            m.append(meanss)
            s.append(stdss)
        
        return m, s

    def study_l(df,delta,lambd,lag):

        m = []
        s = []

        for l in lambd:

            fluxes, Plista = get_fluxes_exp_df_interval(df, delta, l, lag)
            #print('debug')
            means , stds = get_fluxes_stats(fluxes)
            #print('debug 1')
            m.append(means)
            s.append(stds)
        
        #print('xegou')
        return m, s

    def map_l_delta(df,delta,lambd,lag):

        means3d=[]
        stds3d=[]

        for d in delta:
            print(d)
            m , s = study_l(df,d,lambd,lag)
            means3d.append(m)
            stds3d.append(s)
        return means3d, stds3d

