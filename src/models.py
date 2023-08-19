
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from crop import crop_statements_until_t
from dataclasses import dataclass
from datetime import datetime

#PLista deve ser substituido por lista de objetos PoliticiansOpinionInTime

@dataclass
class PoliticianOpinion:
    """Class for identifying a single politician opinion"""
    politician_id: int
    opinion: int 

@dataclass
class PoliticiansOpinionInTime:
    """Class for keeping track of politician opinion evolution"""
    politicians_opinions: list[PoliticianOpinion]
    datetime: datetime


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

    def lastOr0(obj):

        if len(obj)==0:
            return 0
        else:
            return obj[-1]

    def h_exp(self,l):

        h = np.zeros(self.N)

        h[0] = self.tau[0]

        for i in range(1,self.N):

            h[i] = l * h[i-1] + (1-l) * self.tau[i]

        return h

    # End result Score

    def h_exp_escalar(self, l):

        h =  self.tau[0]

        for i in range(1,self.N):

            h = l * h + (1-l) * self.tau[i]

        return h

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

    def classifierlite(self,score,delta):

        if score<-delta: return -1
        if score>delta: return 1
        if score<delta and score>-delta: return 0
 

    def runlite(self, l , delta, method='exp'): # t é n de enesimo tweet

        if method=='exp':
            function = self.h_exp_escalar
            scores = function(l)

        if method=='mean':
            function = self.h_mean
            scores = function()

        return self.classifierlite(scores,delta)
