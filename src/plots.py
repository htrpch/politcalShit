
import matplotlib.pyplot as plt
import numpy as np

def plot_fluxes(Experimento, save = False):

    fig, axs = plt.subplots(3, figsize = (10,5))

    # AXES: A -> K ; K -> O ; A -> O;
    PhiAK = np.transpose(Experimento.fluxes)[0]
    PhiKO = np.transpose(Experimento.fluxes)[1]
    PhiAO = np.transpose(Experimento.fluxes)[2]

    fluxmax = np.max(np.concatenate([PhiAK,PhiKO,PhiAO]))
    fluxmin =np.min(np.concatenate([PhiAK,PhiKO,PhiAO]))

    plt.setp(axs, yticks=[-fluxmin, 0, fluxmax])

    axs[0].set_title(r'$\phi_{AK}$') 
    axs[0].plot(PhiAK)
    axs[0].set_ylim(-4,4)

    axs[1].set_title(r'$\phi_{KO}$') 
    axs[1].plot(PhiKO)
    axs[1].set_ylim(-4,4)

    axs[2].set_title(r'$\phi_{AO}$') 
    axs[2].plot(PhiAO)
    axs[2].set_ylim(-4,4)

    fig.tight_layout()

    if save:
        plt.savefig('Fluxes_λ_'+str(Experimento.l)+'_δ_'+str(Experimento.delta)+'.png')


def plot_vote_sets_evolution(Experimento, save = False):

    plt.figure(figsize=(10,7))

    plt.title('Opinions Sets Evolution λ = %s, δ = %s'%(Experimento.l,Experimento.delta), fontsize = 20)

    plt.plot(Experimento.times, Experimento.A, label = 'number of politicians in Λ')

    plt.plot(Experimento.times, Experimento.O, label = 'number of politicians in Ω')

    plt.plot(Experimento.times,Experimento.K, label = 'number of politicians in K')

    plt.legend()

    if save:
        plt.savefig('Opinions_Sets_Evolution_λ_'+str(Experimento.l)+'_δ_'+str(Experimento.delta)+'.png')



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


def create_animation(Plista, ordered = True):
    
    if ordered: 
        for i in range(5,120,5):
            plt.imshow(np.sort( create_visualization(Plista,i)), cmap = 'magma')
            #plt.imshow( create_visualization(Plista,i))
            plt.savefig('partisan' + str(i)+'.png')

    if ordered:
        for i in range(5,120,5):
            #plt.imshow(np.sort( create_visualization(Plista,i)))
            plt.imshow( create_visualization(Plista,i), cmap = 'magma')
            plt.savefig('partisanloc' + str(i)+'.png')