
import matplotlib.pyplot as plt
import numpy as np

def plot_fluxes(Experimento):

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