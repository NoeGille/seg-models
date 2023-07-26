'''Plot scatter plot of dice score vs CO2 emissions | training time | number of parameters'''

import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(dice_list, emissions_list, model_names_list, ax):
    dice_list = np.array(dice_list)
    emissions_list = np.array(emissions_list)
    model_names_list = np.array(model_names_list)

    idx = dice_list != None
    dice_list = dice_list[idx]
    emissions_list = emissions_list[idx]
    model_names_list = model_names_list[idx]


    ax.scatter(dice_list, emissions_list, marker='X')
    for i, txt in enumerate(model_names_list):
        ax.annotate(txt, (dice_list[i], emissions_list[i]))
    

def plot_graph(dice_list, model_names_list, ax, x=None):
    dice_list = np.array(dice_list)
    model_names_list = np.array(model_names_list)

    idx = dice_list != None
    dice_list = dice_list[idx]
    model_names_list = model_names_list[idx]

    if x is None:
        x = np.arange(len(dice_list))
    ax.plot(x, dice_list, 'o-')
    for i, txt in enumerate(model_names_list):
        ax.annotate(txt, (x[i], dice_list[i]))

    
    


if __name__ == '__main__':
    '''
    unetr_dice_list = [ 0.857, 0.870, 0.866, 0.872]
    unetr_names_list = ['UNETR 3', 'UNETR 6', 'UNETR 9', 'UNETR 12']
    unetr_nb_parameters = [27817482, 55589002, 86621322, 111817610]

    unet_dice_list = [0.836, 0.871, 0.884, 0.884]
    unet_names_list = ['U-Net 2', 'U-Net 3', 'U-Net 4', 'U-Net 5']
    unet_nb_parameters = [17825354, 22502602, 23338506, 23749546]
  
    fig, ax = plt.subplots(figsize=(7, 6))
   
    #plot_scatter(np.array(unetr_nb_parameters), unetr_dice_list, unetr_names_list, ax)
    #plot_scatter(np.array(unet_nb_parameters), unet_dice_list, unet_names_list, ax)
    plot_graph(unetr_dice_list, unetr_names_list, ax)
    plot_graph(unet_dice_list, unet_names_list, ax)
    
    plt.ylabel('Dice score')
    plt.xlabel('Depth')
    plt.title('Dice score in relation to depth on Camus dataset')
    plt.xticks(range(0, 4))
    plt.annotate("Depth doesn't mean the same thing for UNETR & U-Net but it shows how much the depth impacts the performances of each architecture", xy=(-0.1, -0.12), xytext=(-0.1, -0.12), xycoords='axes fraction', fontsize=6, va='center')
    plt.grid(True)
    plt.show()'''

    pre_dice_list = [0.923, 0.853, 0.871, 0.887, 0.901, 0.894, 0.901]
    emissions_list = [2.673, 6.783, 6.959, 7.356, 11.558, 14.821, 16.984]
    training_time_list = [18.88, 40.57, 42.7, 44.67, 64.95, 78.87, 83.05]
    model_names_list = ['U-Net', 'UNETR 1', 'UNETR 2', 'UNETR 3', 'UNETR 6', 'UNETR 9', 'UNETR 12']
    
    
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_scatter(np.array(training_time_list) / 5, pre_dice_list, model_names_list, ax)
    plt.ylabel('Dice score')
    plt.xlabel('Training time (min)')
    plt.title('Dice score vs training time')
    plt.grid(True)

    fig,ax = plt.subplots(figsize=(7, 6))
    plot_scatter(np.array(emissions_list) / 5, pre_dice_list, model_names_list, ax)
    plt.ylabel('Dice score')
    plt.xlabel('CO2 emissions (kg)')
    plt.title('Dice score vs CO2 emissions')
    plt.grid(True)

    plt.show()





import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(dice_list, emissions_list, model_names_list, ax):
    dice_list = np.array(dice_list)
    emissions_list = np.array(emissions_list)
    model_names_list = np.array(model_names_list)

    idx = dice_list != None
    dice_list = dice_list[idx]
    emissions_list = emissions_list[idx]
    model_names_list = model_names_list[idx]


    ax.scatter(dice_list, emissions_list)
    for i, txt in enumerate(model_names_list):
        ax.annotate(txt, (dice_list[i], emissions_list[i]))
    
    

if __name__ == '__main__':
    pre_dice_list = [0.923, 0.852, 0.871, 0.887, 0.901, 0.894, 0.901]
    emissions_list = [2.673, 6.783, 6.959, 7.356, 11.558, 14.821, 16.984]
    model_names_list = ['U-Net', 'UNETR 1', 'UNETR 2', 'UNETR 3', 'UNETR 6', 'UNETR 9', 'UNETR 12']
    fig, ax = plt.subplots(figsize=(10, 10))

    plot_scatter(pre_dice_list, np.array(emissions_list) / 5, model_names_list, ax)
    
    dice_list = [0.903, 0.805, 0.836, 0.823, 0.784, 0.847, 0.769]
    plot_scatter(dice_list, np.array(emissions_list) / 5, model_names_list, ax)

    e100_pre_dice_list = [None, 0.879, 0.886, None, None, None, None]
    plot_scatter(e100_pre_dice_list, emissions_list, model_names_list, ax)

    plt.xlabel('Dice score')
    plt.ylabel('CO2 emissions (g)')
    plt.title('Dice score vs CO2 emissions')
    plt.legend(['Pretrained (20 epochs)', 'Random init(20 epochs)', 'Pretrained (100 epochs)'])
    plt.grid(True)
    plt.show()