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