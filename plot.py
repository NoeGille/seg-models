import matplotlib.pyplot as plt
import numpy as np
import os
# Plot a multicurve graph of dice score evolution depending on the number of frozen layers 
# starting from the first one. Each curve correponds to a different dataset size.

RESULTS_PATH = 'results/vgg_pretrained/'

dice_list = []
headers = []
number_of_frozen_layers = []

with open(RESULTS_PATH + 'results.csv', 'r') as csvfile:
    lines = csvfile.readlines()
    headers = lines[0].split(';')[1:]
    number_of_frozen_layers = [str(line.split(';')[0]) for line in lines[1:]]
    for line in lines[1:]:
        dice_list.append([float(dice.replace(',', '.')) for dice in line.split(';')[1:]])

dice_list = np.array(dice_list)

# Plotting the evolution of dice scores for each dataset size with shaded colored region

plt.figure(figsize=(10, 6))

number_of_frozen_layers = [str(i + 1) for i in range(len(number_of_frozen_layers))]
number_of_frozen_layers[-1] = 'All layers'


for i, header in enumerate(headers):
    plt.plot(number_of_frozen_layers, dice_list[:, i], 'o-', label=header)

plt.xlabel('Number of updated layers (starting from the last one)')
plt.ylabel('Dice score')
plt.title('Evolution of Dice score by increasing number of updated layers')
plt.grid()
plt.legend(title='Dataset size', loc='upper left')
plt.annotate('*Models have been pretrained on ImageNet then fully retrained for 5 epochs on the synthetic dataset then retrained on 20 epochs', xy=(-0.1, -0.12), xytext=(-0.1, -0.12), xycoords='axes fraction', fontsize=6, va='center')
plt.show()



        


                    













'''# Specify the path to the CSV file
csv_file = 'to_plot.csv'

# Specify the path to the result file
result_path = 'results/binary/'

# Initialize empty lists to store data
depths = []
dice_scores = []
dice_std = []

# Read data from the CSV file
with open(csv_file, 'r') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    for row in reader:
        depths.append(int(row['Depth']))
        dice_scores.append([float(row['Dilation1']), float(row['Dilation2']), float(row['Dilation3']),
                            float(row['Dilation4'])])
        dice_std.append([float(row['Var dil1']) ** 0.5, float(row['Var dil2']) ** 0.5, float(row['Var dil3']) ** 0.5,
                         float(row['Var dil4']) ** 0.5])

# Convert lists to numpy arrays
depths = np.array(depths)
dice_scores = np.array(dice_scores)
dice_std = np.array(dice_std)

# Plotting the evolution of dice scores for each dilation with shaded colored region
dilations = ['Dilation1', 'Dilation2', 'Dilation3', 'Dilation4']
colors = ['blue', 'orange', 'green', 'red']

plt.figure(figsize=(10, 6))

for i, dilation in enumerate(dilations):
    plt.plot(depths, dice_scores[:, i], 'o-', label=dilation, color=colors[i])
    plt.fill_between(depths, dice_scores[:, i] - dice_std[:, i], dice_scores[:, i] + dice_std[:, i],
                     alpha=0.3, color=colors[i])

plt.xlabel('Depth')
plt.ylabel('Dice score')
plt.title('Evolution of Dice score by Dilation and Depth')
plt.legend()

# Save the plot as an image file
plt.savefig(result_path + 'Dice_score.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
'''