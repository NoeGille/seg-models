import csv
import matplotlib.pyplot as plt
import numpy as np

# Specify the path to the CSV file
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
