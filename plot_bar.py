import csv
import matplotlib.pyplot as plt
import numpy as np

# Specify the path to the CSV file
csv_file = 'new_ouput2.csv'

# Initialize empty lists to store data
file_names = []
precision = []
recall = []
dice_score = []

# Read data from the CSV file
with open(csv_file, 'r') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        print(row)
        file_names.append(row['File'])
        precision.append(float(row['Precision']))
        recall.append(float(row['Recall']))
        dice_score.append(float(row['Dice Score']))

# Get unique frozen and frozen_nobottle file names
pretrained_file_names = np.unique([name for name in file_names if 'pre' in name])
frozen_file_names = np.unique([name for name in file_names if 'frozen' in name and not 'frozen_nobottle' in name and not 'all' in name])
frozen_nobottle_file_names = np.unique([name for name in file_names if 'frozen_nobottle' in name])
all_file_names = np.unique([name for name in file_names if 'all' in name])

# Sort the file names in ascending order
frozen_file_names = np.sort(frozen_file_names)
frozen_nobottle_file_names = np.sort(frozen_nobottle_file_names)

# Function to convert file names to layer names
def get_layer_name(file_name):
    if 'all' in file_name:
        return 'all'
    if 'pre' in file_name:
        return 'Pretrained'
    layer_num = file_name.split('_')[-1].split('.')[0]
    if layer_num == '1':
        return '1st'
    elif layer_num == '2':
        return '2nd'
    elif layer_num == '3':
        return '3rd'
    elif layer_num == '4':
        return '4th'
    if 'bottle' in file_name:
        return 'Bottleneck'
    else:
        return layer_num + 'th'

# Create bar plots for precision, recall, and dice score
num_pretrained = len(pretrained_file_names)
num_frozen = len(frozen_file_names)
num_frozen_nobottle = len(frozen_nobottle_file_names)
num_all = len(all_file_names)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))

print(dice_score)
print(recall)
print(precision)

# Plot precision
ax[0].bar(np.arange(num_pretrained), precision[:num_pretrained], color='gray')
ax[0].bar(np.arange(num_frozen) + num_pretrained, precision[num_pretrained:num_pretrained + num_frozen],
            label='With bottleneck', color='red')
ax[0].bar(np.arange(num_frozen_nobottle) + num_pretrained + num_frozen, precision[num_pretrained + num_frozen:num_pretrained + num_frozen + num_frozen_nobottle],
            label='Without bottleneck', color='green')
ax[0].axhline(y=precision[num_pretrained + num_frozen + num_frozen_nobottle:], color='black', linestyle='--')
ax[0].set_xticks(np.arange(num_frozen + num_frozen_nobottle + num_pretrained))
ax[0].set_xticklabels([get_layer_name(name) for name in np.concatenate((pretrained_file_names, frozen_file_names, frozen_nobottle_file_names), axis=0)], rotation=45)
ax[0].set_xlabel('Layers')
ax[0].set_ylabel('Precision')
ax[0].set_title('Precision')
ax[0].set_yticks(np.arange(0, 1.1, 0.1))

ax[0].legend()

# Plot recall
ax[1].bar(np.arange(num_pretrained), recall[:num_pretrained], color='gray')
ax[1].bar(np.arange(num_frozen) + num_pretrained, recall[num_pretrained:num_pretrained + num_frozen],
            label='With bottleneck', color='red')
ax[1].bar(np.arange(num_frozen_nobottle) + num_pretrained + num_frozen, recall[num_pretrained + num_frozen:num_pretrained + num_frozen + num_frozen_nobottle],
            label='Without bottleneck', color='green')
ax[1].axhline(y=recall[num_pretrained + num_frozen + num_frozen_nobottle:], color='black', linestyle='--')
ax[1].set_xticks(np.arange(num_frozen + num_frozen_nobottle + num_pretrained))
ax[1].set_xticklabels([get_layer_name(name) for name in np.concatenate((pretrained_file_names, frozen_file_names, frozen_nobottle_file_names), axis=0)], rotation=45)
ax[1].set_xlabel('Layers')
ax[1].set_ylabel('Recall')
ax[1].set_title('Recall')
ax[1].set_yticks(np.arange(0, 1.1, 0.1))

ax[1].legend()

# Plot dice score
ax[2].bar(np.arange(num_pretrained), dice_score[:num_pretrained], color='gray')
ax[2].bar(np.arange(num_frozen) + num_pretrained, dice_score[num_pretrained:num_pretrained + num_frozen],
            label='With bottleneck', color='red')
ax[2].bar(np.arange(num_frozen_nobottle) + num_pretrained + num_frozen, dice_score[num_pretrained + num_frozen:num_pretrained + num_frozen + num_frozen_nobottle],
            label='Without bottleneck', color='green')
ax[2].axhline(y=dice_score[num_pretrained + num_frozen + num_frozen_nobottle:], color='black', linestyle='--')

ax[2].set_xticks(np.arange(num_frozen + num_frozen_nobottle + num_pretrained))
ax[2].set_xticklabels([get_layer_name(name) for name in np.concatenate((pretrained_file_names, frozen_file_names, frozen_nobottle_file_names), axis=0)], rotation=45)
ax[2].set_xlabel('Layers')
ax[2].set_ylabel('Dice Score')
ax[2].set_title('Dice Score')
ax[2].set_yticks(np.arange(0, 1.1, 0.1))

ax[2].legend()

# Adjust spacing between subplots
plt.tight_layout()

# Save the plot as an image file
plt.savefig('bar_plot.png')

# Display the plot
plt.show()
