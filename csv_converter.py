import os
import csv

# Specify the directory where the text files are located
directory = 'results/new_unetr/'

# Specify the output CSV file name
output_file = 'output_unetr.csv'

# Specify the header for the CSV file
header = ['File', 'Kwargs', 'Dataset', 'Epochs', 'Precision', 'Recall', 'Dice Score', 'Precision Variance', 'Recall Variance', 'Dice Score Variance']

# Initialize an empty list to store the extracted data
data = []

# Iterate over the files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            content = file.read()
            print(content)
            # Extract the relevant values from the text file
            kwargs_start = content.index('Kwargs :') + len('Kwargs :')
            kwargs_end = content.index('\n')
            kwargs = content[kwargs_start:kwargs_end].strip()

            dataset_start = content.index('Dataset :') + len('Dataset :')
            dataset_end = content.index('\n', dataset_start)
            dataset = content[dataset_start:dataset_end].strip()

            epochs_start = content.index('Number of epochs :') + len('Number of epochs :')
            epochs_end = content.index('\n', epochs_start)
            epochs = content[epochs_start:epochs_end].strip()

            precision_start = content.index('Precision :') + len('Precision :')
            precision_end = content.index('\n', precision_start)
            precision = content[precision_start:precision_end].strip()

            recall_start = content.index('Recall :') + len('Recall :')
            recall_end = content.index('\n', recall_start)
            recall = content[recall_start:recall_end].strip()

            dice_start = content.index('Dice score :') + len('Dice score :')
            dice_end = content.index('\n', dice_start)
            dice = content[dice_start:dice_end].strip()

            precision_var_start = content.index('Precision variance :') + len('Precision variance :')
            precision_var_end = content.index('\n', precision_var_start)
            precision_var = content[precision_var_start:precision_var_end].strip()

            recall_var_start = content.index('Recall variance :') + len('Recall variance :')
            recall_var_end = content.index('\n', recall_var_start)
            recall_var = content[recall_var_start:recall_var_end].strip()

            dice_var_start = content.index('Dice score variance :') + len('Dice score variance :')
            #dice_var_end = content.index('\n', dice_var_start)
            dice_var = content[dice_var_start].strip()

            # Append the extracted data to the list
            data.append([
                filename,
                kwargs,
                dataset,
                epochs,
                precision,
                recall,
                dice,
                precision_var,
                recall_var,
                dice_var
            ])

# Write the data to the CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(data)

print('Data extraction completed. CSV file created:', output_file)
