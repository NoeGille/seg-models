import matplotlib.pyplot as plt

def plot_loss(losses):
    plt.plot(range(1, len(losses) + 1), losses)
    

def convert_file_to_array(filename):
    with open(filename) as f:
        lines = f.readlines()
    return [float(line.strip()) for line in lines]

if __name__ == '__main__':

    FILE_PATH = 'results/loss/'

    filenames = ['unetr_depth12_pre_lr_loss_e100.txt', 'unetr_depth1_pre_lr_loss_e100.txt', 'unetr_depth6_pre_lr_loss_e100.txt', 'unet_unetr_lr_loss_loss.txt']
    losses = [convert_file_to_array(FILE_PATH + filename) for filename in filenames]
    for loss in losses:
        plot_loss(loss)

    xlim = 100

    plt.grid(True)
    plt.xlabel('Epoch')
    plt.xlim(0, xlim)
    plt.xticks(range(0, xlim , 5))
    plt.ylabel('Loss')
    plt.legend(['UNETR depth 12', 'UNETR depth 1', 'UNETR depth 6', 'UNET'])
    plt.show()