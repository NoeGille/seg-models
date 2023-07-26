import matplotlib.pyplot as plt

def plot_loss(losses):
    plt.plot(range(1, len(losses) + 1), losses)
    

def convert_file_to_array(filename):
    with open(filename) as f:
        lines = f.readlines()
    return [float(line.strip()) for line in lines]

if __name__ == '__main__':

    FILE_PATH = 'results/loss/'

    filenames = ['unet_vgg_depth5_pre_e55_len1000_b_loss.txt', 'unet_vgg_depth5_pre_e55_len1000_b4_loss.txt', 'unet_vgg_depth5_pre_e55_len1000_b43_loss.txt', 'unet_vgg_depth5_pre_e55_len1000_b432_loss.txt', 'unet_vgg_depth5_pre_e55_len1000_b4321_loss.txt']
    losses = [convert_file_to_array(FILE_PATH + filename) for filename in filenames]
    for loss in losses:
        plot_loss(loss)

    xlim = 55

    plt.grid(True)
    plt.xlabel('Epoch')
    plt.xlim(0, xlim)
    plt.xticks(range(0, xlim , 5))
    plt.ylabel('Loss')
    plt.legend(['b', 'b4', 'b43', 'b432', 'b4321'])
    plt.show()