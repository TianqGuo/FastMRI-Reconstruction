import matplotlib.pyplot as plt
import numpy as np

import Utils
import config_file
from Data import Fastmri_Data_Processing as dp
from torch.utils.data import DataLoader
import time
from matplotlib import pyplot as plot
'''
Test the implemenation of data processing of data pre_processing 
'''
def show_images(target, input,  start_index):
    fig, axes = plt.subplots(nrows =3 , ncols= 2, figsize =(20,20))
    counter = start_index
    for i in range(3):
        axes[i,0].set_title('target'+ str(counter))
        axes[i,0].imshow(target[counter, 0, :,:], cmap= 'gray')

        axes[i, 1].set_title('input' + str(counter))
        axes[i, 1].imshow(np.abs(input[counter, 0, :, :]), cmap='gray')

        axes[i, 0].get_xaxis().set_visible(False)
        axes[i, 0].get_yaxis().set_visible(False)
        axes[i, 1].get_xaxis().set_visible(False)
        axes[i, 1].get_yaxis().set_visible(False)

        counter += 1
    plot.show()

def test_data_processing():
    '''
    The purpose of this function is given an input data dir and input annotation
    to produce a dataloader which can be pass to the model later
    '''
    start_time = time.time()
    ###
    input_data_dir = config_file.INPUT_DATA_DIR
    input_annotation_dir = config_file.INPUT_ANOTATION_DIR
    slice_idxs = config_file.SLICES
    crop_size = config_file.CROP_SIZE
    batch_size = config_file.BATCH_SIZE
    max_num_data = config_file.MAX_FILE_LIMIT

    h5_lists = dp.get_h5_file_list(input_data_dir, max_num_data)
    if input_annotation_dir != '':
        annotation_lists = dp.get_annotation_file_list(input_annotation_dir)

    if len(h5_lists) == 0:
        print("There is no h5 file input.")
        return

    fastmri_dataset = dp.FastMriDataset(h5_lists, slice_idxs, crop_size)

    train_loader = DataLoader(fastmri_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


    ifft_img, target = next(iter(train_loader))
    print(f"ifft image batch shape: {ifft_img.size()}")
    print(f"target image batch shape: {target.size()}")


    elapsed_time = time.time()- start_time
    print("Elapsed time:", elapsed_time, "secs")
    print("Files count:", config_file.MAX_FILE_LIMIT)

    show_images(target, ifft_img, 0)

def main():
    print("This is a test for data preprocessing")
    test_data_processing()

if __name__ == '__main__':
    main()
