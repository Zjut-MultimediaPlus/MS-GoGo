import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE

def vis_2d_tensor(arr:np.ndarray,dst_shape:tuple,save_folder:str,img_name:str,show=False):
    """

    Args:
        arr: the array to be visulized
        dst_shape: the wanted size
        save_folder: where the fig will save
        img_name: the name of img
        show: if true the fig will be shown

    Returns:

    """
    arr = cv2.resize(arr, dst_shape, interpolation=cv2.INTER_AREA)
    plt.matshow(arr)
    if show:
        plt.show()
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(save_folder+'/'+img_name+'.png',dpi=300)
    plt.close()

def label_tsne_vis(arr:np.ndarray, dimension:int, save_folder:str,img_name:str,show=False):

    tsne = TSNE(n_components=dimension,learning_rate=1)
    tsne.fit_transform(arr)
    cord = tsne.embedding_

    size = 20
    if cord.shape[0]==17:
        # draw dots

        plt.scatter(cord[0:4, 0], cord[0:4, 1], c='#FF0000', s=size)
        plt.scatter(cord[4:8, 0], cord[4:8, 1], c='#00FF00', s=size)
        plt.scatter(cord[8:,  0], cord[8:,  1], c='#0000FF', s=size)

        # add txt annotation
        label_names = ['TC','EC','FS','WJ','Snow','Ocean','Desert','Vegetation','Ci','Cs','DC','Ac','As','Ns','Cu','Sc','St']
        ann_disx = np.std(cord[:,0])
        ann_disy = np.std(cord[:,1])
        for i in range(17):
            plt.annotate(label_names[i], xy=cord[i], xytext=(cord[i][0]+0.05*ann_disx, cord[i][1]+0.05*ann_disy),size=5)

        if show:
            plt.show()


    else:
        plt.scatter(cord[:, 0], cord[:, 1], c='#FF0000', s=size)
        ann_disx = np.std(cord[:, 0])
        ann_disy = np.std(cord[:, 1])
        for i in range(arr.shape[0]):
            plt.annotate('%3d'%i, xy=cord[i], xytext=(cord[i][0]+0.05*ann_disx, cord[i][1]+0.05*ann_disy),size=5)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(save_folder+'/'+img_name+'.png', dpi=300)
    plt.close()
