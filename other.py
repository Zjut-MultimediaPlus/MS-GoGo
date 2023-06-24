import rarfile
import gzip
import numpy as np
import lzma
import gzip
import shutil
import glob

def compress_to_xz(input_path,output_path):
    with open(input_path,'rb') as input:
        with lzma.open(output_path,'wb') as output:
            shutil.copyfileobj(input,output)

def compress_folder_to_xz(input_dir,output_dir):
    file_list = glob.glob(input_dir+'/*')
    num_total = len(file_list)
    iter=0
    for input_path in file_list:
        print(iter,'/',num_total)
        compress_to_xz(input_path,output_dir + input_path.replace(input_dir,'')+'.xz')
        iter +=1

def compress_to_gz(input_path,output_path):
    with open(input_path,'rb') as input:
        with gzip.open(output_path,'wb') as output:
            shutil.copyfileobj(input,output)

def compress_folder_to_gz(input_dir,output_dir):
    file_list = glob.glob(input_dir+'/*')
    num_total = len(file_list)
    iter=0
    for input_path in file_list:
        print(iter,'/',num_total)
        compress_to_xz(input_path,output_dir + input_path.replace(input_dir,'')+'.gz')
        iter +=1



if __name__=='__main__':
    #compress_folder_to_xz('/data/zdxy/DataSets/MLC_16c/multi_channle_data_16','/data/zdxy/DataSets/MLC_16c/multi_channle_data_16_xz')
    compress_folder_to_gz('/data/zdxy/DataSets/MLC_16c/multi_channle_data_16','/data/zdxy/DataSets/MLC_16c/multi_channle_data_16_gzip')
    '''
    npy_data = f.read()
    arr=np.frombuffer(npy_data)
    print(arr)
    '''