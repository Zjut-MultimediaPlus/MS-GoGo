import numpy as np
import torch

#@profile
def augmentation_plus(image):

    # flip up and down
    flag = np.random.randint(0, 1)
    if flag == 1:
        image = np.flip(image,axis=1)

    # flip right and left
    flag = np.random.randint(0, 1)
    if flag == 1:
        image = np.flip(image,axis=2)

    # rotate
    rotation = np.random.randint(0, 3)
    image = np.rot90(image, k=rotation, axes=(1, 2))

    # exposure
    image = random_exposure(image,0.5).copy()

    # contraction
    image = random_contraction(image,0.05,0.05).copy()

    return image

def augmentation_plus_gpu(image):

    # flip up and down

    flag = torch.randint(0,2,(1,))
    if flag.data == 1:
        image = torch.flip(image,dims=[1]).detach()

    # flip right and left
    flag = torch.randint(0,2,(1,))
    if flag == 1:
        image = torch.flip(image,dims=[2]).detach()

    # rotate
    rotation = torch.randint(0,4,(1,))
    image = torch.rot90(image, k=rotation.item(), dims=(1, 2)).detach()


    # exposure
    image = random_exposure_gpu(image,1).detach()#.copy()

    
    # contraction
    image = random_contraction_gpu(image,0.1,0.1).detach()#.copy()
    

    # gaussion noise
    flag = torch.randint(0, 2, (1,))
    if flag.data == 1:
        m = torch.randn(1)/150
        std = torch.randn(1)/150 + 0.01
        image = random_gaussion_noise_gpu(image, m.data, abs(std.data))
    


    # drop a channel
    '''
    flag = torch.randint(0, 2, (1,))
    if flag.data == 1:
        image = random_replace_channel_with_gaussion_noise_gpu(image, image.shape[0])
    '''




    return image

def random_exposure(image, sigma):
    '''
    use gamma function to random adjust the exposure
    note:the imput image should be normalized to (0,1)
    :param image: image
    :param sigma: a parameter to adjust the distribution that r follows
    :return:
    '''

    assert np.max(image) <= 1
    assert np.min(image) >= 0

    r = np.random.randn(1)*sigma+1

    image = image**r

    image[image < 0] = 0
    image[image > 1] = 1

    return image

def random_contraction(image, sigma1, sigma2):
    '''
        use gamma function to random adjust the exposure
        note:the imput image should be normalized to (0,1)
        :param image: image
        :param sigma: a parameter to adjust the distribution that slope follows
        :return: image
    '''

    assert np.max(image) <= 1
    assert np.min(image) >= 0

    a = np.random.randn(1)*sigma1+1
    b = np.random.randn(1)*sigma2



    image = a*image+b

    image[image<0] = 0
    image[image>1] = 1

    assert np.max(image) <= 1
    assert np.min(image) >= 0

    return image

def random_exposure_gpu(image, sigma):
    '''
    use gamma function to random adjust the exposure
    note:the imput image should be normalized to (0,1)
    :param image: image, gpu_version
    :param sigma: a parameter to adjust the distribution that r follows
    :return:
    '''

    assert torch.max(image) <= 1
    assert torch.min(image) >= 0


    r = torch.randn(1)*sigma+1


    image[0:6,:,:] = image[0:6,:,:]**r.to(image.device)

    image[image < 0] = 0
    image[image > 1] = 1

    return image

def random_contraction_gpu(image, sigma1, sigma2):
    '''
        use gamma function to random adjust the exposure
        note:the imput image should be normalized to (0,1)
        :param image: image
        :param sigma: a parameter to adjust the distribution that slope follows
        :return: image
    '''
    assert torch.max(image) <= 1
    assert torch.min(image) >= 0



    a = torch.randn(1)*sigma1+1
    b = torch.randn(1)*sigma2

    image = a.to(image.device)*image+b.to(image.device)



    image[image<0] = 0
    image[image>1] = 1

    assert torch.max(image) <= 1
    assert torch.min(image) >= 0

    return image

def random_gaussion_noise_gpu(image, mean, std):
    '''
        add gaussion noise to image
        note:the imput image should be normalized to (0,1)
        :param image: image
        :param mean: center of noise
        :return: image
    '''

    assert torch.max(image) <= 1
    assert torch.min(image) >= 0

    noise = torch.Tensor(np.random.normal(0,1,image.numpy().shape))*std+mean
    image = noise+image

    image[image<0] = 0
    image[image>1] = 1

    assert torch.max(image) <= 1
    assert torch.min(image) >= 0

    return image

def random_replace_channel_with_gaussion_noise_gpu(image, total_channel):
    '''
        randomly drop a channel and use gaussion noise to replace
        note:the imput image should be normalized to (0,1)
        :param image: image
        :param total number of channels of image: a parameter to adjust the distribution that slope follows
        :return: image
    '''

    assert torch.max(image) <= 1
    assert torch.min(image) >= 0

    channel_index = torch.randint(0, total_channel, (1,))

    noise = torch.Tensor(np.random.normal(0.5,0.5,image.numpy().shape[1:]))
    image[channel_index.data,:] = noise
    #print(noise)

    image[image<0] = 0
    image[image>1] = 1

    assert torch.max(image) <= 1
    assert torch.min(image) >= 0

    return image





if __name__ == '__main__':
    x = torch.randn(3,256,256,device='cpu')
    x = (x-torch.min(x))/(torch.max(x)-torch.min(x)+1e-8)
    y = augmentation_plus_gpu(x)

    a = torch.nn.functional.binary_cross_entropy(torch.Tensor([1,1,1]),torch.Tensor([1,1,1]))
    print(a)

