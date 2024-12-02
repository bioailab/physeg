import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing
from skimage.transform import downscale_local_mean
from skimage.util import random_noise
import skimage.io
import microscPSFmod as msPSF
import os
from tifffile import imwrite
import matplotlib.pyplot as plt
import matlab.engine
from skimage import filters
from pathlib import Path
import cv2
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from multiprocessing import Pool
from itertools import repeat
from helper_generator_v6 import generate_save_vesicles, get_config, save_physics_gt, save_physics_gt_rgb

#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["NUMEXPR_NUM_THREADS"] = "1"
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_MAIN_FREE"] = "1"


def process_matrix(locations, size_x, step_size_xy, psf_size_x, stage_delta, sampling, mp, wvl=0.510):
    result = np.zeros((locations.shape[0], size_x * size_x))
    for i in range(locations.shape[0]):
        if(i % 100 == 0):
            print(i)
        psf = msPSF.gLXYZParticleScan(
            mp, step_size_xy, psf_size_x, np.array(locations[i, 2]),
            zv=stage_delta, wvl=wvl, normalize=False,
            px=locations[i, 0], py=locations[i, 1])
        psf_sampled = downscale_local_mean(psf, (1, sampling, sampling))
        result[i, :] = psf_sampled.ravel()
    return result

def process_matrix_all_z(locations, size_x, step_size_xy, psf_size_x, stage_delta, sampling, mp, wvl=0.510):
    psf = msPSF.gLXYZParticleScan(
        mp, step_size_xy, psf_size_x, locations[:,2],
        zv=stage_delta, wvl=wvl, normalize=False,
        px=locations[:, 0], py=locations[:, 1])
    #psf_sampled = downscale_local_mean(psf, (1, sampling, sampling))
    #result = np.reshape(psf_sampled, (psf_sampled.shape[0], -1))

    #return result
    return np.reshape(psf, (psf.shape[0], -1))



def brightness_trace(t_off, t_on, rate, frames):

    T = np.zeros((2, frames))
    T[0, :] = np.random.exponential(scale=t_off, size=(1, frames))
    T[1, :] = np.random.exponential(scale=t_on, size=(1, frames))

    B = np.zeros((2, frames))
    B[1, :] = rate * T[1, :]

    T_t = T.ravel(order="F")
    B_t = B.ravel(order="F")

    T_cum = np.cumsum(T_t)
    B_cum = np.cumsum(B_t)

    start = np.random.randint(0, 10)

    br = np.diff(np.interp(np.arange(start, start + frames + 1), T_cum, B_cum))
    #br = 100
    return br

def add_noise(img,max_sig,offset):

    img2=((max_sig-offset))*img+(offset)
    #noise_mask = np.random.poisson(img2)
    noise_mask=random_noise(img2, mode='poisson')
    plt.figure()
    plt.imshow(noise_mask,cmap='gray')
    noisy_img = img2 + noise_mask
    
    return noisy_img
    

        
def generate_vesicles(batch, number, nlow, nhigh):
    np.random.seed(0)

    # Output parameters
    size_x = 128  # size of x/y in pixels (image)
    size_t = 1 # number of acquisitions

    # System parameters
    step_size_xy = 0.042  # [um] pixel size
    stage_delta = -1  # [um] negative means toward the objective
    mp = msPSF.m_params  # Default parameters
    sampling = 1
    psf_size_x = sampling * size_x

    # Fluorophore parameters
    t_off = 0 # changed by krishna, needs to be checked, original value 8
    t_on = 1 # changed by krishna, needs to be checked, original value 2
    rate = 10
    
    fname='data/'+str(batch)+'/'+str(number)+'.csv'
    my_df = pd.read_csv(fname)
    particlesArray = my_df.to_numpy()
    particlesArray_gt = my_df.to_numpy()
    particlesArray = particlesArray.reshape((int(len(particlesArray)/3),3))
    particlesArray_gt = particlesArray_gt.reshape((int(len(particlesArray_gt)/3),3))
    brightness = np.zeros((particlesArray.shape[0], size_t))
    for i in range(particlesArray.shape[0]):
        brightness[i, :] = brightness_trace(t_off, t_on, rate, size_t)
    result = {}
    n_process =1
    particlesData = np.zeros((particlesArray.shape[0], size_x * size_x))

    if n_process > 1:
        partitions = []
        step_partition = particlesArray.shape[0] // n_process
        idxs = list(range(0, particlesArray.shape[0], step_partition))
        if idxs[-1] < particlesArray.shape[0] - 1:
            idxs.append(particlesArray.shape[0] - 1)
        for i in range(len(idxs) - 1):
            partitions.append(particlesArray[idxs[i]:idxs[i+1], :])
        pool = multiprocessing.Pool(n_process)
        results = {}
        for i in range(n_process):
            results["process" +
                    str(i)] = pool.apply_async(process_matrix, (partitions[i], size_x, step_size_xy, psf_size_x, stage_delta, sampling, mp))
            print(f"Process {i} loaded")
        for i in range(n_process):
            particlesData[idxs[i]:idxs[i+1], :] = results["process" + str(i)].get()

        pool.close()
    
    else:
        particlesData = process_matrix_all_z(particlesArray, size_x, step_size_xy, psf_size_x, stage_delta, sampling, mp)
        
    # Image generation
    image = np.zeros((size_t, size_x, size_x))
    for t in range(size_t):
        #print(f"T{t}")
        b = brightness[:, t]
        image[t, :, :] = np.reshape(
            np.sum(particlesData * b[:, None], axis=0), (size_x, size_x))
        #poisson_noise = np.random.poisson(image[t, :, :])
        #image[t, :, :] += poisson_noise
    #eng = matlab.engine.start_matlab()
    for t in range(size_t):
        d=image[t, :, :]
        d/=np.max(d)
        d*=255
        d=d.astype(np.uint16)
        save_fname='output/'+str(batch)+'/'+str(batch)+"_"+str(number)+'_'+str(t)+'.tif'
        save_noise_fname='output/'+str(batch)+'/N_'+str(batch)+"_"+str(number)+'_'+str(t)+'.tif'
        
        if(t==0):
            th=int((nlow/nhigh)*255)
            seg_noise=d<th
            seg_noise2=d<(th/2)
            save_seg_noise_fname='output/'+str(batch)+'/segn/'+str(batch)+"_"+str(number)+'.tif'
            save_seg_noise_fname2='output/'+str(batch)+'/segn2/'+str(batch)+"_"+str(number)+'.tif'
            #imwrite(save_seg_noise_fname,seg_noise)
            #imwrite(save_seg_noise_fname2,seg_noise2)
            d1=cv2.UMat(np.array(d, dtype=np.uint8))
            #d = cv2.GaussianBlur(d,(1,1),0)
            #seg = filters.threshold_otsu(d)
            #print(d)
            ret2,seg= cv2.threshold(d1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #print(type(seg))
            kernel = np.ones((2,2),np.uint8)
            save_seg_fname1='output/'+str(batch)+'/otsu/'+str(batch)+"_"+str(number)+'.tif'
            #cv2.imwrite(save_seg_fname1,seg)
            seg = cv2.erode(seg,kernel,iterations = 1)
            #seg=np.asarray(seg)
            save_seg_fname='output/'+str(batch)+'/seg/'+str(batch)+"_"+str(number)+'_seg'+'.tif'
            #cv2.imwrite(save_seg_fname,seg)
            save_physics_gt(particlesArray, batch, number, 42, 128, 1344) # todo: replace last 3 parameters
            #save_physics_gt_rgb(particlesArray_gt, batch, number, 80, 100, 2000) # todo: replace last 3 parameters
            #imwrite(save_seg_fname,seg,compress=6)
        imwrite(save_fname,d,compress=6)
        #print(save_fname,save_noise_fname)
        #x=eng.add_poisson_noise_image(save_fname,int(nhigh),int(nlow),save_noise_fname)  #sending input to the function

def parallel_run(batch,i,nlow,nhigh):
    generate_save_vesicles(batch,i)
    generate_vesicles(batch,i,nlow,nhigh)
    
def generate_save_vesicles_in_batch(total_sample,batch):
    num_cores = 20#multiprocessing.cpu_count()
    config=get_config(batch)
    nlow=np.random.randint(40,70,total_sample+1)
    nhigh=np.random.randint(200, 240,total_sample+1)
    s=range(total_sample)
    #for i in range(1,total_sample+1):
    #    print('generating'+str(i))
    #    parallel_run(batch, i,nlow[i],nhigh[i])
        #generate_save_vesicles(batch,i)
    #    print('generating'+str(i))
    pool = Pool(num_cores) # four parallel jobs
    # Compute
    #with Pool(processes = num_cores) as p:
    #    results = p.starmap_async(parallel_run, [(batch, i,nlow[i],nhigh[i]) for i in range(total_sample)])
    for i in range(total_sample):
        pool.apply_async(parallel_run, args=(batch, i,nlow[i],nhigh[i]))
    pool.close()
    pool.join()  # postpones the execution of next line of code until all processes in the queue are done.

        
if __name__ == "__main__":
    batch=1
    number_of_sample=20
    
    
    Path('output/'+str(batch)+'/param').mkdir(parents=True, exist_ok=True)
    Path('output/'+str(batch)+'/otsu').mkdir(parents=True, exist_ok=True)
    Path('output/'+str(batch)+'/seg').mkdir(parents=True, exist_ok=True)
    Path('output/'+str(batch)+'/segn').mkdir(parents=True, exist_ok=True)
    Path('output/'+str(batch)+'/segn2').mkdir(parents=True, exist_ok=True)
    Path('output/'+str(batch)+'/physics_gt').mkdir(parents=True, exist_ok=True)
    Path('output/'+str(batch)+'/physics_gt_rgb').mkdir(parents=True, exist_ok=True)
    Path('param/'+str(batch)).mkdir(parents=True, exist_ok=True)
    from pathlib import Path

    data_path="data/"+str(batch)
    param_path="param/"+str(batch)
    Path(data_path).mkdir(parents=True, exist_ok=True)
    Path(param_path).mkdir(parents=True, exist_ok=True)
    
    generate_save_vesicles_in_batch(number_of_sample,batch)