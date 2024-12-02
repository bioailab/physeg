from PIL import Image, ImageOps
import numpy as np
import matlab.engine
from tifffile import imwrite
import os
from pathlib import Path

def generate_image(no,nhigh,nlow,file_list):
    eng = matlab.engine.start_matlab()
    img=Image.new('RGB', (256,256))
    img_seg=Image.new('RGB', (256,256))
    count=4*no
    for r in range(2):
        for c in range(2):
            im1no=count
            batch=int(file_list[im1no][1])
            im1no=int(file_list[im1no][0])
            count=count+1
            if(os.path.isfile('output/'+str(batch)+'/'+str(batch)+'_'+str(im1no)+'_0.tif') and os.path.isfile('output/'+str(batch)+'/physics_gt/'+str(batch)+'_'+str(im1no)+'_0.png')):
                im1 = Image.open('output/'+str(batch)+'/'+str(batch)+'_'+str(im1no)+'_0.tif')
                im3 = Image.open('output/'+str(batch)+'/physics_gt/'+str(batch)+'_'+str(im1no)+'_0.png')
                img.paste(im1, (r*128, c*128))
                img_seg.paste(im3, (r*128, c*128))
            else:
                exit()
    img_seg.save('data/segment/'+str(no)+'.png')
    save_fname='data/original/'+str(no)+'.tif'
    img=np.asarray(img)
    img=img[:,:,0]
    img=img.astype(np.uint16)
    imwrite(save_fname,img,compress=6)
    save_noise_fname='data/noisy/'+str(no)+'.tif'
    x=eng.add_poisson_noise_image(save_fname,int(nhigh),int(nlow),save_noise_fname)  #sending input to the function
    img=Image.open(save_noise_fname).convert('LA')
    img.save('data/image/'+str(no)+'.png')

Path('data').mkdir(parents=True, exist_ok=True)
Path('data/segment').mkdir(parents=True, exist_ok=True)
Path('data/original').mkdir(parents=True, exist_ok=True)
Path('data/noisy').mkdir(parents=True, exist_ok=True)
Path('data/image').mkdir(parents=True, exist_ok=True)

np.random.seed(5)#for reproducibility
total_image_in_a_batch=20#simulation batch size
total_batch=1#simualtion batch size

total_image=int((total_image_in_a_batch*total_batch)/4)#
file_list = [[[i,j] for i in range (0,total_image_in_a_batch)] for j in range (1,total_batch+1)]
file_list=np.array(file_list)
file_list=file_list.reshape(total_image_in_a_batch*total_batch,2)
np.random.shuffle(file_list)

for i in range(0,total_image):
    nhigh=np.random.randint(160,220)
    nlow=np.random.randint(20,40)
    generate_image(i,nhigh,nlow,file_list)
