"""
Helper file for simulating Mitochondra.

Arif 02/2020

Updated minimum distence among emitters
"""

import numpy as np
import scipy.interpolate as si
from numpy.random import uniform
from scipy import interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from numpy import cos,sin
import skimage.measure
import scipy.misc
from PIL import Image
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
from numpy import pi, cos, sin, arccos, arange
from scipy.interpolate import CubicSpline
from scipy import interpolate

def roll( x,y,z, Euler):               
               

    # Euler rotation                                
    rot = np.dot(                                                
        Euler,                                            
        np.array([x.ravel(), y.ravel(), z.ravel()]) 
    )                                               
    x_rot = rot[0,:].reshape(x.shape)               
    y_rot = rot[1,:].reshape(y.shape)               
    z_rot = rot[2,:].reshape(z.shape)               
    return x_rot, y_rot, z_rot  
    
def bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
                  False - Curve is open
    """

    # If periodic, extend the point array by count+degree+1
    cv = np.asarray(cv)
    count = len(cv)

    if periodic:
        factor, fraction = divmod(count+degree+1, count)
        cv = np.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)
        degree = np.clip(degree,1,degree)

    # If opened, prevent degree from exceeding count-1
    else:
        degree = np.clip(degree,1,count-1)


    # Calculate knot vector
    kv = None
    if periodic:
        kv = np.arange(0-degree,count+degree+degree-1)
    else:
        kv = np.clip(np.arange(count+degree+1)-degree,0,count-degree)

    # Calculate query range
    u = np.linspace(periodic,(count-degree),n)


    # Calculate result
    return np.array(si.splev(u, (kv,cv.T,degree))).T

#Calculate the length of mitochondria in nm

def arc_length(x, y, z):
    npts = len(x)
    arc = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2  + (z[1] - z[0])**2)
    for k in range(1, npts):
        arc = arc + np.sqrt((x[k] - x[k-1])**2 + (y[k] - y[k-1])**2 + (z[k] - z[k-1])**2)

    return arc
    
def arc_length_2d(x, y):
    npts = len(x)
    arc = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2 )
    for k in range(1, npts):
        arc = arc + np.sqrt((x[k] - x[k-1])**2 + (y[k] - y[k-1])**2)

    return arc

#remove random photons from the surface of the sphare
def random_selector(data_x,data_y,data_z,percentage):
    random_list=np.random.randint(0,len(data_x),int(len(data_x)*percentage))
    final_data_x=np.take(data_x, random_list)
    final_data_y=np.take(data_y, random_list)
    final_data_z=np.take(data_z, random_list)
    return final_data_x,final_data_y,final_data_z
    
def random_selector_density(data_x,data_y,data_z,total_emitters):
    random_list=np.random.randint(0,len(data_x),total_emitters)
    final_data_x=np.take(data_x, random_list)
    final_data_y=np.take(data_y, random_list)
    final_data_z=np.take(data_z, random_list)
    return final_data_x,final_data_y,final_data_z
    
   
 
def get_vesicle_3D_points(r,cx,cy,cz,density,percentage):
    
    area=4*np.pi*r*r
    num_pts=int((area/1000)*density)
    indices = arange(0, num_pts, dtype=float) + 0.5
    phi = arccos(1 - 2*indices/num_pts)
    theta = pi * (1 + 5**0.5) * indices
    xi, yi, zi = r*cos(theta) * sin(phi), r*sin(theta) * sin(phi), r*cos(phi);

    xi+=(cx)
    yi+=(cy)
    zi+=(cz)
    
    final_data_x,final_data_y,final_data_z=random_selector(xi,yi,zi,percentage)
    datax=[]
    datay=[]
    dataz=[]
    for k in range(len(final_data_x)):
        datax.append(final_data_x[k])
        datay.append(final_data_y[k])
        dataz.append(final_data_z[k])
    
    data=[]
    data_x=[]
    data_y=[]
    data_z=[]
    for k in range(len(datax)):
            data.append(datax[k])
            data.append(datay[k])
            data.append(dataz[k])
    data_plot=np.array(data)
    data_plot = data_plot.reshape((int(len(data_plot)/3),3))
    return data, data_plot
    
    

    
def get_config(batch):
    data=pd.read_csv('batch/b'+str(batch)+'.csv')
    return data
    
def write_config(batch, num, no_of_vesi, r, total_emitters):
    fowt='param/'+str(batch)+'/'+str(num)+'.txt'
    outF = open(fowt, "w")
    
    outF.write(str(no_of_vesi))
    outF.write("\n")
    
    outF.write(str(r))
    outF.write("\n")
    
   
    outF.write(str(total_emitters))
    outF.write("\n")
    
    outF.close()
    
def write_config2(batch, num, no_of_vesi, r1, r2, total_emitters1, total_emitters2):
    fowt='param/'+str(batch)+'/'+str(num)+'.txt'
    outF = open(fowt, "w")
    
    outF.write(str(no_of_vesi))
    outF.write("\n")
    
    outF.write(str(r1))
    outF.write("\n")
    
    outF.write(str(r2))
    outF.write("\n")
    
        
    outF.write(str(total_emitters1))
    outF.write("\n")
    
    outF.write(str(total_emitters2))
    outF.write("\n")
    
    outF.close()
def generate_save_vesicles(batch, number):
   
    config=get_config(batch)
    
    max_xy=int(config['max_xy'])
    density=int(config['density'])
    number_of_vesicles=int(config['no_of_vesi'])
    number_of_vesicles=np.random.randint(1,number_of_vesicles+1)
    percentage=config['emmiters_percentage']
    
    save_data=[]
    save_data.append('x')
    r=np.random.randint(int(config['min_r']),int(config['max_r']))
    z=np.random.randint(int(config['zlow']),int(config['zhigh']))
    x,y=np.random.randint(-max_xy,max_xy,2)
    data,data_plot=get_vesicle_3D_points(r,x,y,z,density,percentage)
    data1=np.array(data)   
    total_emitters=int(len(data1)/3)
    
    

    data1=np.array(data)   
    data1*=0.001
    for element in data1:
        save_data.append(element)
    if(number_of_vesicles==2):
        r1=np.random.randint(int(config['min_r']),int(config['max_r']))
        z1=np.random.randint(int(config['zlow']),int(config['zhigh']))
        if(number%2==0):
            x1,y1=np.random.randint(-max_xy,max_xy,2)
        else:
            x1=x1+np.random.randint(-4,4)
            y1=y1+np.random.randint(-4,4)
            
        data,data_plot1=get_vesicle_3D_points(r1,x1,y1,z1,density,percentage)
        data2=np.array(data)
        total_emitters1=int(len(data2)/3)
        
        data2*=0.001
        for element in data2:
            save_data.append(element)
        write_config2(batch, number, number_of_vesicles, r, r1, total_emitters,  total_emitters1)
    else:
        write_config(batch, number, number_of_vesicles, r, total_emitters)
    my_df = pd.DataFrame(save_data)
    my_df.to_csv('data/'+str(batch)+'/'+str(number)+'.csv', index=False, header=False)
    return data_plot
    
def save_physics_gt(particlesArray, batch, number, pixel_size, image_size, max_xy):
    particlesArray=np.delete(particlesArray,2,1)
    particlesArray*=1000
    particlesArray=particlesArray
    pimage=np.zeros((pixel_size*image_size,pixel_size*image_size))
    particlesArray+=(2*max_xy)
    pimage[particlesArray[:,1].astype(int),particlesArray[:,0].astype(int)]=255
    pimage2=skimage.measure.block_reduce(pimage, (pixel_size,pixel_size), np.max)
    save_fname='output/'+str(batch)+'/physics_gt/'+str(batch)+"_"+str(number)+'_0.png'
    #scipy.misc.imsave(save_fname, pimage2)
    img = Image.fromarray(pimage2)
    img=img.convert('RGB')
    img.save(save_fname)
    
def save_physics_gt_rgb(particlesArray, batch, number, pixel_size, image_size, max_xy):
    param_fname='param/'+str(batch)+'/'+str(number)+'.txt'
    f=open(param_fname)
    lines=f.readlines()
    number_of_mito=int(lines[0])
    number_of_emitters=int(lines[6])
    
    particlesArray=np.delete(particlesArray,2,1)
    particlesArray*=1000
    particlesArray1=particlesArray[:number_of_emitters, :]
    particlesArray2=particlesArray[number_of_emitters:, :]
    
    pimage=np.zeros((pixel_size*image_size,pixel_size*image_size))
    particlesArray1+=(2*max_xy)
    pimage[particlesArray1[:,1].astype(int),particlesArray1[:,0].astype(int)]=255
    pimage2=skimage.measure.block_reduce(pimage, (pixel_size,pixel_size), np.max)
    
    pimage=np.zeros((pixel_size*image_size,pixel_size*image_size))
    particlesArray2+=(2*max_xy)
    pimage[particlesArray2[:,1].astype(int),particlesArray2[:,0].astype(int)]=255
    pimage3=skimage.measure.block_reduce(pimage, (pixel_size,pixel_size), np.max)
    
    pimage=np.zeros((pixel_size*image_size,pixel_size*image_size))
    pimage4=skimage.measure.block_reduce(pimage, (pixel_size,pixel_size), np.max)
     
    rgbImg = np.zeros((image_size,image_size,3), 'uint8')
    
    rgbImg[..., 0] = pimage2
    rgbImg[..., 1] = pimage3
    rgbImg[..., 2] = pimage4
    img = Image.fromarray(rgbImg)
    img.save('myimg.jpeg')
    
    save_fname='output/'+str(batch)+'/physics_gt_rgb/'+str(batch)+"_"+str(number)+'_0.png'
    #img.save(save_fname)
    #scipy.misc.imsave(save_fname, pimage2)  

