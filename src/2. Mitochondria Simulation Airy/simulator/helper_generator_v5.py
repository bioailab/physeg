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
    
def get_mitochondria_2D_points(zhigh=800, zlow=600, max_xy=3000, max_length=5000, curve_point=3):
    
    zhigh1=zhigh
    while True:
        ###inputs
        #p1=np.random.uniform(-max_xy,max_xy,(curve_point)) #first point
        #p2=np.random.uniform(-max_xy,max_xy,(curve_point)) #second point
        
        p=np.random.uniform(-max_xy,max_xy,(max_xy))
        np.random.shuffle(p)
        p1=np.random.choice(p,curve_point)
        p2=np.random.choice(p,curve_point)
        p1.sort()
        tck,u = interpolate.splprep([p1,p2], k=2,s=0)
        unew = np.arange(0, 1, 0.01)
        out = interpolate.splev(unew, tck)
        x=out[0]
        y=out[1]
        arc_2d=arc_length_2d(x, y);
        
        if(arc_2d<(zhigh-zlow)):
            zhigh1=zlow+int(arc_2d)
        z=np.linspace(zlow,zhigh1,len(x))
        dist=arc_length(x,y,z)
        #print(dist)
        #if(dist<max_length or np.min(x)<-max_xy or  np.min(y)<-max_xy or  np.max(x)>max_xy  or  np.max(y)>max_xy):
        if(dist<max_length and np.min(x)>-max_xy and  np.min(y)>-max_xy and  np.max(x)<max_xy  and  np.max(y)<max_xy):
            break;
            
    number_of_points=int(dist)
    
    unew = np.arange(0, 1, 1/arc_2d)
    out = interpolate.splev(unew, tck)
    x1=out[0]
    y1=out[1]
    z1=np.linspace(zlow,zhigh1,len(x1))
    
    return x1, y1, z1, number_of_points, dist, p1, p2

def get_points(r,p1,p2,num_pts,num_of_points_in_circle):
    
    
    num_pts=np.random.randint(int(num_pts*0.8),num_pts)
    indices = arange(0, num_pts, dtype=float) + 0.5
    phi = arccos(1 - 2*indices/num_pts)
    theta = pi * (1 + 5**0.5) * indices
    xi, yi, zi = r*cos(theta) * sin(phi), r*sin(theta) * sin(phi), r*cos(phi);

    xi+=(p1[0])
    yi+=(p1[1])
    zi+=(p1[2])
    b = np.array([p1[0], p1[1], p1[2]])
    c = np.array([p2[0], p2[1], p2[2]])
    
    a = np.array([xi, yi,zi])
    ba=np.subtract(a.T,b)
    bc = c-b
    d=np.dot(ba, bc)
    m= np.multiply(np.linalg.norm(ba,axis=-1) , np.linalg.norm(bc))
    cosine_angle = d / m
    angle = np.arccos(cosine_angle)
    theta=np.degrees(angle)
    xii=xi[np.where((theta>88) & (theta<92))]
    yii=yi[np.where((theta>88) & (theta<92))]
    zii=zi[np.where((theta>88) & (theta<92))]
    random_list=np.random.randint(0,len(xii),num_of_points_in_circle)
    final_data_x=np.take(xii, random_list)
    final_data_y=np.take(yii, random_list)
    final_data_z=np.take(zii, random_list)
    return final_data_x,final_data_y,final_data_z
    
 
def get_mitochondria_3D_points(x, y, z, dist, width=500,density=10, percentage=0.9):
    """
    Conver 2D points to 3D volume with random photon.
    width: is the width of the mitochondria
    percentage: is the amount of photon will present
    """
    r=int(width/2)
    area=2*np.pi*r*dist
    total_expected_emitters=int((area/1000)*density)
    num_of_points_in_circle=int(math.floor(total_expected_emitters/dist))
    print(num_of_points_in_circle)
    prv_ang=0
    dp=5
    datax=[]
    datay=[]
    dataz=[]
    xx=[]
    yy=[]
    zz=[]
    num_pts=2*np.pi*r*5
    for i in range(len(x)-1):
        p1=np.array([x[i],y[i],z[i]])
        p2=np.array([x[i+1],y[i+1],z[i+1]])    
        
        data_x,data_y,data_z=get_points(r,p1,p2,num_pts,num_of_points_in_circle) 
        final_data_x,final_data_y,final_data_z=random_selector(data_x,data_y,data_z,1)
        
        for k in range(len(final_data_x)):
            datax.append(final_data_x[k])
            datay.append(final_data_y[k])
            dataz.append(final_data_z[k])
    
    data=[]
    for k in range(len(datax)):
            data.append(datax[k])
            data.append(datay[k])
            data.append(dataz[k])
    data_plot=np.array(data)
    data_plot = data_plot.reshape((int(len(data_plot)/3),3))
    return data, data_plot
    
def get_mitochondria_3D_points_rot(x, y, z, dist, width=500,density=10, percentage=0.9):
    """
    Conver 2D points to 3D volume with random photon.
    width: is the width of the mitochondria
    percentage: is the amount of photon will present
    """
    twopi = 2.0 * np.pi
    r=int(width/2)
    area=twopi*r*dist
    no_of_points=int(((area/10000)*density))
    #print(no_of_points)
    
    prv_ang=0
    dp=5
    data=[]
    xx=[]
    yy=[]
    zz=[]
    for i in range(len(x)-1):
        p1=np.array([x[i],y[i],z[i]])
        p2=np.array([x[i+1],y[i+1],z[i+1]])    
        
        data_x=[]
        data_y=[]
        data_z=[]
        theta=np.random.uniform(0,twopi,no_of_points)
        xd=r*cos(theta)
        yd=r*sin(theta)
        zd=0
        
        for k in range(len(xd)):
            data_x.append(xd[k])
            data_y.append(yd[k])
            data_z.append(zd)
        final_data_x,final_data_y,final_data_z=random_selector(data_x,data_y,data_z,percentage)
        
        theta = (np.arctan2(p2[2] - p1[2], p2[1] - p1[1]))
        rx=np.array([[1,0,0],[0,math.cos(theta),-math.sin(theta)],[0,math.sin(theta),math.cos(theta)]])
        xx,yy,zz=roll(final_data_x,final_data_y,final_data_z,rx)

        theta = (np.arctan2(p2[0] - p1[0], p2[2] - p1[2]))
        ry=np.array([[math.cos(theta),0,math.sin(theta)],[0,1,0],[-math.sin(theta),0,math.cos(theta)]])
        xx,yy,zz=roll(xx,yy,zz,ry)
        
        theta = (np.arctan2(p2[0] - p1[0], p2[1] - p1[1]))
        rz=np.array([[math.cos(theta),-math.sin(theta),0],[math.sin(theta),math.cos(theta),0],[0,0,1]])
        xx,yy,zz=roll(xx,yy,zz,rz)
        
        sx=x[i]-np.mean(xx)
        sy=y[i]-np.mean(yy)
        sz=z[i]-np.mean(zz)
        for k in range(len(xx)):
            data.append(xx[k]+sx)
            data.append(yy[k]+sy)
            data.append(zz[k]+sz)
        data_plot=np.array(data)
        data_plot = data_plot.reshape((int(len(data_plot)/3),3))
    return data, data_plot
    
def get_mitochondria_3D_points_backup(x, y, z, dist, width=500,density=10, percentage=0.9):
    """
    Conver 2D points to 3D volume with random photon.
    width: is the width of the mitochondria
    percentage: is the amount of photon will present
    """
    twopi = 2.0 * np.pi
    r=int(width/2)
    area=twopi*r*dist
    no_of_points=int(((area/1000)*density)/dist)
    #print(no_of_points)
    
    data=[]
    for p in range(0,len(x)):
        data_x=[]
        data_y=[]
        data_z=[]
        theta=np.random.uniform(0,twopi,no_of_points)
        xd=r*cos(theta)
        yd=r*sin(theta)
        zd=z[p]
        for i in range(len(xd)):
            data_x.append(xd[i])
            data_y.append(yd[i])
            data_z.append(zd)
        final_data_x,final_data_y,final_data_z=random_selector(data_x,data_y,data_z,percentage)
        for k in range(len(final_data_x)):
            data.append((x[p]+final_data_x[k]))
            data.append((y[p]+final_data_y[k]))
            data.append((final_data_z[k]))
        data_plot=np.array(data)
        data_plot = data_plot.reshape((int(len(data_plot)/3),3))
    return data, data_plot
    
def get_config(batch):
    data=pd.read_csv('batch/b'+str(batch)+'.csv')
    return data
    
def write_config(batch, num, no_of_mito, width, length, total_emitters):
    fowt='param/'+str(batch)+'/'+str(num)+'.txt'
    outF = open(fowt, "w")
    
    outF.write(str(no_of_mito))
    outF.write("\n")
    
    outF.write(str(width))
    outF.write("\n")
    
    outF.write(str(length))
    outF.write("\n")
    
    outF.write(str(total_emitters))
    outF.write("\n")
    
    outF.close()
    
def write_config2(batch, num, no_of_mito, width1, length1, width2, length2, total_emitters1, total_emitters2):
    fowt='param/'+str(batch)+'/'+str(num)+'.txt'
    outF = open(fowt, "w")
    
    outF.write(str(no_of_mito))
    outF.write("\n")
    
    outF.write(str(width1))
    outF.write("\n")
    
    outF.write(str(length1))
    outF.write("\n")
    
    outF.write(str(width2))
    outF.write("\n")
    
    outF.write(str(length2))
    outF.write("\n")
    
    outF.write(str(total_emitters1))
    outF.write("\n")
    
    outF.write(str(total_emitters2))
    outF.write("\n")
    
    outF.close()
def generate_save_mito(batch, number, width, plot=False):
    if(plot):
        fig2 = plt.figure(1)
        ax = fig2.add_subplot(111, projection='3d')
    config=get_config(batch)
    
    zlow=int(config['zlow'])
    zhigh=int(config['zhigh'])
    max_xy=int(config['max_xy'])
    max_mito_length=int(config['max_length'])
    density=int(config['density'])
    number_of_mito=int(config['no_of_mito'])
    
    percentage=config['emmiters_percentage']
    control_points_lower=config['control_points_low']
    control_points_up=config['control_points_up']+1
    control_points=np.random.randint(control_points_lower,control_points_up)
    #print(zhigh,zlow,max_xy,max_mito_length)
    x1, y1, z1, number_of_points1, dist1, p1, p2=get_mitochondria_2D_points(zhigh,zlow,max_xy,max_mito_length,control_points)
    
    
    save_data=[]
    save_data.append('x')
    data,data_plot=get_mitochondria_3D_points(x1,y1,z1,dist1,width,density,percentage)
    data1=np.array(data)   
    total_emitters=int(len(data1)/3)
    area=2*math.pi*int(width/2)*dist1
    count=int((total_emitters/area)*1000)
    #print(total_emitters,int(dist1),count)
    
    if(plot):
        #x = np.arange(1400,3)
        data3 = data1.reshape((int(len(data1)/3),3))
        cmhot = plt.get_cmap("jet")
        ff=ax.scatter(data3[:,0],data3[:,1],data3[:,2],c=data3[:,2],cmap=cmhot)
        #fig2.colorbar(ff, orientation='horizontal')
        #ax.plot(data3[:,0],data3[:,1],data3[:,2],'.g')
        ax.set_xlim(-max_xy, max_xy)
        ax.set_ylim(-max_xy, max_xy)
        ax.set_zlim(0, 1400)
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y  (nm)')
        ax.set_zlabel('z  (nm)')
        ax.auto_scale_xyz([-max_xy, max_xy], [-max_xy, max_xy], [0, 1400])
        plt.show()

    data1=np.array(data)   
    data1*=0.001
    for element in data1:
        save_data.append(element)
    if(number_of_mito==2):
        x2, y2, z2, number_of_points2, dist2, p3, p4=get_mitochondria_2D_points(zhigh,zlow,max_xy,max_mito_length,control_points)
        width2=np.random.randint(int(config['wlow']),int(config['whigh']))
        data,data_plot1=get_mitochondria_3D_points(x2,y2,z2,dist2,width2,density,percentage)
        data2=np.array(data)
        total_emitters1=int(len(data2)/3)
        if(plot):
            data4=np.array(data)
            data5 = data4.reshape((int(len(data4)/3),3))
            ax.plot(data5[:,0],data5[:,1],data5[:,2],'.r')
            plt.show()
        data2*=0.001
        for element in data2:
            save_data.append(element)
        write_config2(batch, number, number_of_mito, width, dist1, width2, dist2,  total_emitters,  total_emitters1)
        
    else:
        write_config(batch, number, number_of_mito, width, dist1, total_emitters)
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
    #from skimage.transform import downscale_local_mean
    #pimage2=downscale_local_mean(pimage, (100, 100))
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
    if(number_of_mito==2):
        number_of_emitters=int(lines[5])
        
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
        #img.save('myimg.jpeg')
        
        save_fname='output/'+str(batch)+'/physics_gt_rgb/'+str(batch)+"_"+str(number)+'_0.png'
        img.save(save_fname)
        #scipy.misc.imsave(save_fname, pimage2)  
    else:
        number_of_emitters=int(lines[3])
        
        particlesArray=np.delete(particlesArray,2,1)
        particlesArray*=1000
        particlesArray1=particlesArray[:number_of_emitters, :]
        
        pimage=np.zeros((pixel_size*image_size,pixel_size*image_size))
        particlesArray1+=(2*max_xy)
        pimage[particlesArray1[:,1].astype(int),particlesArray1[:,0].astype(int)]=255
        pimage2=skimage.measure.block_reduce(pimage, (pixel_size,pixel_size), np.max)
        
         
        rgbImg = np.zeros((image_size,image_size,3), 'uint8')
        
        rgbImg[..., 0] = pimage2
        img = Image.fromarray(rgbImg)
        #img.save('myimg.jpeg')
        
        save_fname='output/'+str(batch)+'/physics_gt_rgb/'+str(batch)+"_"+str(number)+'_0.png'
        img.save(save_fname)
        #scipy.misc.imsave(save_fname, pimage2)