import pandas as pd
import numpy as np
from matplotlib.pyplot import imshow, show, subplots
import skimage as sk
from skimage import segmentation as seg
from skimage import filters, util
import os

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

dataset=pd.read_csv('dataset_classify.csv',delimiter=",")
print(dataset)

Wkt=np.array(dataset['Waktu Hujan'])
lintang0=pd.read_csv('lintang.csv',header=None)
lintang_1=lintang0.iloc[::-1]
bujur0=pd.read_csv('bujur.csv',header=None)
lintang=np.array(dataset['Latitude'])
bujur=np.array(dataset['Longitude'])

t=5.9
def get_max_cell(ev,time):
    im0=sk.io.imread('/Users/cin/Documents/pyfile/segmentasi/testing/'+ev+'/'+time+'.tif')
    im_arr=np.array(im0)
    Maks_point=np.max(im_arr)
    ind_y,ind_x=np.where(im_arr==Maks_point)
    return ind_x[0],ind_y[0]


def get_char_cell(bj,lt,ev,time):
    print(time)
    im=sk.io.imread('/Users/cin/Documents/pyfile/segmentasi/testing/'+ev+'/'+time+'.tif')
    mask=im>t
    ima=im.copy()
    ima[~mask]=0
    #print(ima[400,400])
    ima_sobel=filters.sobel(mask)
    ima_sobel_inv=util.invert(ima_sobel)
    mask2=ima_sobel_inv==1
    points=(bj,lt)
    print(points)
    snake=seg.flood_fill(ima_sobel,points,new_value=1)
    snake[~mask]=0
    snake[~mask2]=0
    mask_snake=snake==1
    #print(np.max(snake))
    imb=im.copy()
    imb[~mask_snake]=0

    sk.io.imsave('/Users/cin/Documents/pyfile/mask_file/'+str(time)+'.bmp',snake)

    luas_peta=400*400
    Jumlah_piksel=np.size(snake)
    Piksel_cell=np.sum(snake)
    luas=Piksel_cell*luas_peta/Jumlah_piksel
    #print(luas_peta,Jumlah_piksel,Piksel_cell,luas)

    Maks_hujan=np.max(imb)
    Rata2_hujan=np.sum(imb)/luas
    print(luas,Rata2_hujan,Maks_hujan)
    return luas,Rata2_hujan,Maks_hujan

#fig,(ax1,ax2,ax3)=subplots(3)
#imshow(ima,cmap='gray')
#ax1.imshow(ima_sobel_inv,cmap='gray')
#ax2.imshow(ima_sobel,cmap='gray')
#ax3.imshow(snake,cmap='gray')
#show()

fileall=[]
for root,dirnames,filenames in os.walk('/Users/cin/Documents/pyfile/segmentasi/testing/'):
    for dir in dirnames:
        fileall.append(dir)
print(fileall)

data_waktu=[]
data_luas=[]
data_chmax=[]
data_chmean=[]
data_benc=[]

for f in fileall:
    path="/Users/cin/Documents/pyfile/segmentasi/testing/"+f
    dict=[]
    os.chdir(path)
    files=sorted(os.listdir(os.getcwd()))
    for i in files:
        if i.endswith(".tif"):
            dict.append(i)
    print(dict)
    for d in dict:
        waktu=d[:16]
        if waktu in Wkt:
            ind_tgl=np.where(Wkt==waktu)
            ind_tgl=ind_tgl[0]
            for i in ind_tgl:
                bjr=find_nearest(bujur0,bujur[i])
                ltg=find_nearest(lintang_1,lintang[i])     
        elif waktu not in Wkt:
            bjr,ltg=get_max_cell(f,waktu)
        ch_area,ch_mean,ch_max=get_char_cell(bjr,ltg,f,waktu)
        data_waktu.append(waktu)
        data_luas.append(ch_area)
        data_chmax.append(ch_max)
        data_chmean.append(ch_mean)
        data_benc.append(f)

dataset=pd.DataFrame(list(zip(data_waktu,data_luas,data_chmax,data_chmean,data_benc)),columns=['waktu','luas cell','ch maks','ch rata2','bencana'])
print(dataset)
dataset.to_csv('/Users/cin/Documents/pyfile/dataset_testing_ML_classify.csv',index=False)