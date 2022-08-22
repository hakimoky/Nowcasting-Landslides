import wradlib as wrl
#import wradlib.georef as georef
import numpy as np
import pandas as pd
from osgeo import osr
#from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as pl
import os
from datetime import datetime
from pyproj import Proj
from scipy import ndimage
import csv

fileall=[]
for root,dirnames,filenames in os.walk('/Users/cin/Documents/pyfile/Radar/2018/201802/'):
    for dir in dirnames:
        fileall.append(dir)
        #if file.endswith('dBZ.vol'):
        #    files.append(os.path.join(root,file))
urut=sorted(fileall)
#urut=urut[27:]
print(urut)

for u in urut:
    os.environ['WRADLIB_DATA']='/Users/cin/Documents/pyfile/Radar/2018/201802/'+u
    pl.interactive(True)

    path="/Users/cin/Documents/pyfile/Radar/2018/201802/"+u
    dict=[]
    os.chdir(path)
    files=sorted(os.listdir(os.getcwd()))
    for i in files:
        if i.endswith("dBZ.vol"):
            dict.append(i)
    print(dict)

    df=pd.read_csv('/Users/cin/Documents/pyfile/DaftarPosHujan.csv',delimiter=";")
    Buj_Pos=(np.array(df["Bujur"])).tolist()
    Lin_Pos=(np.array(df["Lintang"])).tolist()
    #Nama_Pos=(np.array(df["Nama_Pos"])).tolist()
    #Nama_Pos.insert(0,"Waktu")
    #with open ("Pos_Refl.csv","w") as f:
    #    write=csv.writer(f)
    #    write.writerow(Nama_Pos)
    #print(Buj_Pos)
    #print(Lin_Pos)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    #proyeksi
    proj=osr.SpatialReference()
    proj.ImportFromEPSG(23889)
    p=Proj("epsg:23889", preserve_units=True)

    #Buat Koordinat awal
    fpath0=dict[0]
    f0=wrl.util.get_wradlib_data_file(fpath0)
    raw0=wrl.io.read_rainbow(f0)

    #koordinat pusat radar
    radarLon0=float(raw0['volume']['sensorinfo']['lon'])
    radarLat0=float(raw0['volume']['sensorinfo']['lat'])
    radarAlt0=float(raw0['volume']['sensorinfo']['alt'])
    sitecoords=(radarLon0,radarLat0,radarAlt0)
    #print(sitecoords)

    xyz0=np.array([]).reshape((-1, 3))
    nElevation0=len(raw0['volume']['scan']['slice']) # jumlah seluruh elevasi
    elevs0=[]
    #list_stoprange0=[]
    #list_azirange0=[]
    for i in range(nElevation0):
        elevation0= float(raw0['volume']['scan']['slice'][i]['posangle'])
        elevs0.append(elevation0)

        # ekstrak azimuth data
        try:
            azi0 = raw0['volume']['scan']['slice'][i]['slicedata']['rayinfo']['data']
            azidepth0 = float(raw0['volume']['scan']['slice'][i]['slicedata']['rayinfo']['@depth'])
            azirange0 = float(raw0['volume']['scan']['slice'][i]['slicedata']['rayinfo']['@rays'])
            #list_azirange0.append(azirange0)  
        except:
            azi00 = raw0['volume']['scan']['slice'][i]['slicedata']['rayinfo'][0]['data']
            azi10 = raw0['volume']['scan']['slice'][i]['slicedata']['rayinfo'][1]['data']
            azi0 = (azi00/2) + (azi10/2)
            del azi00, azi10
            azidepth0 = float(raw0['volume']['scan']['slice'][i]['slicedata']['rayinfo'][0]['@depth'])
            azirange0 = float(raw0['volume']['scan']['slice'][i]['slicedata']['rayinfo'][0]['@rays'])
            #list_azirange0.append(azirange0)            
        try:
            azires0 = float(raw0['volume']['scan']['slice'][i]['anglestep'])
        except:
            azires0 = float(raw0['volume']['scan']['slice'][0]['anglestep'])
        azims0=(azi0*azirange0/2**azidepth0)*azires0
        #print(azims0)

        # esktrak range data
        try:
            stoprange0 = float(raw0['volume']['scan']['slice'][i]['stoprange'])
            rangestep0 = float(raw0['volume']['scan']['slice'][i]['rangestep'])
            #list_stoprange0.append(stoprange0)
        except:
            stoprange0 = float(raw0['volume']['scan']['slice'][0]['stoprange'])
            rangestep0 = float(raw0['volume']['scan']['slice'][0]['rangestep'])
            #list_stoprange0.append(stoprange0)
        ranges0=np.arange(0,stoprange0,rangestep0)*1000
        #print(ranges0)
        polxyz0=wrl.vpr.volcoords_from_polar(sitecoords,elevation0,azims0,ranges0,proj)
        xyz0=np.vstack((xyz0,polxyz0))
    elevs0=np.array(elevs0)
    #print(elevs0)
    #print(list_stoprange0)
    #print(list_azirange0)
    #print(azi0)
    #print(azirange0)
    #print(azidepth0)
    #print(azires0)
    #print(stoprange0)
    #print(rangestep0)

    #membuat grid koordinat cartesian 3D
    maxrange=200000.
    minelev0=elevs0.min()
    maxelev0=elevs0.max()
    maxalt=10000.
    horiz_res=400.
    vert_res=1000.
    trgxyz0,trgshape0=wrl.vpr.make_3d_grid(sitecoords,proj,maxrange,maxalt,horiz_res,vert_res)
    #print(trgxyz)
    #print(trgshape)
    gridder0=wrl.vpr.PseudoCAPPI(xyz0,trgxyz0,trgshape0,maxrange,minelev0,maxelev0,ipclass=wrl.ipol.Idw)
    print(gridder0)

    for j in range(len(dict)):
        print("data ke ",j)
        fpath=dict[j]
        f=wrl.util.get_wradlib_data_file(fpath)
        raw=wrl.io.read_rainbow(f)

        #menentukan waktu pengamatan
        nSlices=len(raw['volume']['scan']['slice'])
        date=(raw['volume']['scan']['slice'][nSlices-1]['slicedata']['@date'])
        time=(raw['volume']['scan']['slice'][nSlices-1]['slicedata']['@time'])
        try:timeEnd=datetime.strptime('{}{}'.format(date,time),"%Y-%m-%d%H:%M:%S")
        except:timeEnd=datetime.strptime('{}{}'.format(date,time),"%Y-%m-%d%H:%M:%S.%f")

        #koordinat pusat radar
        #radarLon=float(raw['volume']['sensorinfo']['lon'])
        #radarLat=float(raw['volume']['sensorinfo']['lat'])
        #radarAlt=float(raw['volume']['sensorinfo']['alt'])
        #sitecoords=(radarLon,radarLat,radarAlt)
        #print(sitecoords)

        #ambil data
        data=np.array([])
        nElevation=len(raw['volume']['scan']['slice']) # jumlah seluruh elevasi
        #elevs=[]
        #data=[]
        for k in range(nElevation):
            #elevation= float(raw['volume']['scan']['slice'][i]['posangle'])
            #elevs.append(elevation)

            #ekstrak data
            data_=raw['volume']['scan']['slice'][k]['slicedata']['rawdata']['data']
            datadepth=float(raw['volume']['scan']['slice'][k]['slicedata']['rawdata']['@depth'])
            datamin=float(raw['volume']['scan']['slice'][k]['slicedata']['rawdata']['@min'])
            datamax=float(raw['volume']['scan']['slice'][k]['slicedata']['rawdata']['@max'])
            data_=datamin+data_*(datamax-datamin)/2** datadepth
            #print(np.size(data_))

            #hilangkan clutter dan atenuasi
            clutter=wrl.clutter.filter_gabella(data_, tr1=12, n_p=6, tr2=1.1)
            data_nc=wrl.ipol.interpolate_polar(data_, clutter)
            pia_mkraemer = wrl.atten.correct_attenuation_constrained(
            data_nc,
            a_max=1.67e-4,
            a_min=2.33e-5,
            n_a=100,
            b_max=0.7,
            b_min=0.65,
            n_b=6,
            gate_length=1.,
            constraints=[wrl.atten.constraint_dbz,
                        wrl.atten.constraint_pia],
            constraint_args=[[59.0],[20.0]])
            data0=data_nc+pia_mkraemer

            #print(np.size(data0))
            data=np.append(data,data0.ravel())
            #print(np.size(data))   
        #elevs=np.array(elevs)
        #print(elevs)
        #print(data)
        #print(np.size(data))
        
        #print(elevs0)
        #print(stoprange0)
        #list_stoprange=[]
        #list_azirange=[]
        #for i in range(nElevation):
        #    try:
        #        stoprange = float(raw['volume']['scan']['slice'][i]['stoprange'])
        #        list_stoprange.append(stoprange)
        #    except:
        #        stoprange = float(raw['volume']['scan']['slice'][0]['stoprange'])
        #        list_stoprange.append(stoprange)
        #    try:
        #        azirange = float(raw['volume']['scan']['slice'][i]['slicedata']['rayinfo']['@rays'])
        #        list_azirange.append(azirange)
        #    except:
        #        azirange = float(raw['volume']['scan']['slice'][i]['slicedata']['rayinfo'][0]['@rays'])
        #        list_azirange.append(azirange)
        #print(elevs)
        #print(azirange)
        #print(azidepth)
        #print(list_stoprange)
        #print(list_azirange)
        #if np.all(elevs==elevs0) and np.all(list_stoprange0==list_stoprange) and np.all(list_azirange0==list_azirange):
        try:
            vol=np.ma.masked_invalid(gridder0(data).reshape(trgshape0))
            print("sama")
            #gridder=gridder0
            trgxyz=trgxyz0
            trgshape=trgshape0
        #else:
        except:
            try:
                #print(elevs)
                #print(stoprange)
                #elevs0=elevs
                #list_stoprange0=list_stoprange
                #list_azirange0=list_azirange
                elevs=[]
                xyz=np.array([]).reshape((-1, 3))
                for i in range(nElevation):
                    elevation= float(raw['volume']['scan']['slice'][i]['posangle'])
                    elevs.append(elevation)

                    # ekstrak azimuth data
                    try:
                        #print("try")
                        azi = raw['volume']['scan']['slice'][i]['slicedata']['rayinfo']['data']
                        azidepth = float(raw['volume']['scan']['slice'][i]['slicedata']['rayinfo']['@depth'])
                        azirange = float(raw['volume']['scan']['slice'][i]['slicedata']['rayinfo']['@rays'])  
                    except:
                        #print("except")
                        azi0 = raw['volume']['scan']['slice'][i]['slicedata']['rayinfo'][0]['data']
                        azi1 = raw['volume']['scan']['slice'][i]['slicedata']['rayinfo'][1]['data']
                        azi = (azi0/2) + (azi1/2)
                        del azi0, azi1
                        azidepth = float(raw['volume']['scan']['slice'][i]['slicedata']['rayinfo'][0]['@depth'])
                        azirange = float(raw['volume']['scan']['slice'][i]['slicedata']['rayinfo'][0]['@rays'])            
                    try:
                        #print("try")
                        azires = float(raw['volume']['scan']['slice'][i]['anglestep'])
                    except:
                        #print("except")
                        azires = float(raw['volume']['scan']['slice'][0]['anglestep'])
                    #print(azi)
                    #print(azidepth)
                    #print(azirange)
                    #print(azires)
                    azims=(azi*azirange/2**azidepth)*azires
                    #print(azims)

                    # esktrak range data
                    try:
                        stoprange = float(raw['volume']['scan']['slice'][i]['stoprange'])
                        rangestep = float(raw['volume']['scan']['slice'][i]['rangestep'])
                    except:
                        stoprange = float(raw['volume']['scan']['slice'][0]['stoprange'])
                        rangestep = float(raw['volume']['scan']['slice'][0]['rangestep'])
                    #print(stoprange)
                    #print(rangestep)
                    ranges=np.arange(0,stoprange,rangestep)*1000
                    #print(ranges)

                    #print(np.size(azims))
                    #print(np.size(ranges))

                    polxyz=wrl.vpr.volcoords_from_polar(sitecoords,elevation,azims,ranges,proj)
                    xyz=np.vstack((xyz,polxyz))
                #membuat grid koordinat cartesian 3D
                elevs=np.array(elevs)
                #print(elevs)
                minelev=elevs.min()
                maxelev=elevs.max()
                trgxyz,trgshape=wrl.vpr.make_3d_grid(sitecoords,proj,maxrange,maxalt,horiz_res,vert_res)
                #print(np.size(trgxyz))
                #print((trgshape))
                gridder=wrl.vpr.PseudoCAPPI(xyz,trgxyz,trgshape,maxrange,minelev,maxelev,ipclass=wrl.ipol.Idw)
                xyz0=xyz
                trgxyz0=trgxyz
                trgshape0=trgshape
                gridder0=gridder
                print(gridder)

                #Ubah data dari koordinat sferis ke cartesian
                vol=np.ma.masked_invalid(gridder(data).reshape(trgshape))
            except:
                #print(elevs)
                #print(stoprange)
                #elevs0=elevs
                #list_stoprange0=list_stoprange
                #list_azirange0=list_azirange
                elevs=[]
                xyz=np.array([]).reshape((-1, 3))
                for i in range(nElevation):
                    elevation= float(raw['volume']['scan']['slice'][i]['posangle'])
                    elevs.append(elevation)

                    # esktrak range data
                    try:
                        stoprange = float(raw['volume']['scan']['slice'][i]['stoprange'])
                        rangestep = float(raw['volume']['scan']['slice'][i]['rangestep'])
                    except:
                        stoprange =float(raw['volume']['scan']['slice'][0]['stoprange'])
                        rangestep = float(raw['volume']['scan']['slice'][0]['rangestep'])
                    if stoprange == 240.0 or stoprange == 500:
                        stoprange=stoprange-1.0
                    #print(stoprange)
                    #print(rangestep)
                    ranges=np.arange(0,stoprange,rangestep)*1000
                    #print(ranges)

                    # ekstrak azimuth data
                    try:
                        #print("try")
                        azi = raw['volume']['scan']['slice'][i]['slicedata']['rayinfo']['data']
                        azidepth = float(raw['volume']['scan']['slice'][i]['slicedata']['rayinfo']['@depth'])
                        azirange = float(raw['volume']['scan']['slice'][i]['slicedata']['rayinfo']['@rays'])  
                    except:
                        #print("except")
                        azi0 = raw['volume']['scan']['slice'][i]['slicedata']['rayinfo'][0]['data']
                        #print(azi0)
                        azi1 = raw['volume']['scan']['slice'][i]['slicedata']['rayinfo'][1]['data']
                        #print(azi1)
                        if stoprange == 240 or stoprange == 500:
                            azi0=np.delete(azi0,np.where(azi0>=np.max(azi0)))
                            azi1=np.delete(azi1,np.where(azi1>=np.max(azi1)))
                        azi = (azi0/2) + (azi1/2)
                        #if stoprange == 240:
                        #    azi=np.delete(azi,np.where(azi>=np.max(azi)))
                        del azi0, azi1
                        azidepth = float(raw['volume']['scan']['slice'][i]['slicedata']['rayinfo'][0]['@depth'])
                        azirange = float(raw['volume']['scan']['slice'][i]['slicedata']['rayinfo'][0]['@rays'])            
                    try:
                        #print("try")
                        azires = float(raw['volume']['scan']['slice'][i]['anglestep'])
                    except:
                        #print("except")
                        azires = float(raw['volume']['scan']['slice'][0]['anglestep'])
                    #print(azi)
                    #print(azidepth)
                    #print(azirange)
                    #print(azires)
                    azims=(azi*azirange/2**azidepth)*azires
                    #if stoprange == 240:
                    #    azims.append(np.max(azims)+0.5)
                    #print(np.size(azims))

                    #print(np.size(azims))
                    #print(np.size(ranges))

                    polxyz=wrl.vpr.volcoords_from_polar(sitecoords,elevation,azims,ranges,proj)
                    xyz=np.vstack((xyz,polxyz))
                #membuat grid koordinat cartesian 3D
                elevs=np.array(elevs)
                #print(elevs)
                minelev=elevs.min()
                maxelev=elevs.max()
                trgxyz,trgshape=wrl.vpr.make_3d_grid(sitecoords,proj,maxrange,maxalt,horiz_res,vert_res)
                #print(np.size(trgxyz))
                #print((trgshape))
                gridder=wrl.vpr.PseudoCAPPI(xyz,trgxyz,trgshape,maxrange,minelev,maxelev,ipclass=wrl.ipol.Idw)
                xyz0=xyz
                trgxyz0=trgxyz
                trgshape0=trgshape
                gridder0=gridder
                print(gridder)

                #Ubah data dari koordinat sferis ke cartesian
                vol=np.ma.masked_invalid(gridder(data).reshape(trgshape))

        #print(vol)
        #print(np.max(vol))
        trgx = trgxyz[:, 0].reshape(trgshape)[0, 0, :]
        trgy = trgxyz[:, 1].reshape(trgshape)[0, :, 0]
        trgz = trgxyz[:, 2].reshape(trgshape)[:, 0, 0]
        trgxl,trgyl=p(trgx,trgy,inverse=True)
        #print((trgxl))
        #print((trgyl))
        lonMin=trgxl.min()
        lonMax=trgxl.max()
        latMin=trgyl.min()
        latMax=trgyl.max()
        lonMesh, latMesh=np.meshgrid(trgxl,trgyl)
        #print(len(lonMesh))
        #print(len(latMesh))

        vol_mask=vol[:, :, 0]
        #print(vol_mask)

        #median filter
        mask=ndimage.median_filter(vol_mask,footprint=np.ones((5,5)),mode='reflect')
        #print(vol[0, 0, :])
        #print(vol[0, :, 0])
        #print(vol[:, 0, 0])
        #vol=[vol[0, 0, :],vol[0, :, 0],vol[:, 0, 0]]
        #print(mask)
        #print(np.max(mask))
        mask=np.where(mask<0,1,0)
        #print(mask)
        #print(np.max(mask))
        #print(np.min(mask))
        vol_mask=np.ma.masked_array(vol_mask,mask)
        vol[:, :, 0]=vol_mask
        vol=np.around(vol,decimals=2)
        #print(vol)
        #print(np.max(vol))
        #print(np.min(vol))

        #cmax
        cmaxData=np.nanmax(vol[:,:,:],axis=0)
        cmaxData[cmaxData<-100]=np.nan;cmaxData[cmaxData>100]=np.nan
        cmaxData=np.asarray(cmaxData)

        nama="/Users/cin/Documents/pyfile/file_array/"+timeEnd.strftime("%Y%m%d-%H%M")+'UTC.npz'
        with open(nama,"wb") as arr:
            filetxt=np.savez_compressed(arr,cmaxData)
        #content=str(cmaxData)
        #filetxt.write(content)
        #filetxt.close()

        #temukan reflektivitas
        cmax_pos=[timeEnd.strftime("%Y%m%d-%H%M")]
        for i in range(len(Buj_Pos)):
            bjr=find_nearest(trgxl,Buj_Pos[i])
            ltg=find_nearest(trgyl,Lin_Pos[i])
            refl=cmaxData[bjr,ltg]
            cmax_pos.append(refl)
        #print(len(cmax_pos))
        #print(cmax_pos)
        
        #with open ("/Users/cin/Documents/pyfile/Pos_Refl0.csv","a") as fcsv:
        #    write=csv.writer(fcsv)
        #    write.writerow(cmax_pos)
        #    fcsv.close()

        #print(cmaxData)

        #print(np.size(cmaxData))
        #m=Basemap(llcrnrlat=latMin,urcrnrlat=latMax,\
        #            llcrnrlon=lonMin,urcrnrlon=lonMax,\
        #            resolution='i')
        #x0,y0=m(radarLon,radarLat)
        #x1,y1=m(lonMesh,latMesh)
        #print(x1)
        #print(y1)
        #clevsZ = [5,10,15,20,25,30,35,40,45,50,55,60,65,70]
        #colors=['#07FEF6','#0096FF','#0002FE','#01FE03','#00C703','#009902','#FFFE00','#FFC801','#FF7707','#FB0103','#C90002','#980001','#FF00FF','#9800FE']
        #pl.figure(figsize=(10,10))
        #m.plot(x0, y0, 'ko', markersize=3)
        #m.contourf(x1, y1, cmaxData,clevsZ,colors=colors)
        #m.colorbar(ticks=clevsZ,location='right',size='4%',label='Reflectivity (dBZ)')
        #m.drawparallels(np.arange(math.floor(latMin),math.ceil(latMax),1.5),labels=[1,0,0,0],linewidth=0.2)
        #m.drawmeridians(np.arange(math.floor(lonMin),math.ceil(lonMax),1.5),labels=[0,0,0,1],linewidth=0.2)
        #m.drawcoastlines() 
        #title='CMAX CAPPI '+timeEnd.strftime("%Y%m%d-%H%M")+' UTC'
        #pl.title(title,weight='bold',fontsize=15)
        #pl.savefig('/Users/cin/Documents/pyfile/CMAX.png',bbox_inches='tight',dpi=200,pad_inches=0.1)
        #pl.close()

#np.savetxt('/Users/cin/Documents/pyfile/bujur.csv',trgxl,delimiter=';')
#np.savetxt('/Users/cin/Documents/pyfile/lintang.csv',trgyl,delimiter=';')