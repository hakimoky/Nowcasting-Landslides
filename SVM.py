import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pickle
import haversine as hs

def hitung_jarak(bj,lt):
    jarak=hs.haversine((lt,bj),(-7.4107,112.7600))
    return jarak

#buat meshgrid
def make_meshgrid(x, y, h=.002):
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

#plot kontur y=1 dan y=-1
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

dataset=pd.read_csv('dataset_svm.csv',delimiter=",")
dataset.dropna(axis=0,how='any',thresh=None,subset=None,inplace=True)
dataset['hujan']=dataset['hujan'].replace(['ttu'],[0.01])
dataset['hujan']=pd.to_numeric(dataset['hujan'],downcast='float')
reflek=np.array(dataset['reflektivitas'])
ch=np.array(dataset['hujan'])
ch_filter=[]

for i in range(len(ch)):
    if ch[i] == 0.0 and reflek[i] >= 20.0:
        ch_filter.append('')
    elif ch[i] != 0.0 and reflek[i] < -20.0:
        ch_filter.append('')
    else:
        ch_filter.append(ch[i])
dataset['hujan']=ch_filter
dataset['hujan']=pd.to_numeric(dataset['hujan'],downcast='float')
dataset.dropna(axis=0,how='any',thresh=None,subset=None,inplace=True)
dataset['event']=np.where(dataset['hujan']>0,1,-1)
print(dataset)

indeks=(dataset['tanggal'].tolist()).index(20200101)
print(indeks)

input2=np.array(dataset['reflektivitas'])
input1=np.array(dataset['jarak'])
output=np.array(dataset['event'])

input2_norm=(input2+31.5)/(95.5+31.5)
input1_norm=(input1)/200
input_norm=[]
for i in range(len(input1_norm)):
    input_norm.append([input1_norm[i],input2_norm[i]])
output_norm=output.tolist()
input_train,input_test,output_train,output_test=input_norm[:indeks],input_norm[indeks:],output_norm[:indeks],output_norm[indeks:]

clf=svm.SVC(C=1.0,kernel='poly',degree=2)
models=clf.fit(input_train,output_train)

svm_model='svm_model.sav'
pickle.dump(clf,open(svm_model,'wb'))
out_mod=clf.predict(input_train)
out_pred=clf.predict(input_test)
pc_b=accuracy_score(output_train,out_mod)
pc_f=accuracy_score(output_test,out_pred)
fss=(pc_f-pc_b)/(1.0-pc_b)
akurasi=open('verifikasi_svm.txt','w')
akurasi.write('PCf = '+str(pc_f)+'\n'+'PCb = '+str(pc_b)+'\n'+'FSS = '+str(fss))
akurasi.close()
cm=confusion_matrix(output_test,out_pred,labels=[1,-1])
np.savetxt('confusion_matrix.csv',cm,delimiter=',')
print(cm)

cr=classification_report(output_test,out_pred)
#np.savetxt('classification_report.csv',cr,delimiter=',')
print(cr)

#get support vectors
sup_vect=clf.support_vectors_
param=clf.get_params(deep=True)
np.savetxt('support_vector.csv',sup_vect,delimiter=',')
print(param)

#meshgrid
xx,yy=make_meshgrid(input1_norm,input2_norm)
X1=[]
X2=[]
for Xa in input_train:
    X1.append(Xa[0])
    X2.append(Xa[1])

#Buat figure
fig1 = plt.figure(figsize=(7, 5))
ax=fig1.add_axes([0.1,0.1,0.85,0.85])
plot_contours(ax, models, xx, yy,cmap=plt.cm.coolwarm, alpha=1)
ax.scatter(X1, X2, c=output_train, cmap=plt.cm.coolwarm, s=5, edgecolors='k', linewidth=0.5)

#atur axis
ax.set_xlim(np.min(xx), np.max(xx))
ax.set_ylim(np.min(yy), np.max(yy))
ax.set(xlabel="Jarak ternormalisasi",ylabel="Reflektivitas ternormalisasi")

#buat legenda
red_patch=mpatches.Patch(color='red', label='y = Hujan')
blue_patch=mpatches.Patch(color='blue', label='y = Tidak Hujan')
black_line=mlines.Line2D([], [], color='black',markersize=5, label='Hyperplane')
dashed_line=mlines.Line2D([], [],linestyle='--', color='black',markersize=5, label='H1 atau H2')
ax.legend(handles=[red_patch,blue_patch,black_line,dashed_line])
ax.set_title('Batas Reflektivitas Ukuran Tetes Hujan')
xy = np.vstack([xx.ravel(), yy.ravel()]).T
z= clf.decision_function(xy).reshape(xx.shape)

# plot hyperplane dan marginnya
ax.contour(xx, yy, z, colors='k', levels=[-1, 0, 1], alpha=1, linestyles=['--', '-', '--'])

# tandai support vectors
#ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=10, linewidth=1, facecolors='none', edgecolors='k')
#ax.draw()
#plt.show()

#buat mask
lintang=pd.read_csv('lintang.csv',header=None)
bujur=pd.read_csv('bujur.csv',header=None)
buj=np.array(bujur)
lin=np.array(lintang)

jarak=[]
for l in lin:
    for b in buj:
        jar=hitung_jarak(b,l)
        jarak.append(jar)
jarak_norm=np.array(jarak)/200
#print(np.shape(jarak_norm))

refl=np.arange(-31.5,95.6,0.1)
reflek=(refl+31.5)/(95.5+31.5)
print(reflek)
refl_rain=[]
for j in jarak_norm:
    for r in reflek:
        uji=clf.predict([[j,r]])
        if uji == 1:
            refl_rain.append(r)
            print(r)
            break
arr_refl_rain=np.array(refl_rain)
print(np.shape(arr_refl_rain))
batas_refl=np.reshape(arr_refl_rain,(1001,1001))
print(batas_refl)
np.savetxt('/Users/cin/Documents/pyfile/bujur.csv',refl_rain,delimiter=',')