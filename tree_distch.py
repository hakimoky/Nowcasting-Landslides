import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import pickle
from sklearn import tree
from IPython.display import SVG
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

dataset=pd.read_csv('dataset_regres.csv',delimiter=",")
#print(len(np.array(dataset['hujan'])))

dataset['hujan obs']=dataset['hujan obs'].replace([''],[np.NaN])
dataset['hujan obs']=dataset['hujan obs'].replace(['ttu'],[0.0])
dataset['hujan obs']=pd.to_numeric(dataset['hujan obs'],downcast='float')
dataset['hujan obs']=dataset['hujan obs'].replace([0],[np.NaN])
dataset['hujan obs']=dataset['hujan obs'].replace([0.0],[np.NaN])
dataset['hujan radar']=dataset['hujan radar'].replace([''],[np.NaN])
dataset['hujan radar']=dataset['hujan radar'].replace([0.0],[np.NaN])

dataset.dropna(axis=0,how='any',thresh=None,subset=None,inplace=True)

#dataset['bias']=dataset['hujan obs']/dataset['hujan radar']
#print(sum(dataset['bias']<20.0))
#print(sum(dataset['bias']>0.05))

obs=np.array(dataset['hujan obs'])
radar=np.array(dataset['hujan obs'])
lead=np.array(dataset['total lead'])
obs_filter=[]
for i in range(len(lead)):  
    if lead[i] >= 85800.0:
        obs_filter.append(obs[i])
    elif lead[i] < 85800.0:
        if abs(obs[i]-radar[i]) < 20:
            obs_filter.append(obs[i])
        else:
            obs_filter.append(np.NaN)
dataset['hujan obs']=obs_filter
dataset['hujan obs']=pd.to_numeric(dataset['hujan obs'],downcast='float')
dataset.dropna(axis=0,how='any',thresh=None,subset=None,inplace=True)

dataset['bias']=dataset['hujan obs']/dataset['hujan radar']
print(sum(dataset['bias']<0.1))
print(sum(dataset['bias']>10.0))
Q1b=np.quantile(dataset['bias'],0.25)
Q3b=np.quantile(dataset['bias'],0.75)
IQRb=Q3b-Q1b
lr=Q1b-1.5*IQRb
ur=Q3b+1.5*IQRb
bias=np.array(dataset['bias'])
bias_filter=[]
for i in bias:
    if i <= ur and i >= lr:
        bias_filter.append(i)
    else:
        bias_filter.append(np.NaN)
dataset['bias']=bias_filter
dataset.dropna(axis=0,how='any',thresh=None,subset=None,inplace=True)

print(dataset)
print(dataset.describe())

input2=np.array(dataset['hujan radar'])
input1=np.array(dataset['jarak'])

input2_norm=input2
input1_norm=(input1)/200
#input1_norm=input1

input_norm=[]
for i in range(len(input1_norm)):
    input_norm.append([input1_norm[i],input2_norm[i]])


indeks=(dataset['tanggal'].tolist()).index(20200101)
print(indeks)

output=np.array(dataset['hujan obs'])

output_norm=output.tolist()
#print(X)
#print(y)

#input_norm=[]
#for i in range(len(inp_2)):
#    input_norm.append([inp_2[i],inp_1[i]])
#output_norm=output.tolist()
#inp_2_norm=(input2+31.5)/(95.5+31.5)
#inp_1_norm=(input1)/200

input_train,input_test,output_train,output_test=input_norm[:indeks],input_norm[indeks:],output_norm[:indeks],output_norm[indeks:]

tc_depth=[]
for d in range(1,21,1):
    tc0= DecisionTreeRegressor(criterion='mse',splitter='best',max_depth=d,min_weight_fraction_leaf=0.001)
    model_tc0 = tc0.fit(input_train, output_train)
    tc_pred0 = model_tc0.predict(input_test)
    mse_tc=metrics.mean_squared_error(output_test, tc_pred0)
    tc_depth.append(mse_tc)
best_tc=tc_depth.index(min(tc_depth))+1
print('depth = ',best_tc)
plt.plot(range(1,21,1),tc_depth)
plt.xlabel('Depth')
plt.ylabel('Mean Squared Error')
plt.xticks(range(1,21,1))
plt.show()

rt = DecisionTreeRegressor(criterion='mse',splitter='best',max_depth=best_tc,min_weight_fraction_leaf=0.001,max_features=None)
model_r = rt.fit(input_train, output_train)
y_pred = model_r.predict(input_test)
text_representation = tree.export_text(model_r)
print(text_representation)
rmse=np.sqrt(metrics.mean_squared_error(output_test, y_pred))
print('Root Mean Squared Error:', rmse)
tc_mod=model_r.predict(input_train)
rmse_mod=np.sqrt(metrics.mean_squared_error(output_train, tc_mod))
skill_tc=1-(rmse/rmse_mod)
bias=np.sum((np.array(y_pred)-np.array(output_test)))/np.size(np.array(y_pred))
print(rmse,rmse_mod,skill_tc,bias)
np.savetxt('stats_treereg.csv',[rmse,rmse_mod,skill_tc,bias],delimiter=',')

tree_model='tree_model.sav'
pickle.dump(rt,open(tree_model,'wb'))


X_grid=np.linspace(0.1,1.0,200)
Y_grid=np.linspace(1,200,200)
x_jrk,y_rad=np.meshgrid(X_grid,Y_grid)
#print(x_jrk)
#print(y_rad)
input_grid=[]
for j in range(len(Y_grid)):
    for i in range(len(X_grid)):
        input_grid.append([X_grid[i],Y_grid[j]])
#print(input_grid)
z_ch=model_r.predict(input_grid)
z_ch=np.reshape(z_ch,(200,200))
#print(z_ch)
fig1=plt.figure()
ax1=plt.axes(projection='3d')
ax1.plot_surface(x_jrk,y_rad,z_ch,cmap='viridis')
ax1.set_xlabel("Jarak")
ax1.set_ylabel("CH Radar")
ax1.set_zlabel("CH")
ax1.view_init(60,35)
plt.figure(ax1)

#fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=300)
#fig= tree.plot_tree(model_r,feature_names=["Jarak","Curah Hujan Radar"],filled=True)
fig1.savefig("decission_tree.png")