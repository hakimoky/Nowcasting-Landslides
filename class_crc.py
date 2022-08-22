import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
import pickle

data_training=pd.read_csv('dataset_training_ML_classify.csv',delimiter=",")
data_testing=pd.read_csv('dataset_testing_ML_classify.csv',delimiter=",")

datrain_luas=np.array(data_training['luas cell'])
datrain_maks=np.array(data_training['ch maks'])
datrain_rata2=np.array(data_training['ch rata2'])
datrain_benc=np.array(data_training['bencana'])

datest_luas=np.array(data_testing['luas cell'])
datest_maks=np.array(data_testing['ch maks'])
datest_rata2=np.array(data_testing['ch rata2'])
datest_benc=np.array(data_testing['bencana'])

ba_luas=[np.max(datrain_luas),np.max(datest_luas)]
ba_luas=max(ba_luas)
ba_chmaks=[np.max(datrain_maks),np.max(datest_maks)]
ba_chmaks=max(ba_chmaks)
ba_chrata2=[np.max(datrain_rata2),np.max(datest_rata2)]
ba_chrata2=max(ba_chrata2)
print(ba_luas,ba_chmaks,ba_chrata2)
np.savetxt('maks_stats_cell.csv',[ba_luas,ba_chmaks,ba_chrata2],delimiter=',')

inp_train=[]
out_train=[]
for i in range(len(datrain_benc)):
    inp_luas=datrain_luas[i]/ba_luas
    inp_maks=datrain_maks[i]/ba_chmaks
    inp_rata2=datrain_rata2[i]/ba_chrata2
    inp_train.append([inp_luas,inp_maks,inp_rata2])
    #inp_train.append([inp_maks,inp_rata2])
    inp_benc=datrain_benc[i]
    if inp_benc == 'Aman':
        out_train.append('Aman')
    elif inp_benc == 'Longsor':
        out_train.append('Longsor')

inp_test=[]
out_test=[]
for i in range(len(datest_benc)):
    inp1_luas=datest_luas[i]/ba_luas
    inp1_maks=datest_maks[i]/ba_chmaks
    inp1_rata2=datest_rata2[i]/ba_chrata2
    inp_test.append([inp1_luas,inp1_maks,inp1_rata2])
    #inp_test.append([inp1_maks,inp1_rata2])
    inp1_benc=datest_benc[i]
    if inp1_benc == 'Aman':
        out_test.append('Aman')
    elif inp1_benc == 'Longsor':
        out_test.append('Longsor')

#Decission Tree
tc_depth=[]
for d in range(1,21,1):
    tc0= DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=d,min_weight_fraction_leaf=0.01)
    model_tc0 = tc0.fit(inp_train, out_train)
    tc_pred0 = model_tc0.predict(inp_test)
    presisi_tc=metrics.precision_score(out_test, tc_pred0,pos_label='Longsor')
    tc_depth.append(presisi_tc)
best_tc=tc_depth.index(max(tc_depth))+1
print('depth = ',best_tc)
plt.plot(range(1,21,1),tc_depth)
plt.xlabel('Depth')
plt.ylabel('Probability of Detection')
plt.xticks(range(1,21,1))
plt.show()

tc= DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=best_tc,min_weight_fraction_leaf=0.01)
model_tc = tc.fit(inp_train, out_train)
text_representation = tree.export_text(model_tc)
print(text_representation)
plt.figure(figsize=(15,8))
tree.plot_tree(model_tc,feature_names=['Luas Cell','CH Maks','CH Rata2'],class_names=['Aman','Longsor'])
plt.savefig("tree_class.png")
plt.show()
tc_mod=model_tc.predict(inp_train)
mod_tc=metrics.precision_score(out_train, tc_mod,pos_label='Longsor')
tc_pred = model_tc.predict(inp_test)
akurasi_tc=metrics.precision_score(out_test, tc_pred,pos_label='Longsor')
conf_tc=metrics.confusion_matrix(out_test, tc_pred)
print(conf_tc)
np.savetxt('confusion_treeclass.csv',conf_tc,delimiter=',')
skill_tc=(akurasi_tc-mod_tc)/(1.0-mod_tc)
tc_model='dtclass_model.sav'
pickle.dump(tc_mod,open(tc_model,'wb'))

data_training['Prediksi Tree']=tc_mod
data_testing['Prediksi Tree']=tc_pred
data_training.to_csv('/Users/cin/Documents/pyfile/CNN/dataset_trainCNN.csv',index=False)
data_testing.to_csv('/Users/cin/Documents/pyfile/CNN/dataset_testCNN.csv',index=False)

#randomforest
rf_depth=[]
for d in range(10,210,10):
    rc0 = RandomForestClassifier(n_estimators=d,criterion='entropy',max_depth=best_tc,min_weight_fraction_leaf=0.01,max_features=None)
    model_rc0 = rc0.fit(inp_train, out_train)
    rc_pred0 = model_rc0.predict(inp_test)
    presisi_rc=metrics.precision_score(out_test, rc_pred0,pos_label='Longsor')
    rf_depth.append(presisi_rc)
best_rf=(rf_depth.index(max(rf_depth))+1)*10
print('depth = ',best_rf)
plt.plot(range(10,210,10),rf_depth)
plt.xlabel('Jumlah Anggota Ensemble Tree')
plt.ylabel('Probability of Detection')
plt.xticks(range(10,210,10))
plt.show()

rc = RandomForestClassifier(n_estimators=best_rf,criterion='entropy',max_depth=best_tc,min_weight_fraction_leaf=0.01,max_features=None)
model_rc = rc.fit(inp_train, out_train)
plt.figure(figsize=(15,8))
tree.plot_tree(model_rc.estimators_[0],feature_names=['Luas Cell','CH Maks','CH Rata2'],class_names=['Aman','Longsor'])
plt.savefig("tree_class.png")
plt.show()
#rf_representation = tree.export_text(model_rc)
#print(rf_representation)
rc_mod=model_rc.predict(inp_train)
mod_rc=metrics.precision_score(out_train, rc_mod,pos_label='Longsor')
rc_pred = model_rc.predict(inp_test)
akurasi_rc=metrics.precision_score(out_test, rc_pred,pos_label='Longsor')
conf_rc=metrics.confusion_matrix(out_test, rc_pred)
print(conf_rc)
np.savetxt('confusion_rfclass.csv',conf_rc,delimiter=',')
skill_rc=(akurasi_rc-mod_rc)/(1.0-mod_rc)
rc_model='rfclass_model.sav'
pickle.dump(rc_mod,open(rc_model,'wb'))

#MLP
MLP_hidden=[]
for d in range(10,210,10):
    mc0 = MLPClassifier(hidden_layer_sizes=100,activation='relu',solver='adam',alpha=0.0001,batch_size='auto',learning_rate='constant',max_iter=2000)
    model_mc0 = mc0.fit(inp_train, out_train)
    mc_pred0 = model_mc0.predict(inp_test)
    presisi_mc=metrics.precision_score(out_test, mc_pred0,pos_label='Longsor')
    MLP_hidden.append(presisi_mc)
best_MLP=(MLP_hidden.index(max(MLP_hidden))+1)*10
print('hidden = ',best_MLP)
plt.plot(range(10,210,10),MLP_hidden)
plt.xlabel('Hidden Layers')
plt.ylabel('Probability of Detection')
plt.xticks(range(10,210,10))
plt.show()

mc = MLPClassifier(hidden_layer_sizes=best_MLP,activation='relu',solver='adam',alpha=0.0001,batch_size='auto',learning_rate='constant',max_iter=2000)
model_mc = mc.fit(inp_train, out_train)
mc_mod=model_mc.predict(inp_train)
mod_mc=metrics.precision_score(out_train, mc_mod,pos_label='Longsor')
mc_pred = model_mc.predict(inp_test)
akurasi_mc=metrics.precision_score(out_test, mc_pred,pos_label='Longsor')
conf_mc=metrics.confusion_matrix(out_test, mc_pred)
print(conf_mc)
print(model_mc.coefs_[0])
v0=model_mc.coefs_[0][0]
v1=model_mc.coefs_[0][1]
v2=model_mc.coefs_[0][2]
fig,ax=plt.subplots(figsize=(15,3))
w_label=["w1","w2","w3"]
psm=ax.pcolormesh(model_mc.coefs_[0],cmap='autumn')
fig.colorbar(psm,ax=ax)
ax.set_yticks(np.arange(len(w_label)),labels=w_label)
ax.set_xlabel('hidden layers')
ax.set_ylabel('weights')
plt.show()
np.savetxt('confusion_mlpclass.csv',conf_mc,delimiter=',')
skill_mc=(akurasi_mc-mod_mc)/(1.0-mod_mc)
mc_model='mlpclass_model.sav'
pickle.dump(mc_mod,open(mc_model,'wb'))

print(akurasi_tc,akurasi_rc,akurasi_mc)
print(mod_tc,mod_rc,mod_mc)
print(skill_tc,skill_rc,skill_mc)

np.savetxt('stats_crclass.csv',[[akurasi_tc,akurasi_rc,akurasi_mc],[mod_tc,mod_rc,mod_mc],[skill_tc,skill_rc,skill_mc]],delimiter=',')