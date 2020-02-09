import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , LabelEncoder

# creation OF CSV FOLDERS :
try:
    if not os.path.exists('1st_test_csv'):
        os.makedirs('1st_test_csv')
        os.makedirs('2st_test_csv')
        os.makedirs('3st_test_csv')
        os.makedirs('all_data_csv')
except:
    print('folders are already existe')
    
    #change the path or use cwd 
folder1 = 'C:\\Users\\yassir\\PycharmProjects\\untitled1\\1st_test\\'
folder2 = 'C:\\Users\\yassir\\PycharmProjects\\untitled1\\2nd_test\\'
folder3 = 'C:\\Users\\yassir\\PycharmProjects\\untitled1\\4th_test\\'

path_1st_test_csv = os.path.abspath("1st_test_csv")
path_2st_test_csv = os.path.abspath("2st_test_csv")
path_3st_test_csv = os.path.abspath("3st_test_csv")
path_all_data_csv = os.path.abspath("all_data_csv")



def life_time_exp():
    rpm = 2000
    rev = 100000000
    time = rev / rpm
    time_in_h = time / 60
    time_in_days = time_in_h / 24
    return (time_in_days)


df = pd.read_csv(folder1 + '2003.10.22.12.06.24',
           sep='\t',
           header=None,
           names=['Bearing 1 Ch1', 'Bearing 1 Ch2',
                  'Bearing 2 Ch1', 'Bearing 2 Ch2',
                  'Bearing 3 Ch1', 'Bearing 3 Ch2',
                  'Bearing 4 Ch1', 'Bearing 4 Ch2'])
def read_vibration_file(filename):
    df = pd.read_csv(folder1 + filename,
           sep='\t',
           header=None,
           names=['Bearing 1 Ch1', 'Bearing 1 Ch2',
                  'Bearing 2 Ch1', 'Bearing 2 Ch2',
                  'Bearing 3 Ch1', 'Bearing 3 Ch2',
                  'Bearing 4 Ch1', 'Bearing 4 Ch2'])
    return df

df[['Bearing 1 Ch1', 'Bearing 1 Ch2']].plot(marker='o', linewidth=0)

df[['Bearing 1 Ch1', 'Bearing 1 Ch2']].plot(marker='o', linewidth=0)
plt.show()
ch1_mean = df['Bearing 1 Ch1'].mean()
ch2_mean = df['Bearing 1 Ch2'].mean()

ch1_rms = np.sqrt(sum(df['Bearing 1 Ch1'].apply(lambda x: (x - ch1_mean)**2))/len(df))
ch2_rms = np.sqrt(sum(df['Bearing 1 Ch2'].apply(lambda x: (x - ch2_mean)**2))/len(df))

plt.plot([0, len(df)], [ch1_mean, ch1_mean])
plt.plot([0, len(df)], [ch1_mean+ch1_rms, ch1_mean+ch1_rms])
plt.plot([0, len(df)], [ch1_mean-ch1_rms, ch1_mean-ch1_rms])

plt.show()

def calculate_rms(series):
    series_mean = series.mean()

    series_rms = np.sqrt(sum(series.apply(lambda x: (x - series_mean) ** 2)) / len(series))

    return series_rms

print('Number of sample files : ' + str(len(os.listdir(folder1))))



rms_dict = {}
for filename in os.listdir(folder1):
    df = read_vibration_file(filename)
    rms_dict[filename] = [calculate_rms(df[column]) for column in df.columns]


rms = pd.DataFrame.from_dict(rms_dict,
                       orient='index',
                       columns=df.columns)

rms.to_csv(path_all_data_csv + '\\' + 'rms' + '.csv')
rms = pd.read_csv(path_all_data_csv + '\\' + 'rms' + '.csv')

rms = rms.rename(columns={'Unnamed: 0' : 'Sample Datetime'})

rms['Sample Datetime'] = pd.to_datetime(rms['Sample Datetime'], format='%Y.%m.%d.%H.%M.%S')

rms = rms.set_index('Sample Datetime')


rms[['Bearing 1 Ch1', 'Bearing 1 Ch2', 'Bearing 2 Ch1', 'Bearing 2 Ch2']].plot()

rms[['Bearing 3 Ch1', 'Bearing 3 Ch2', 'Bearing 4 Ch1', 'Bearing 4 Ch2']].plot()

plt.show()
plt.plot(rms['Bearing 3 Ch1'], rms['Bearing 3 Ch2'],'o')

plt.plot(rms.iloc[157:-700]['Bearing 3 Ch1'], rms.iloc[157:-700]['Bearing 3 Ch2'],'o')

train = rms[0:-700][['Bearing 3 Ch1', 'Bearing 3 Ch2']].copy()

pca_model = PCA(n_components=2)
pca_model.fit(train[['Bearing 3 Ch1', 'Bearing 3 Ch2']].values)

train['Bearing 3 PC1'], train['Bearing 3 PC2'] = zip(
    *pca_model.transform(train[['Bearing 3 Ch1', 'Bearing 3 Ch2']].values))

# Find guassian distribution parameters
model = train.agg({'Bearing 3 Ch1': ['mean', 'std'],
                   'Bearing 3 Ch2': ['mean', 'std'],
                   'Bearing 3 PC1': ['mean', 'std'],
                   'Bearing 3 PC2': ['mean', 'std']})

for ch in train.columns:
    train[ch + ' z'] = (train[ch] - model[ch]['mean']) / model[ch]['std']

    train[ch + ' p'] = stats.norm.sf(abs(train[ch + ' z'])) * 2

train['Ch_p_value'] = train['Bearing 3 Ch1 p'] * train['Bearing 3 Ch2 p']

train['PC_p_value'] = train['Bearing 3 PC1 p'] * train['Bearing 3 PC2 p']


plt.scatter(train['Bearing 3 Ch1'], train['Bearing 3 Ch2'], [], train['Ch_p_value'])
plt.colorbar()
plt.title('Multivatiate Gaussian (Assuming variation is axis aligned)')
plt.xlabel('Bearing 3 Ch1 RMS')
plt.ylabel('Bearing 3 Ch2 RMS')
plt.show()


plt.scatter(train['Bearing 3 Ch1'], train['Bearing 3 Ch2'], [], train['PC_p_value'])
plt.colorbar()
plt.xlabel('Bearing 3 Ch1 RMS')
plt.ylabel('Bearing 3 Ch2 RMS')
plt.title('Multivatiate Gaussian (Transformed)')
plt.show()


plt.scatter(train['Bearing 3 PC1'], train['Bearing 3 PC2'], [], train['PC_p_value'])
plt.colorbar()
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Multivatiate Gaussian (Transformed) in Principal Components')
plt.show()

rms['Bearing 3 PC1'], rms['Bearing 3 PC2'] = zip(*pca_model.transform(rms[['Bearing 3 Ch1', 'Bearing 3 Ch2']].values))

for ch in ['Bearing 3 PC1', 'Bearing 3 PC2']:
    rms[ch + ' z'] = (rms[ch] - model[ch]['mean']) / model[ch]['std']

    rms[ch + ' p'] = stats.norm.sf(abs(rms[ch + ' z'])) * 2

rms['PC_p_value'] = rms['Bearing 3 PC1 p'] * rms['Bearing 3 PC2 p']


plt.plot(rms['PC_p_value'])
plt.title('Bearing 3 P Values')


plt.plot(np.log(rms['PC_p_value']))
plt.ylabel('Log of P values')
plt.title('Log of Bearing 3 P Values')

plt.show()


plt.plot(np.log(rms['PC_p_value']))
plt.plot([min(rms.index), max(rms.index)], [np.log(1e-4), np.log(1e-4)])
plt.ylabel('Log of P values')
plt.title('Log of Bearing 3 P Values')
plt.ylim([-20, 0])
plt.legend(['Log(p values)', 'proposed threshold'])
plt.show()


anomaly = rms['PC_p_value']<1e-4

plt.plot(rms[~anomaly]['Bearing 3 Ch1'], rms[~anomaly]['Bearing 3 Ch2'], 'o')
plt.plot(rms[anomaly]['Bearing 3 Ch1'], rms[anomaly]['Bearing 3 Ch2'], 'ro')

anomaly = rms['PC_p_value']<1e-4

plt.plot(rms[~anomaly]['Bearing 3 Ch1'], 'o')
plt.plot(rms[anomaly]['Bearing 3 Ch1'], 'ro')
plt.ylabel('Bearing 3 Ch1 RMS Vibration')
plt.legend(['Healthy', 'Unhealthy'])

plt.show()

anomaly = rms['PC_p_value']<1e-4

plt.plot(rms[~anomaly]['Bearing 3 Ch1'], 'o')
plt.plot(rms[anomaly]['Bearing 3 Ch1'], 'ro')
plt.ylabel('Bearing 3 Ch1 RMS Vibration')
plt.legend(['Healthy', 'Unhealthy'])
plt.ylim([0, 0.2])
plt.title('Bearing 3 Health Status')


plt.show()



first_anomaly_time = min(rms[~anomaly].index)
final_run_time = max(rms.index)

predictive_time = final_run_time - first_anomaly_time

def read_vibration_file(filename, folder):
    if folder == folder1:
        df1 = pd.read_csv(folder + filename,
                          sep='\t',
                          header=None,
                          names=['Bearing 1 Ch1', 'Bearing 1 Ch2',
                                 'Bearing 2 Ch1', 'Bearing 2 Ch2',
                                 'Bearing 3 Ch1', 'Bearing 3 Ch2',
                                 'Bearing 4 Ch1', 'Bearing 4 Ch2'])
        df1 = df1.loc[:, ['Bearing 1 Ch1', 'Bearing 2 Ch1', 'Bearing 3 Ch1', 'Bearing 4 Ch1']]
        return df1

    else:
        df1 = pd.read_csv(folder + filename,
                          sep='\t',
                          header=None,
                          names=['Bearing 1 Ch1',
                                 'Bearing 2 Ch1',
                                 'Bearing 3 Ch1',
                                 'Bearing 4 Ch1', ])
        df2 = df1.loc[:, ['Bearing 1 Ch1', 'Bearing 2 Ch1', 'Bearing 3 Ch1', 'Bearing 4 Ch1']]
        return df2




allcsv = []
def fusion_data_file(folder1):
    for filename in os.listdir(folder1):
        df1 = read_vibration_file(filename, folder1)
        df1 = df1.loc[:, ['Bearing 1 Ch1', 'Bearing 2 Ch1', 'Bearing 3 Ch1', 'Bearing 4 Ch1']]
        #allcsv.append(df1)
life_time = life_time_exp()
def add_featurs (allcsv):
    fusion_data_file(folder1)
    all_data = pd.concat(allcsv, ignore_index=True)
    all_data_to_pd = pd.DataFrame(all_data)
    all_data_to_pd.loc['KUR'] = all_data_to_pd.kurtosis().tolist()
    all_data_to_pd.loc['var'] = all_data_to_pd.var().tolist()
    all_data_to_pd.loc['skew'] = all_data_to_pd.skew().tolist()
    all_data_to_pd.loc['STD'] = all_data_to_pd.std().tolist()
    all_data_to_pd.loc['Mean'] = all_data_to_pd.mean().tolist()
    all_data_to_pd.loc['time'] ={'Bearing 1 Ch1': life_time , 'Bearing 2 Ch1':life_time,
                                 'Bearing 3 Ch1':37.9 , 'Bearing 4 Ch1': 38.72}
    all_data_to_pd.to_csv(path_all_data_csv + '\\' + 'all_data' + '.csv' , index=False )

add_featurs(allcsv)


data = pd.read_csv(path_all_data_csv+'\\all_data.csv' )
data1 = data.T


X = data1.loc[:, data1.columns != 61445]


bins = (30 ,37.5 ,50)
group_names = ['good_life' , 'normal_life']
data1[61445] = pd.cut(data1[61445],  bins = bins   , labels=group_names)
data1[61445].unique()
print(data1[61445].unique())
label_time = LabelEncoder()
data1[61445] = label_time.fit_transform(data1[61445])
y =(data1[61445])
X_train , X_test , y_train ,y_test  = train_test_split( X , y , test_size=0.2 )

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


clf = svm.SVC()
clf.fit(X_train, y_train)
pred_clf = clf.predict(X_test)
print(classification_report(y_test,pred_clf))
print(confusion_matrix(y_test , pred_clf))

print('Bearing 3 Predicting Time : ' + str(predictive_time))

