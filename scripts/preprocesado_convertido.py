# -*- coding: utf-8 -*-
"""
Archivo generado automáticamente a partir de un notebook (.ipynb).
Se preserva la secuencia original del proceso:
- Celdas Markdown -> comentarios
- Celdas de código -> código Python (sin modificar)
"""


#==============================================================================
# CELDA 1 | Tipo: CODE
#==============================================================================
# --- Código ---
import pandas as pd
import seaborn as sns
import matplotlib.pyplot  as plt
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


#==============================================================================
# CELDA 2 | Tipo: MARKDOWN
#==============================================================================
# ### https://www.kaggle.com/c/titanic/data


#==============================================================================
# CELDA 3 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic1 = pd.read_csv('../train.csv')


#==============================================================================
# CELDA 4 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic1.head()

# --- Salida (capturada del notebook) ---
#    PassengerId  Survived  Pclass  \
# 0            1         0       3   
# 1            2         1       1   
# 2            3         1       3   
# 3            4         1       1   
# 4            5         0       3   
# 
#                                                 Name     Sex   Age  SibSp  \
# 0                            Braund, Mr. Owen Harris    male  22.0      1   
# 1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
# 2                             Heikkinen, Miss. Laina  female  26.0      0   
# 3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
# 4                           Allen, Mr. William Henry    male  35.0      0   
# 
#    Parch            Ticket     Fare Cabin Embarked  
# 0      0         A/5 21171   7.2500   NaN        S  
# 1      0          PC 17599  71.2833   C85        C  
# 2      0  STON/O2. 3101282   7.9250   NaN        S  
# 3      0            113803  53.1000  C123        S  
# 4      0            373450   8.0500   NaN        S  


#==============================================================================
# CELDA 5 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic1.shape

# --- Salida (capturada del notebook) ---
# (891, 12)


#==============================================================================
# CELDA 6 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic2 = pd.read_csv('../test.csv')


#==============================================================================
# CELDA 7 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic2.head()

# --- Salida (capturada del notebook) ---
#    PassengerId  Pclass                                          Name     Sex  \
# 0          892       3                              Kelly, Mr. James    male   
# 1          893       3              Wilkes, Mrs. James (Ellen Needs)  female   
# 2          894       2                     Myles, Mr. Thomas Francis    male   
# 3          895       3                              Wirz, Mr. Albert    male   
# 4          896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)  female   
# 
#     Age  SibSp  Parch   Ticket     Fare Cabin Embarked  
# 0  34.5      0      0   330911   7.8292   NaN        Q  
# 1  47.0      1      0   363272   7.0000   NaN        S  
# 2  62.0      0      0   240276   9.6875   NaN        Q  
# 3  27.0      0      0   315154   8.6625   NaN        S  
# 4  22.0      1      1  3101298  12.2875   NaN        S  


#==============================================================================
# CELDA 8 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic2.shape

# --- Salida (capturada del notebook) ---
# (418, 11)


#==============================================================================
# CELDA 9 | Tipo: CODE
#==============================================================================
# --- Código ---
df_y = pd.read_csv('../titanic-y.csv',sep=';')


#==============================================================================
# CELDA 10 | Tipo: CODE
#==============================================================================
# --- Código ---
df_y.head()

# --- Salida (capturada del notebook) ---
#    PassengerId  Survived
# 0            1         0
# 1            2         1
# 2            3         1
# 3            4         1
# 4            5         0


#==============================================================================
# CELDA 11 | Tipo: CODE
#==============================================================================
# --- Código ---
df_y.shape

# --- Salida (capturada del notebook) ---
# (1309, 2)


#==============================================================================
# CELDA 12 | Tipo: CODE
#==============================================================================
# --- Código ---
len(df_y.PassengerId.unique())

# --- Salida (capturada del notebook) ---
# 1309


#==============================================================================
# CELDA 13 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic1.drop(columns = ['Survived'], inplace =True)


#==============================================================================
# CELDA 14 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic = pd.concat([df_titanic1, df_titanic2])
df_titanic.set_index(['PassengerId'], inplace=True)


#==============================================================================
# CELDA 15 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic.head()

# --- Salida (capturada del notebook) ---
#              Pclass                                               Name  \
# PassengerId                                                              
# 1                 3                            Braund, Mr. Owen Harris   
# 2                 1  Cumings, Mrs. John Bradley (Florence Briggs Th...   
# 3                 3                             Heikkinen, Miss. Laina   
# 4                 1       Futrelle, Mrs. Jacques Heath (Lily May Peel)   
# 5                 3                           Allen, Mr. William Henry   
# 
#                 Sex   Age  SibSp  Parch            Ticket     Fare Cabin  \
# PassengerId                                                                
# 1              male  22.0      1      0         A/5 21171   7.2500   NaN   
# 2            female  38.0      1      0          PC 17599  71.2833   C85   
# 3            female  26.0      0      0  STON/O2. 3101282   7.9250   NaN   
# 4            female  35.0      1      0            113803  53.1000  C123   
# 5              male  35.0      0      0            373450   8.0500   NaN   
# 
#             Embarked  
# PassengerId           
# 1                  S  
# 2                  C  
# 3                  S  
# 4                  S  
# 5                  S  


#==============================================================================
# CELDA 16 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic.columns

# --- Salida (capturada del notebook) ---
# Index(['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare',
#        'Cabin', 'Embarked'],
#       dtype='object')


#==============================================================================
# CELDA 17 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic.shape

# --- Salida (capturada del notebook) ---
# (1309, 10)


#==============================================================================
# CELDA 18 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic.describe()

# --- Salida (capturada del notebook) ---
#             Pclass          Age        SibSp        Parch         Fare
# count  1309.000000  1046.000000  1309.000000  1309.000000  1308.000000
# mean      2.294882    29.881138     0.498854     0.385027    33.295479
# std       0.837836    14.413493     1.041658     0.865560    51.758668
# min       1.000000     0.170000     0.000000     0.000000     0.000000
# 25%       2.000000    21.000000     0.000000     0.000000     7.895800
# 50%       3.000000    28.000000     0.000000     0.000000    14.454200
# 75%       3.000000    39.000000     1.000000     0.000000    31.275000
# max       3.000000    80.000000     8.000000     9.000000   512.329200


#==============================================================================
# CELDA 19 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic.dtypes

# --- Salida (capturada del notebook) ---
# Pclass        int64
# Name         object
# Sex          object
# Age         float64
# SibSp         int64
# Parch         int64
# Ticket       object
# Fare        float64
# Cabin        object
# Embarked     object
# dtype: object


#==============================================================================
# CELDA 20 | Tipo: MARKDOWN
#==============================================================================
# #### ¿Tenemos valores categóricos?


#==============================================================================
# CELDA 21 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic.dtypes

# --- Salida (capturada del notebook) ---
# Pclass        int64
# Name         object
# Sex          object
# Age         float64
# SibSp         int64
# Parch         int64
# Ticket       object
# Fare        float64
# Cabin        object
# Embarked     object
# dtype: object


#==============================================================================
# CELDA 22 | Tipo: MARKDOWN
#==============================================================================
# #### ¿Tenemos valores nulos?


#==============================================================================
# CELDA 23 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic.isnull().sum()

# --- Salida (capturada del notebook) ---
# Pclass         0
# Name           0
# Sex            0
# Age          263
# SibSp          0
# Parch          0
# Ticket         0
# Fare           1
# Cabin       1014
# Embarked       2
# dtype: int64


#==============================================================================
# CELDA 24 | Tipo: CODE
#==============================================================================
# --- Código ---
edad = df_titanic['Age'].value_counts(ascending=False)


#==============================================================================
# CELDA 25 | Tipo: CODE
#==============================================================================
# --- Código ---
edad.head(10)

# --- Salida (capturada del notebook) ---
# 24.0    47
# 22.0    43
# 21.0    41
# 30.0    40
# 18.0    39
# 25.0    34
# 28.0    32
# 36.0    31
# 26.0    30
# 29.0    30
# Name: Age, dtype: int64


#==============================================================================
# CELDA 26 | Tipo: CODE
#==============================================================================
# --- Código ---
plt.figure(figsize=(12,6))
sns.distplot(df_titanic['Age'].dropna(),kde=False,color='darkred',bins=30)

# --- Salida (capturada del notebook) ---
# <matplotlib.axes._subplots.AxesSubplot at 0x20d2fd33668>
# <Figure size 864x432 with 1 Axes>


#==============================================================================
# CELDA 27 | Tipo: CODE
#==============================================================================
# --- Código ---
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=df_titanic,palette='rainbow')

# --- Salida (capturada del notebook) ---
# <matplotlib.axes._subplots.AxesSubplot at 0x20d2fd19278>
# <Figure size 864x504 with 1 Axes>


#==============================================================================
# CELDA 28 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic.iloc[5]

# --- Salida (capturada del notebook) ---
# Pclass                     3
# Name        Moran, Mr. James
# Sex                     male
# Age                       25
# SibSp                      0
# Parch                      0
# Ticket                330877
# Fare                  8.4583
# Cabin                    NaN
# Embarked                   Q
# Name: 6, dtype: object


#==============================================================================
# CELDA 29 | Tipo: RAW
#==============================================================================
# (Tipo de celda no soportado: raw)


#==============================================================================
# CELDA 30 | Tipo: CODE
#==============================================================================
# --- Código ---
lista_edad = []

for i in range(len(df_titanic)):
    if pd.isnull(df_titanic.iloc[i]['Age']):
        if df_titanic.iloc[i]['Pclass'] == 1:
            lista_edad.append(39)
        elif df_titanic.iloc[i]['Pclass'] == 2:
            lista_edad.append(29)
        else:
            lista_edad.append(25)
    else:
        lista_edad.append(df_titanic.iloc[i]['Age'])
        
df_titanic['Age'] = lista_edad        
            


#==============================================================================
# CELDA 31 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic.isnull().sum()

# --- Salida (capturada del notebook) ---
# Pclass         0
# Name           0
# Sex            0
# Age            0
# SibSp          0
# Parch          0
# Ticket         0
# Fare           1
# Cabin       1014
# Embarked       0
# dtype: int64


#==============================================================================
# CELDA 32 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic[df_titanic['Embarked'].isnull()]

# --- Salida (capturada del notebook) ---
#              Pclass                                       Name     Sex   Age  \
# PassengerId                                                                    
# 62                1                        Icard, Miss. Amelie  female  38.0   
# 830               1  Stone, Mrs. George Nelson (Martha Evelyn)  female  62.0   
# 
#              SibSp  Parch  Ticket  Fare Cabin Embarked  
# PassengerId                                             
# 62               0      0  113572  80.0   B28      NaN  
# 830              0      0  113572  80.0   B28      NaN  


#==============================================================================
# CELDA 33 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic['Embarked'].value_counts()

# --- Salida (capturada del notebook) ---
# S    914
# C    270
# Q    123
# Name: Embarked, dtype: int64


#==============================================================================
# CELDA 34 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic['Embarked']=df_titanic['Embarked'].fillna(df_titanic['Embarked'].mode()[0])


#==============================================================================
# CELDA 35 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic['Fare'].mean()

# --- Salida (capturada del notebook) ---
# 33.2954792813456


#==============================================================================
# CELDA 36 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic['Fare']=df_titanic['Fare'].fillna(df_titanic['Fare'].mean())


#==============================================================================
# CELDA 37 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic.isnull().sum()

# --- Salida (capturada del notebook) ---
# Pclass         0
# Name           0
# Sex            0
# Age            0
# SibSp          0
# Parch          0
# Ticket         0
# Fare           0
# Cabin       1014
# Embarked       0
# dtype: int64


#==============================================================================
# CELDA 38 | Tipo: MARKDOWN
#==============================================================================
# #### ¿Qué hacemos con los nombres repetidos ?


#==============================================================================
# CELDA 39 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic['Name'].value_counts()

# --- Salida (capturada del notebook) ---
# Kelly, Mr. James                                          2
# Connolly, Miss. Kate                                      2
# Mitkoff, Mr. Mito                                         1
# Clarke, Mr. Charles Valentine                             1
# Beavan, Mr. William Thomas                                1
# MacKay, Mr. George William                                1
# Peter, Master. Michael J                                  1
# O'Dwyer, Miss. Ellen "Nellie"                             1
# O'Brien, Mrs. Thomas (Johanna "Hannah" Godfrey)           1
# Rintamaki, Mr. Matti                                      1
# Boulos, Miss. Nourelain                                   1
# Baclini, Miss. Helene Barbara                             1
# Youseff, Mr. Gerious                                      1
# Lyntakoff, Mr. Stanko                                     1
# Lahoud, Mr. Sarkis                                        1
# Johnson, Mr. Alfred                                       1
# Stokes, Mr. Philip Joseph                                 1
# Kalvik, Mr. Johannes Halvorsen                            1
# Phillips, Mr. Escott Robert                               1
# Navratil, Mr. Michel ("Louis M Hoffman")                  1
# Johanson, Mr. Jakob Alfred                                1
# Jenkin, Mr. Stephen Curnow                                1
# Johnson, Mr. Malkolm Joackim                              1
# Faunthorpe, Mr. Harry                                     1
# Svensson, Mr. Johan Cervin                                1
# Karlsson, Mr. Nils August                                 1
# Sheerlinck, Mr. Jan Baptist                               1
# Horgan, Mr. John                                          1
# Lievens, Mr. Rene Aime                                    1
# Jensen, Mr. Hans Peder                                    1
#                                                          ..
# Widener, Mrs. George Dunton (Eleanor Elkins)              1
# Keane, Mr. Andrew "Andy"                                  1
# Cook, Mrs. (Selena Rogers)                                1
# Greenfield, Mrs. Leo David (Blanche Strouse)              1
# Cavendish, Mrs. Tyrell William (Julia Florence Siegel)    1
# Minahan, Miss. Daisy E                                    1
# Louch, Mr. Charles Alexander                              1
# Crosby, Miss. Harriet R                                   1
# Beckwith, Mrs. Richard Leonard (Sallie Monypeny)          1
# Braund, Mr. Lewis Richard                                 1
# Yasbeck, Mrs. Antoni (Selini Alexander)                   1
# Peacock, Mrs. Benjamin (Edith Nile)                       1
# Albimona, Mr. Nassef Cassem                               1
# Taussig, Miss. Ruth                                       1
# Pernot, Mr. Rene                                          1
# Nankoff, Mr. Minko                                        1
# Turcin, Mr. Stjepan                                       1
# Asplund, Mr. Carl Oscar Vilhelm Gustafsson                1
# Spencer, Mrs. William Augustus (Marie Eugenie)            1
# Osen, Mr. Olaf Elon                                       1
# Asplund, Master. Clarence Gustaf Hugo                     1
# Byles, Rev. Thomas Roussel Davids                         1
# Gee, Mr. Arthur H                                         1
# Ali, Mr. William                                          1
# Johnson, Master. Harold Theodor                           1
# Buckley, Miss. Katherine                                  1
# Cann, Mr. Ernest Charles                                  1
# Calic, Mr. Jovo                                           1
# Hold, Mrs. Stephen (Annie Margaret Hill)                  1
# Allum, Mr. Owen George                                    1
# Name: Name, Length: 1307, dtype: int64


#==============================================================================
# CELDA 40 | Tipo: CODE
#==============================================================================
# --- Código ---
len(list(df_titanic['Name'].unique()))
len(df_titanic['Name'])
df_titanic['Name'][df_titanic['Name'].duplicated() == True]
print(df_titanic[df_titanic['Name'] == 'Kelly, Mr. James'])
print(df_titanic[df_titanic['Name'] == 'Connolly, Miss. Kate'])

# --- Salida (capturada del notebook) ---
#              Pclass              Name   Sex   Age  SibSp  Parch  Ticket  \
# PassengerId                                                               
# 697               3  Kelly, Mr. James  male  44.0      0      0  363592   
# 892               3  Kelly, Mr. James  male  34.5      0      0  330911   
# 
#                Fare Cabin Embarked  
# PassengerId                         
# 697          8.0500   NaN        S  
# 892          7.8292   NaN        Q  
#              Pclass                  Name     Sex   Age  SibSp  Parch  Ticket  \
# PassengerId                                                                     
# 290               3  Connolly, Miss. Kate  female  22.0      0      0  370373   
# 898               3  Connolly, Miss. Kate  female  30.0      0      0  330972   
# 
#                Fare Cabin Embarked  
# PassengerId                         
# 290          7.7500   NaN        Q  
# 898          7.6292   NaN        Q  


#==============================================================================
# CELDA 41 | Tipo: CODE
#==============================================================================
# --- Código ---
len(df_titanic['Name'].unique())

# --- Salida (capturada del notebook) ---
# 1307


#==============================================================================
# CELDA 42 | Tipo: MARKDOWN
#==============================================================================
# #### Podemos crear variables nuevas


#==============================================================================
# CELDA 43 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic['Title'] = df_titanic.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


#==============================================================================
# CELDA 44 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic.Title.unique()

# --- Salida (capturada del notebook) ---
# array(['Mr', 'Mrs', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms',
#        'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess',
#        'Jonkheer', 'Dona'], dtype=object)


#==============================================================================
# CELDA 45 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic.Title.value_counts()

# --- Salida (capturada del notebook) ---
# Mr          757
# Miss        260
# Mrs         197
# Master       61
# Dr            8
# Rev           8
# Col           4
# Major         2
# Ms            2
# Mlle          2
# Lady          1
# Jonkheer      1
# Mme           1
# Don           1
# Capt          1
# Countess      1
# Dona          1
# Sir           1
# Name: Title, dtype: int64


#==============================================================================
# CELDA 46 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic['Title'] = df_titanic['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df_titanic['Title'] = df_titanic['Title'].replace('Mlle', 'Miss')
df_titanic['Title'] = df_titanic['Title'].replace('Ms', 'Miss')
df_titanic['Title'] = df_titanic['Title'].replace('Mme', 'Mrs')


#==============================================================================
# CELDA 47 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic.drop(columns = ['Name'], inplace=True)


#==============================================================================
# CELDA 48 | Tipo: MARKDOWN
#==============================================================================
# #### ¿Tenemos valores únicos?


#==============================================================================
# CELDA 49 | Tipo: CODE
#==============================================================================
# --- Código ---
for j in df_titanic.columns:
    print(j)
    print(df_titanic[j].value_counts())

# --- Salida (capturada del notebook) ---
# Pclass
# 3    709
# 1    323
# 2    277
# Name: Pclass, dtype: int64
# Sex
# male      843
# female    466
# Name: Sex, dtype: int64
# Age
# 25.00    242
# 39.00     59
# 24.00     47
# 29.00     46
# 22.00     43
# 21.00     41
# 30.00     40
# 18.00     39
# 28.00     32
# 36.00     31
# 27.00     30
# 26.00     30
# 19.00     29
# 23.00     26
# 32.00     24
# 20.00     23
# 35.00     23
# 31.00     23
# 33.00     21
# 45.00     21
# 17.00     20
# 16.00     19
# 42.00     18
# 40.00     18
# 34.00     16
# 50.00     15
# 38.00     14
# 47.00     14
# 48.00     14
# 2.00      12
#         ... 
# 0.83       3
# 59.00      3
# 28.50      3
# 34.50      2
# 45.50      2
# 70.00      2
# 0.92       2
# 30.50      2
# 71.00      2
# 36.50      2
# 14.50      2
# 0.17       1
# 67.00      1
# 60.50      1
# 23.50      1
# 55.50      1
# 38.50      1
# 66.00      1
# 0.33       1
# 20.50      1
# 0.42       1
# 70.50      1
# 11.50      1
# 26.50      1
# 76.00      1
# 22.50      1
# 74.00      1
# 24.50      1
# 80.00      1
# 0.67       1
# Name: Age, Length: 98, dtype: int64
# SibSp
# 0    891
# 1    319
# 2     42
# 4     22
# 3     20
# 8      9
# 5      6
# Name: SibSp, dtype: int64
# Parch
# 0    1002
# 1     170
# 2     113
# 3       8
# 5       6
# 4       6
# 9       2
# 6       2
# Name: Parch, dtype: int64
# Ticket
# CA. 2343              11
# CA 2144                8
# 1601                   8
# 347082                 7
# 3101295                7
# 347077                 7
# PC 17608               7
# S.O.C. 14879           7
# 113781                 6
# 347088                 6
# 19950                  6
# 382652                 6
# 220845                 5
# 4133                   5
# 113503                 5
# PC 17757               5
# 349909                 5
# 16966                  5
# W./C. 6608             5
# 113760                 4
# C.A. 2315              4
# PC 17760               4
# C.A. 33112             4
# SC/Paris 2123          4
# PC 17755               4
# C.A. 34651             4
# 36928                  4
# 2666                   4
# 12749                  4
# 24160                  4
#                       ..
# 2664                   1
# PC 17601               1
# 2674                   1
# 3101267                1
# 2700                   1
# 364859                 1
# 350045                 1
# 4135                   1
# 2623                   1
# SOTON/O.Q. 3101310     1
# 330920                 1
# STON/O2. 3101270       1
# 236854                 1
# A/5 2817               1
# 367232                 1
# STON/O 2. 3101289      1
# 2670                   1
# 312992                 1
# 368402                 1
# 7267                   1
# 349221                 1
# 315096                 1
# 345501                 1
# 349242                 1
# PC 17475               1
# 349257                 1
# 315093                 1
# 65306                  1
# SC/A.3 2861            1
# 2657                   1
# Name: Ticket, Length: 929, dtype: int64
# Fare
# 8.0500     60
# 13.0000    59
# 7.7500     55
# 26.0000    50
# 7.8958     49
# 10.5000    35
# 7.7750     26
# 7.2292     24
# 7.9250     23
# 26.5500    22
# 7.8542     21
# 7.2250     21
# 8.6625     21
# 7.2500     18
# 0.0000     17
# 21.0000    14
# 16.1000    12
# 9.5000     12
# 27.7208    11
# 69.5500    11
# 14.5000    11
# 15.5000    10
# 7.8792     10
# 14.4542    10
# 7.7958     10
# 15.2458     9
# 24.1500     9
# 7.0500      9
# 52.0000     8
# 7.5500      8
#            ..
# 7.2833      1
# 8.4333      1
# 6.8583      1
# 9.3250      1
# 7.5792      1
# 7.5750      1
# 9.4750      1
# 28.7125     1
# 40.1250     1
# 25.5875     1
# 7.7292      1
# 9.8458      1
# 7.8500      1
# 7.7417      1
# 6.4500      1
# 7.7208      1
# 10.1708     1
# 12.6500     1
# 10.5167     1
# 34.6542     1
# 27.4458     1
# 26.3875     1
# 8.1375      1
# 15.5792     1
# 8.1583      1
# 8.0292      1
# 12.7375     1
# 8.6542      1
# 34.0208     1
# 7.1417      1
# Name: Fare, Length: 282, dtype: int64
# Cabin
# C23 C25 C27        6
# G6                 5
# B57 B59 B63 B66    5
# B96 B98            4
# D                  4
# F4                 4
# F2                 4
# C22 C26            4
# F33                4
# C78                4
# B58 B60            3
# E101               3
# E34                3
# C101               3
# A34                3
# B51 B53 B55        3
# B49                2
# D30                2
# E24                2
# B78                2
# C46                2
# E44                2
# E31                2
# E121               2
# C31                2
# C62 C64            2
# C65                2
# C52                2
# C92                2
# C93                2
#                   ..
# E52                1
# A5                 1
# F E46              1
# C70                1
# B50                1
# C110               1
# C105               1
# B82 B84            1
# B102               1
# A31                1
# A18                1
# B36                1
# A36                1
# E10                1
# B101               1
# C91                1
# A6                 1
# C132               1
# E60                1
# T                  1
# C28                1
# C51                1
# C82                1
# E39 E41            1
# B42                1
# D48                1
# C128               1
# E12                1
# B37                1
# D11                1
# Name: Cabin, Length: 186, dtype: int64
# Embarked
# S    916
# C    270
# Q    123
# Name: Embarked, dtype: int64
# Title
# Mr        757
# Miss      264
# Mrs       198
# Master     61
# Rare       29
# Name: Title, dtype: int64


#==============================================================================
# CELDA 50 | Tipo: MARKDOWN
#==============================================================================
# #### Variables categóricas


#==============================================================================
# CELDA 51 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic.dtypes

# --- Salida (capturada del notebook) ---
# Pclass        int64
# Sex          object
# Age         float64
# SibSp         int64
# Parch         int64
# Ticket       object
# Fare        float64
# Cabin        object
# Embarked     object
# Title        object
# dtype: object


#==============================================================================
# CELDA 52 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic['Sex'].unique()

# --- Salida (capturada del notebook) ---
# array([0, 1], dtype=int64)


#==============================================================================
# CELDA 53 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic['Sex'] = df_titanic['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


#==============================================================================
# CELDA 54 | Tipo: CODE
#==============================================================================
# --- Código ---
#¿borramos variable ticket?
df_titanic.drop('Ticket', axis=1, inplace=True)


#==============================================================================
# CELDA 55 | Tipo: CODE
#==============================================================================
# --- Código ---
len(df_titanic['Cabin'].unique())

# --- Salida (capturada del notebook) ---
# 187


#==============================================================================
# CELDA 56 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic['Cabin'].isnull().sum()

# --- Salida (capturada del notebook) ---
# 1014


#==============================================================================
# CELDA 57 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic.drop('Cabin', axis=1, inplace=True)


#==============================================================================
# CELDA 58 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic['Embarked'].unique()

# --- Salida (capturada del notebook) ---
# array(['S', 'C', 'Q'], dtype=object)


#==============================================================================
# CELDA 59 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic.head()

# --- Salida (capturada del notebook) ---
#              Pclass  Sex   Age  SibSp  Parch     Fare Embarked Title
# PassengerId                                                         
# 1                 3    0  22.0      1      0   7.2500        S    Mr
# 2                 1    1  38.0      1      0  71.2833        C   Mrs
# 3                 3    1  26.0      0      0   7.9250        S  Miss
# 4                 1    1  35.0      1      0  53.1000        S   Mrs
# 5                 3    0  35.0      0      0   8.0500        S    Mr


#==============================================================================
# CELDA 60 | Tipo: CODE
#==============================================================================
# --- Código ---
embarked = pd.get_dummies(df_titanic['Embarked'])


#==============================================================================
# CELDA 61 | Tipo: CODE
#==============================================================================
# --- Código ---
embarked?


#==============================================================================
# CELDA 62 | Tipo: CODE
#==============================================================================
# --- Código ---
embarked.head()

# --- Salida (capturada del notebook) ---
#              C  Q  S
# PassengerId         
# 1            0  0  1
# 2            1  0  0
# 3            0  0  1
# 4            0  0  1
# 5            0  0  1


#==============================================================================
# CELDA 63 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic.insert(8,'Cherbourg',embarked['C'])
df_titanic.insert(9,'Queenstown',embarked['Q'])
df_titanic.insert(10,'Southampton',embarked['S'])


#==============================================================================
# CELDA 64 | Tipo: RAW
#==============================================================================
# (Tipo de celda no soportado: raw)


#==============================================================================
# CELDA 65 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic['Title'].unique()

# --- Salida (capturada del notebook) ---
# array(['Mr', 'Mrs', 'Miss', 'Master', 'Rare'], dtype=object)


#==============================================================================
# CELDA 66 | Tipo: CODE
#==============================================================================
# --- Código ---
## Encode target labels with value between 0 and n_classes-1.
transform1 = LabelEncoder()
transform1.fit_transform(['Q','C','S'])
print(transform1.classes_)
print(transform1.transform(df_titanic['Embarked']))

prueba = transform1.transform(df_titanic['Embarked'])

# --- Salida (capturada del notebook) ---
# ['C' 'Q' 'S']
# [2 0 2 ... 2 2 0]


#==============================================================================
# CELDA 67 | Tipo: CODE
#==============================================================================
# --- Código ---
##Encode categorical features as a one-hot numeric array.
transform2 = OneHotEncoder() 
transform2.fit_transform(prueba.reshape(len(prueba), 1)).toarray()

# --- Salida (capturada del notebook) ---
# array([[0., 0., 1.],
#        [1., 0., 0.],
#        [0., 0., 1.],
#        ...,
#        [0., 0., 1.],
#        [0., 0., 1.],
#        [1., 0., 0.]])


#==============================================================================
# CELDA 68 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic.drop('Embarked', axis=1, inplace=True)


#==============================================================================
# CELDA 69 | Tipo: CODE
#==============================================================================
# --- Código ---
mapeo = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
df_titanic['Title'] = df_titanic['Title'].map(mapeo)


#==============================================================================
# CELDA 70 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic.isnull().sum()

# --- Salida (capturada del notebook) ---
# Pclass         0
# Sex            0
# Age            0
# SibSp          0
# Parch          0
# Fare           0
# Title          0
# Cherbourg      0
# Queenstown     0
# Southampton    0
# dtype: int64


#==============================================================================
# CELDA 71 | Tipo: CODE
#==============================================================================
# --- Código ---
df_titanic.dtypes

# --- Salida (capturada del notebook) ---
# Pclass           int64
# Sex              int32
# Age            float64
# SibSp            int64
# Parch            int64
# Fare           float64
# Title            int64
# Cherbourg        uint8
# Queenstown       uint8
# Southampton      uint8
# dtype: object


#==============================================================================
# CELDA 72 | Tipo: MARKDOWN
#==============================================================================
# #### Conjuntos Train-Test


#==============================================================================
# CELDA 73 | Tipo: CODE
#==============================================================================
# --- Código ---
titanic_y = pd.read_csv('../titanic-y.csv',sep = ';')


#==============================================================================
# CELDA 74 | Tipo: CODE
#==============================================================================
# --- Código ---
titanic_y.set_index('PassengerId', inplace=True)


#==============================================================================
# CELDA 75 | Tipo: CODE
#==============================================================================
# --- Código ---
titanic_y.head()

# --- Salida (capturada del notebook) ---
#              Survived
# PassengerId          
# 1                   0
# 2                   1
# 3                   1
# 4                   1
# 5                   0


#==============================================================================
# CELDA 76 | Tipo: CODE
#==============================================================================
# --- Código ---
titanic_y.Survived.unique()

# --- Salida (capturada del notebook) ---
# array([0, 1], dtype=int64)


#==============================================================================
# CELDA 77 | Tipo: CODE
#==============================================================================
# --- Código ---
 X_train, X_test, y_train, y_test = train_test_split( df_titanic,
                        titanic_y,
                        test_size=0.3,
                        random_state=42,
                        stratify = titanic_y)


#==============================================================================
# CELDA 78 | Tipo: CODE
#==============================================================================
# --- Código ---
len(X_train)

# --- Salida (capturada del notebook) ---
# 916


#==============================================================================
# CELDA 79 | Tipo: CODE
#==============================================================================
# --- Código ---
len(X_test)

# --- Salida (capturada del notebook) ---
# 393


#==============================================================================
# CELDA 80 | Tipo: CODE
#==============================================================================
# --- Código ---
len(y_train)

# --- Salida (capturada del notebook) ---
# 916


#==============================================================================
# CELDA 81 | Tipo: CODE
#==============================================================================
# --- Código ---
len(y_test)

# --- Salida (capturada del notebook) ---
# 393


#==============================================================================
# CELDA 82 | Tipo: CODE
#==============================================================================
# --- Código ---
traindf = pd.concat([X_train,y_train], axis=1)


#==============================================================================
# CELDA 83 | Tipo: CODE
#==============================================================================
# --- Código ---
traindf.head()

# --- Salida (capturada del notebook) ---
#              Pclass  Sex   Age  SibSp  Parch     Fare  Title  Cherbourg  \
# PassengerId                                                               
# 494               1    0  71.0      0      0  49.5042      1          1   
# 462               3    0  34.0      0      0   8.0500      1          0   
# 1286              3    0  29.0      3      1  22.0250      1          0   
# 1130              2    1  18.0      1      1  13.0000      2          0   
# 461               1    0  48.0      0      0  26.5500      1          0   
# 
#              Queenstown  Southampton  Survived  
# PassengerId                                     
# 494                   0            0         0  
# 462                   0            1         0  
# 1286                  0            1         0  
# 1130                  0            1         1  
# 461                   0            1         1  


#==============================================================================
# CELDA 84 | Tipo: CODE
#==============================================================================
# --- Código ---
testdf = X_test


#==============================================================================
# CELDA 85 | Tipo: MARKDOWN
#==============================================================================
# ### Correlación de variables


#==============================================================================
# CELDA 86 | Tipo: CODE
#==============================================================================
# --- Código ---
features = ['Pclass','Sex','Age']
from pandas.tools.plotting import scatter_matrix

scatter_matrix(traindf[features], figsize = (12, 12), diagonal = 'kde');

# --- Salida (capturada del notebook) ---
# C:\Users\cx02202\AppData\Local\Continuum\anaconda3\lib\site-packages\ipykernel_launcher.py:4: FutureWarning: 'pandas.tools.plotting.scatter_matrix' is deprecated, import 'pandas.plotting.scatter_matrix' instead.
#   after removing the cwd from sys.path.
# <Figure size 864x864 with 9 Axes>


#==============================================================================
# CELDA 87 | Tipo: CODE
#==============================================================================
# --- Código ---
traindf['Age'][traindf['Survived']==0].hist(bins=20)

# --- Salida (capturada del notebook) ---
# <matplotlib.axes._subplots.AxesSubplot at 0x20d3039ea90>
# <Figure size 432x288 with 1 Axes>


#==============================================================================
# CELDA 88 | Tipo: CODE
#==============================================================================
# --- Código ---
traindf['Age'][traindf['Survived']==1].hist(bins=20)

# --- Salida (capturada del notebook) ---
# <matplotlib.axes._subplots.AxesSubplot at 0x20d303fd048>
# <Figure size 432x288 with 1 Axes>


#==============================================================================
# CELDA 89 | Tipo: MARKDOWN
#==============================================================================
# #### Al comparar los histogramas de los que han sobrevivido por edades se puede ver que la supervivencia aumenta por debajo de los 20 años y vuelve aumentar en torno a los 30 y los 45 años. Esto se puede utilizar para realizar una agrupación de las variables. Podemos ver como se comporta el modelo en una primera versión y añadir una transformación de variables


#==============================================================================
# CELDA 90 | Tipo: MARKDOWN
#==============================================================================
# ### Escalado de Características


#==============================================================================
# CELDA 91 | Tipo: CODE
#==============================================================================
# --- Código ---
## Prueba ejemplo de escalado (cuando el modelo de ML le da más importancia a una categoría que a otra).

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
prueba_escalado = sc.fit_transform(X_train)


#==============================================================================
# CELDA 92 | Tipo: CODE
#==============================================================================
# --- Código ---
prueba_escalado[0]

# --- Salida (capturada del notebook) ---
# array([ 0.83946169, -0.74857042,  0.20352157, -0.47350156, -0.42482001,
#        -0.47346752, -0.71760571, -0.49940298, -0.32495634,  0.64706969])


#==============================================================================
# CELDA 93 | Tipo: MARKDOWN
#==============================================================================
# #### Podemos normalizar o estandarizar
#
# #### Estandarizar =  ( x – media(x) ) / desviación típica(x)
# #### Normalizar = (x-min(x) / max-min(x))
#
# #### Con esto conseguimos que ninguna variable domine sobre otra


#==============================================================================
# CELDA 94 | Tipo: MARKDOWN
#==============================================================================
# ### Detección outliers


#==============================================================================
# CELDA 95 | Tipo: CODE
#==============================================================================
# --- Código ---
sns.boxplot(x=X_train['Fare'])

# --- Salida (capturada del notebook) ---
# <matplotlib.axes._subplots.AxesSubplot at 0x1783ab4b828>
# <Figure size 432x288 with 1 Axes>


#==============================================================================
# CELDA 96 | Tipo: CODE
#==============================================================================
# --- Código ---
X_train[X_train['Fare'] > 262]

# --- Salida (capturada del notebook) ---
#              Pclass  Sex   Age  SibSp  Parch      Fare  Title  Cherbourg  \
# PassengerId                                                                
# 1267              1    1  45.0      0      0  262.3750      2          1   
# 1034              1    0  61.0      1      3  262.3750      1          1   
# 945               1    1  28.0      3      2  263.0000      2          0   
# 956               1    0  13.0      2      2  262.3750      4          1   
# 1235              1    1  58.0      0      1  512.3292      3          1   
# 743               1    1  21.0      2      2  262.3750      2          1   
# 680               1    0  36.0      0      1  512.3292      1          1   
# 312               1    1  18.0      2      2  262.3750      2          1   
# 738               1    0  35.0      0      0  512.3292      1          1   
# 916               1    1  48.0      1      3  262.3750      3          1   
# 259               1    1  35.0      0      0  512.3292      2          1   
# 439               1    0  64.0      1      4  263.0000      1          0   
# 961               1    1  60.0      1      4  263.0000      3          0   
# 
#              Queenstown  Southampton  
# PassengerId                           
# 1267                  0            0  
# 1034                  0            0  
# 945                   0            1  
# 956                   0            0  
# 1235                  0            0  
# 743                   0            0  
# 680                   0            0  
# 312                   0            0  
# 738                   0            0  
# 916                   0            0  
# 259                   0            0  
# 439                   0            1  
# 961                   0            1  


#==============================================================================
# CELDA 97 | Tipo: MARKDOWN
#==============================================================================
# #### Hay varias maneras de "corregir" estos outliers. Aquí se va a utilizar la función matemática Z_score


#==============================================================================
# CELDA 98 | Tipo: CODE
#==============================================================================
# --- Código ---
z = np.abs(stats.zscore(X_train['Fare']))
print(z)

# --- Salida (capturada del notebook) ---
# [0.47346752 2.45248078 0.1861282  ... 0.48380801 0.0639157  0.47634465]


#==============================================================================
# CELDA 99 | Tipo: CODE
#==============================================================================
# --- Código ---
threshold = 3
print(np.where(z > 3))

# --- Salida (capturada del notebook) ---
# (array([   7,    8,   35,  127,  183,  184,  199,  219,  271,  274,  328,
#         329,  420,  508,  535,  684,  725,  726,  741,  771,  786,  790,
#         793,  829,  837,  862,  873,  926,  952, 1004, 1030, 1043],
#       dtype=int64),)


#==============================================================================
# CELDA 100 | Tipo: CODE
#==============================================================================
# --- Código ---
X_train.reset_index().iloc[7]

# --- Salida (capturada del notebook) ---
# PassengerId    1267.000
# Pclass            1.000
# Sex               1.000
# Age              45.000
# SibSp             0.000
# Parch             0.000
# Fare            262.375
# Title             2.000
# Cherbourg         1.000
# Queenstown        0.000
# Southampton       0.000
# Name: 7, dtype: float64


#==============================================================================
# CELDA 101 | Tipo: CODE
#==============================================================================
# --- Código ---
traindf.to_csv('train_titanic.csv')


#==============================================================================
# CELDA 102 | Tipo: CODE
#==============================================================================
# --- Código ---
testdf.to_csv('test_titanic.csv')
