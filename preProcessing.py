#Importing libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
import os
def getdata():
    #Importing dataset
    df = pd.read_csv("Movies_training.csv")
    # print(df.shape)
    #drop the const. coulm
    df = df.loc[:, df.apply(pd.Series.nunique) != 1]
    #drop rows that none data in these coulmns
    df = df.dropna(subset=['IMDb', 'Runtime', 'Country','Language','Directors','Genres','Year',])
    # replace nan to str nan
    df['Age'] = df['Age'].replace(np.nan, 'nan')
    df['Rotten Tomatoes'] = df['Rotten Tomatoes'].replace(np.nan, 'nan')
    #lablel to num
    number = preprocessing.LabelEncoder()
    df['Directors'] = number.fit_transform(df['Directors'])
    df['Age'] = number.fit_transform(df['Age'])
    df['Rotten Tomatoes'] = number.fit_transform(df['Rotten Tomatoes'])
    df['Country'] = number.fit_transform(df['Country'])
    df['Directors'] = number.fit_transform(df['Directors'])

    # one_hot encoding
    Gd = df
    GF = Gd.Genres.str.split(r'\s*,\s*', expand=True).apply(pd.Series.value_counts, 1) .iloc[:, 1:].fillna(0, downcast='infer')
    df1 = pd.concat([df, GF.reindex(df.index)], axis=1, join='inner')

    # CF = df.Country.str.split(r'\s*,\s*', expand=True).apply(pd.Series.value_counts, 1).iloc[:, 1:].fillna(0,downcast='infer')
    # df2 = pd.concat([df1, CF.reindex(df1.index)], axis=1, join='inner')

    LF = df.Language.str.split(r'\s*,\s*', expand=True).apply(pd.Series.value_counts, 1).iloc[:, 1:].fillna(0,downcast='infer')
    df = pd.concat([df1, LF.reindex(df1.index)], axis=1, join='inner')


    #scaling the data
    float_array = pd.Series(df['Year']).values.astype(float).reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(float_array)
    df_normalized = pd.DataFrame(scaled_array)
    df = pd.concat([df_normalized, df.reindex(df_normalized.index)], axis=1, join='inner')

    float_array = pd.Series(df['Age']).values.astype(float).reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(float_array)
    df_normalized = pd.DataFrame(scaled_array)
    df = pd.concat([df_normalized, df.reindex(df_normalized.index)], axis=1, join='inner')

    float_array = pd.Series(df['Rotten Tomatoes']).values.astype(float).reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(float_array)
    df_normalized = pd.DataFrame(scaled_array)
    df = pd.concat([df_normalized, df.reindex(df_normalized.index)], axis=1, join='inner')

    float_array = pd.Series(df['Runtime']).values.astype(float).reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(float_array)
    df_normalized = pd.DataFrame(scaled_array)
    df = pd.concat([df_normalized, df.reindex(df_normalized.index)], axis=1, join='inner')

    float_array = pd.Series(df['Country']).values.astype(float).reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(float_array)
    df_normalized = pd.DataFrame(scaled_array)
    df = pd.concat([df_normalized, df.reindex(df_normalized.index)], axis=1, join='inner')

    float_array = pd.Series(df['Directors']).values.astype(float).reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(float_array)
    df_normalized = pd.DataFrame(scaled_array)
    df = pd.concat([df_normalized, df.reindex(df_normalized.index)], axis=1, join='inner')


    #make the rate last column
    rate = df['IMDb']
    df['IMDb_rate']=rate
    # drop all unuse cloumns
    df.drop(labels=['IMDb', 'Age', 'Rotten Tomatoes', 'Runtime','Title','Language','Country','Genres','Directors','Year',], axis=1, inplace=True)
    #df.insert( -1,'IMDb', rate)
    #
    # df = df.drop('Title', 1)
    # df = df.drop('Language', 1)
    # df = df.drop('Country', 1)
    # df = df.drop('Genres', 1)
    # df = df.drop('Directors', 1)
    # df = df.drop('Year', 1)


    # gf = pd.Series(df['Genres']).apply(frozenset).to_frame(name='Genres')
    # for Genres in frozenset.union(*df.Genres):
    #     gf[Genres] = df.apply(lambda _: int(Genres in _.Genres), axis=1)

    #print(df.shape)
    # separt = df.set_index('Title').Genres.str.split(',', expand=True).stack()
    # Genres= pd.get_dummies(separt, prefix='G').groupby(level=0).sum()
    # df = df.append(Genres)

    #


    # separt = df.set_index('Title').Country.str.split(',', expand=True).stack()
    # Country = pd.get_dummies(separt, prefix='C').groupby(level=0).sum()
    # df = df.append(Country)
    #
    # separt = df.set_index('Title').Language.str.split(',', expand=True).stack()
    # Language = pd.get_dummies(separt, prefix='L').groupby(level=0).sum()
    # df = df.append(Language)
    # drop the title
    #One_hot_encoding
    # df = df.drop('Genres', 1).join(df.Genres.str.join('').str.get_dummies())
    # df = df.drop('Country', 1).join(df.Country.str.join('').str.get_dummies())
    #  df = df.drop('Language', 1).join(df.Language.str.join('').str.get_dummies())

    df= df.dropna(0)
    # print(df.shape)
    # print(df.info())
    df.to_csv('Preproceced_data.csv')


def preProcessing():
    if (os.path.exists("Preproceced_data.csv")): # If you have already created the dataset:
        data= pd.read_csv("Preproceced_data.csv")

    else:
        getdata()
        data= pd.read_csv("Preproceced_data.csv")
    return data
