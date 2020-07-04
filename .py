import  pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

column_name=["user_id","name_id","rating","timestamp"]
read=pd.read_csv("C:/Users/HP/Desktop/ml-100k/u.data",sep="\t",names=column_name)
read.head()
read.head()

read.info()

read.shape

read1=pd.read_csv("C:/Users/HP/Desktop/ml-100k/u.item",sep="\|",header=None)

read_data=read1[[0,1]]

read_data.columns=['name_id','title']
read_data.head()

dg=pd.merge(read,read_data,on="name_id")
dg.tail()

import matplotlib.pyplot as plt
import seaborn as sns


dg

dg.groupby('title').mean()['rating'].sort_values(ascending=False)

dg.groupby('title').count()['rating'].sort_values(ascending=False)

rating=pd.DataFrame(dg.groupby('title').count()['rating'])
rating

rating['mean of rating']=pd.DataFrame(dg.groupby('title').mean()['rating'])

rating

rating.sort_values(by="mean of rating",ascending=False)

rating.columns=['count','rating']

rating.sort_values(by="rating",ascending=False)

plt.hist(rating["count"],bins=50)
plt.show()

plt.hist(rating["rating"],bins=50)
plt.show()

sns.jointplot(x='rating',y='count',data=rating,kind='scatter',alpha=0.5)


dg.head()

movirate=dg.pivot_table(index="user_id",columns="title",values="rating")

movirate.head()

rating.sort_values('count',ascending=False).head()

star_wars_user_rating=movirate['Star Wars (1977)']

star_wars_user_rating.head()

similar_to_starwars=movirate.corrwith(star_wars_user_rating)

similar_to_starwars

convert_corr_of_starwars=pd.DataFrame(similar_to_starwars,columns=['Correlation'])

convert_corr_of_starwars.dropna(inplace=True)

convert_corr_of_starwars.head()

convert_corr_of_starwars.sort_values('Correlation',ascending=False).head(10)

rating

corr_star=convert_corr_of_starwars.join(rating['count'])

corr_star.head()

corr_star[corr_star['count']>100].sort_values('Correlation',ascending=False)

def predict_movies(movie_name):
    movie_user_rating=movirate[movie_name]
    similar_to_movie=movirate.corrwith(movie_user_rating)
    convert_corr_of_movie=pd.DataFrame(similar_to_movie,columns=['Correlation'])
    convert_corr_of_movie.dropna(inplace=True)
    convert_corr_of_movie=convert_corr_of_movie.join(rating['count'])
    predictions=convert_corr_of_movie[convert_corr_of_movie['count']>100].sort_values('Correlation',ascending=False)
    
    
    return predictions

prediction=predict_movies("Titanic (1997)")

prediction.head()



