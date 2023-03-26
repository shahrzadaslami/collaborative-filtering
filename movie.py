from django.forms import inlineformset_factory
import matplotlib
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt

#read the data from the file
#sorting movie info into a pandas dataframe
movies_df = pd.read_csv('movies.csv')
#sort the user info into a pandas dataframe
ratings_df = pd.read_csv('ratings.csv')

#head is a function that takes first N rows of datafram. N defualt is 5.
movies_df.head()

#the release date is written within the title in the dataset
#we want to remove it and place it in its own column
#we use "extract" from pandas for this

movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand =False)
#removew the parantheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand =False)
#remove the years from "title" column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '', regex=True)
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

#see the result

movies_df.head()

#we don't need genres yet lets drop them

movies_df = movies_df.drop(columns='genres')

movies_df.head()
print('Movies file trimmed')

ratings_df.head()
#we don't need the timestamps
ratings_df = ratings_df.drop(columns='timestamp')

ratings_df.head()

#create a user input data (this is the user that movies will be suggested to)
#this is the info the loged in user has on thier profile
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5},
            {'title':'Heat', 'rating':3.5},
            {'title':'Heat', 'rating':3.5},
            {'title':'Persuasion', 'rating':4.5},
            {'title':'Copycat', 'rating':4.5}
         ] 

inputMovies = pd.DataFrame(userInput)
inputMovies
#extract input movies ID from data file and add it to them
#filter movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#merge it now to get the movie Id
inputMovies = pd.merge(inputId, inputMovies)
#drop the infp we don't need 
inputMovies = inputMovies.drop(columns='year')
inputMovies

#now haiving the movie ID we can get the users that have seen this movie and reviewed it

userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
userSubset.head()

#Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
userSubsetGroup = userSubset.groupby(['userId'])

#Let's also sort these groups so the users 
# that share the most movies in common with 
# the input have higher priority.

userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)
print(userSubsetGroup[0:3])

userSubsetGroup = userSubsetGroup[0:100]

#Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficiency
pearsonCorrelationDict = {}

for name, group in userSubsetGroup:
    #Let's start by sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    #get the N for the formula
    nRatings = len(group)
    #get the review scores for the movies that both users have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    tempRatingList = temp_df['rating'].tolist()
    #put the current user reviews in alist format
    tempGroupList = group['rating'].tolist()
    #calculate the Pearson Correlation between the two users
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList), 2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList), 2)/float(nRatings)
    Sxy = sum(i*j for i,j in zip(tempRatingList, tempGroupList))  - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)


    if Sxx != 0 and Syy != 0 and Sxy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)

    else:
        pearsonCorrelationDict[name] = 0


pearsonCorrelationDict.items()
pearsonDf = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDf.columns = ['similarityIndex']
pearsonDf['userId'] = pearsonDf.index
pearsonDf.index = range(len(pearsonDf))
pearsonDf.head()


#get top 50 users that are most similar to input user
topUsers = pearsonDf.sort_values(by='similarityIndex', ascending=False)[0:50]
topUsers.head()
topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
topUsersRating.head()

#Multiplies the similarity by the user's ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
topUsersRating.head()

#Applies a sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating.head()



#Creates an empty dataframe
recommendation_df = pd.DataFrame()
#Now we take the weighted average
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
recommendation_df.head()



recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
recommendation_df.head(10)
movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]
