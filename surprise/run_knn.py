#run a simple knn 
from surprise import Dataset, KNNBasic
from surprise.model_selection import cross_validate

#input the movielens 100k
data = Dataset.load_builtin("ml-100k")

#training data 
trainset = data.build_full_trainset()

algo = KNNBasic()
algo.fit(trainset)

#idk what this does - got it from the documentation
uid = str(196)  # raw user id (as in the ratings file). They are **strings**!
iid = str(302)  # raw item id (as in the ratings file). They are **strings**!

# get a prediction for specific users and items.
pred = algo.predict(uid, iid, r_ui=4, verbose=True)

#use cross validation to get the results with mean absolute error (MAE) and root mean squared error (RMSE)
results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print(results)
print("done")