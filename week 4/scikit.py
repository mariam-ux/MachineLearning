from sklearn.pipeline import Pipeline, full_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import pandas as pd


data =  pd.read_csv('Social_Network_Ads.csv')


# simple imputer is to clean the missing data filling it with the media, mean or avrage
#very pipline should end with fit transform of the data 
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(stratgy="median")),
    ('std_scaler',StandardScaler())
])

data_num_tr = num_pipeline.fit_transform(data)

#split and train the data
#must declare x an dy first from the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# now we choose a model and train the data 
model = LinearRegression()
model(X_train, y_train)

some_data = X_test.iloc[:5]

some_labels = y_test.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", model.predict(some_data_prepared))

# to store the obtained model locally and load it whenever we need it again 
from sklearn.externals import joblib
joblib.dump(model, "model.pkl")
model_laod = joblib.load("model.pkl")

# we use gridnsearch to look for the best combination of the liniear regression 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

lin_model = LinearRegression()

param_grid = {
    'fir_intercept': [True, False],
    'normalize': [True, False]
}

grid_search = GridSearchCV(estimator=lin_model, param_grid=param_grid, cv=5)

# Fit the model, Assuming that X and y are given
grid_search.fit(X,y)

print("best parameters: ", grid_search.best_params_)