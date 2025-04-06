import mglearn
from sklearn.datasets import load_digits#hand written digits 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#load the dataset
digits = load_digits()
X, y = digits.data, digits.target

#creat a logistic regression classifier using OvA stategy
ova_clf = LogisticRegression(multi_class='ovr', solver='liblinear')
#perform cross-validation with cross_val_score to evaluate each classifier's accuracy over 3 fold
scores_ova = cross_val_score(ova_clf, X, y, cv=3, scoring='accuracy')

print(f'acc with ova: {scores_ova.mean():.4f}')