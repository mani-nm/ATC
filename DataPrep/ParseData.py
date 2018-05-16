import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class ParseData:
    def __init__(self, file=None, header=None):
        self.data = pd.read_csv(file, header=header)
        self.columns = self.data.columns.tolist()
        self.dim = self.data.shape

    def drop_negatives(self):
        #temp_df = self.data.replace(-1, np.nan)
        #temp_df = temp_df.replace(temp_df.)
        temp_df = self.data[self.data >= 0].dropna()
        print("Dimension of complete dataset: ", temp_df.shape)
        return temp_df




#data = np.genfromtxt("inp_file1.csv", delimiter=',')
#row_ids = np.where(self.data[:, col] == val)

obj = ParseData("inp_file1.csv")
df = obj.drop_negatives()
X_col = obj.columns[1:-1]
y_col = obj.columns[-1]

X = df[X_col]
y = df[y_col]

clf = RandomForestClassifier(max_depth=3, class_weight="balanced")
score = cross_validate(clf, X, y, cv=5, scoring={'recall_micro'})

print(score)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

print(X_train.shape, X_test.shape)

model = clf.fit(X_train, y_train)
print(model.score(X_test, y_test))

y_pred = model.predict(X_test)

#conf_mat = confusion_matrix(y_test, y_pred, labels=[0, 1])

#print("Confusion Matrix:\n", conf_mat)
print("Recall: ", recall_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))





