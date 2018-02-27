
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def load_data(train_set, test_set):
    
    df_train = pd.read_csv(train_set, header=None)
    df_test = pd.read_csv(test_set, header=None)
    
    return df_train, df_test

def load_training_labels(labels_dir):
    
    df_labels = pd.read_csv(labels_dir, header=None)
    return df_labels


def random_forest_classifier(features, target):
    
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf



    
train_df, test_df = load_data("train.csv", "test.csv")
train_df = train_df.as_matrix()
test_df = test_df.as_matrix()
df_labels = load_training_labels("trainLabels.csv")
df_labels = df_labels.as_matrix().flatten()

clf = random_forest_classifier(train_df, df_labels)

predictions = clf.predict(test_df)

print(predictions)

final_df = pd.DataFrame((dict(id = np.arange(1, predictions.shape[0]+1), solution=predictions)))

print ("Train Accuracy :: ", accuracy_score(df_labels, clf.predict(train_df)))

final_df.to_csv('predictions.csv', index=False)


