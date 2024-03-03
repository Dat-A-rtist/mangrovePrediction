import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# random seed
seed = 42

# Read original dataset
anngrow_df = pd.read_csv('data/Annual_growth.csv')
annsurv_df = pd.read_csv('data/Annual_survival.csv')

anngrow_df.sample(frac=1, random_state=seed)
annsurv_df.sample(frac=1, random_state=seed)

# prep and selecting features and target data
merged_df = pd.merge(anngrow_df, annsurv_df,how = 'inner', on = ['Species','Patch_type'])
merged_df.drop_duplicates(inplace=True)
final_merge = merged_df[['Species','Growth','Log_size_x','Patch_type','Survival']]
final_merge['Species'].replace(['Avicennia marina','Rhizophora mucronata','Sonneratia alba'],[0,1,2],inplace=True)
final_merge['Patch_type'].replace(['Connected','Rectangular','Winged'],[0,1,2],inplace=True)

X = final_merge[['Species', 'Growth', 'Log_size_x', 'Patch_type']]
y = final_merge[['Survival']]

# split data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

# create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=100)

# train the classifier on the training data
clf.fit(X_train, y_train)

# predict on the test set
y_pred = clf.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred) #Accuracy: 0.8710196722367107
#accuracy changed after manual file update to Accuracy: 0.5597782290558618
print(f"Accuracy: {accuracy}")

# save the model to disk
joblib.dump(clf, 'rf_model.sav')