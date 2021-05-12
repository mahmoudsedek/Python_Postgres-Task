import sqlalchemy as db
import pandas as pd
import pickle
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

con = db.create_engine('postgresql://iti:iti@localhost/python_task_db')
# con.table_names()

query = '''
SELECT * FROM "Iris"

df = pd.read_sql(query, con)
df.head()
'''

table_name = 'Iris'
table_df = pd.read_sql(table_name , con = con , columns = ['sepal_length'
                                                        ,'sepal_width'
                                                        ,'petal_length'
                                                        ,'petal_width'
                                                        ,'species'])
# table_df.sample(5)

dropped_species = table_df.drop(['species'],axis=1)
# dropped_species.info()

species_df = table_df['species']
# species_df.sample(5)

# use train/test split with different random_state values
X_train, X_test, y_train, y_test = train_test_split(dropped_species, species_df, test_size=0.4, random_state=5)

# check classification accuracy of KNN with K=15
knn = KNeighborsClassifier(n_neighbors=15)

# Training the model
knn.fit(X_train, y_train)

# Takes new data to test based on it
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

# save the model to disk
filepath = '/home/mahmoud/Desktop/ML_Model/ML_Model.sav'
pickle.dump(knn, open(filepath, 'wb'))
