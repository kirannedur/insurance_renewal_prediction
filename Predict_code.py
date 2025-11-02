#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import os

current_directory = os.getcwd()
print(current_directory)


# In[3]:


train=pd.read_csv('train_Ins.csv')

train.drop(columns=['sourcing_channel', 'residence_area_type'], inplace=True)
train.head()


# In[4]:


# !pip install lazypredict


# In[5]:


test=pd.read_csv('test_66516Ee.csv')

test.drop(columns=['sourcing_channel', 'residence_area_type'], inplace=True)
test.head()


# In[6]:


train=train.drop(['id'],axis=1)
train.head()


# In[7]:


test=test.drop(['id'],axis=1)
test.head()


# In[8]:


train=pd.get_dummies(train,prefix_sep='__')
train.head()


# In[9]:


test=pd.get_dummies(test,prefix_sep='__')
test.head()


# In[10]:


# !rm -r kuma_utils
get_ipython().system('git clone https://github.com/analokmaus/kuma_utils.git')


# In[11]:


import sys
sys.path.append("kuma_utils/")
from kuma_utils.preprocessing.imputer import LGBMImputer


# In[12]:


col=train.columns.tolist()
col.remove('renewal')
col[:5]


# In[13]:


from kuma_utils.preprocessing.imputer import LGBMImputer


# In[14]:


from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from lightgbm import LGBMRegressor


# In[15]:


lgbm_imtr = IterativeImputer(estimator=LGBMRegressor())


# In[16]:


# 🚨 Change 'estimator_params' to 'lgbm_params'
##lgbm_imtr = LGBMImputer(params=lgbm_params, verbose=True)

train_iterimp = lgbm_imtr.fit_transform(train[col])
test_iterimp = lgbm_imtr.transform(test[col])

# Create train test imputed dataframe
train_ = pd.DataFrame(train_iterimp, columns=col)
test = pd.DataFrame(test_iterimp, columns=col)


# In[17]:


train_['renewal'] = train['renewal']
train_.head()


# In[18]:


def undummify(df, prefix_sep="__"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df


# In[19]:


train=undummify(train_)
train.head()


# In[20]:


test=undummify(test)
test.head()


# In[21]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[22]:


import numpy as np
train = reduce_mem_usage(train)
train.info()


# In[23]:


test = reduce_mem_usage(test)
test.info()


# In[24]:


train=train.sample(n=30000)
train.shape


# In[25]:


y = train.pop('renewal')
X = train


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42,shuffle=True, stratify=y)


# In[27]:


from lazypredict.Supervised import LazyClassifier
print("LazyPredict is ready to use!")


# In[28]:


import lazypredict, os
print("LazyPredict location:", os.path.dirname(lazypredict.__file__))


# In[29]:


from lazypredict.Supervised import LazyClassifier


# In[30]:


clf = LazyClassifier(verbose=0,predictions=True)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
models


# In[31]:


from sklearn.metrics import classification_report
for i in predictions.columns.tolist():
    print('\t\t',i,'\n')
    print(classification_report(y_test, predictions[i]),'\n')


# In[32]:


best_model_name = models.index[0]
best_model_score = models.loc[best_model_name, 'Accuracy']

print(f"Best model: {best_model_name} with Accuracy = {best_model_score:.4f}")


# In[33]:


from sklearn.preprocessing import LabelEncoder

X_encoded = X_train.copy()
for col in X_encoded.select_dtypes(include=['object']).columns:
    X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))

X_test_encoded = X_test.copy()
for col in X_test_encoded.select_dtypes(include=['object']).columns:
    X_test_encoded[col] = LabelEncoder().fit_transform(X_test_encoded[col].astype(str))


# In[34]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, classification_report


# In[35]:


X.head(3)


# In[36]:


label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le  # store encoder if you need to inverse-transform later


# In[37]:


print("Data types after encoding:\n", X.dtypes)


# In[38]:


# nominal_cols = ['sourcing_channel', 'residence_area_type'] 

# # Apply one-hot encoding
# df_encoded = pd.get_dummies(X, columns=nominal_cols, drop_first=True)
# df_encoded.head(3)


# In[39]:


##X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[40]:


model = NearestCentroid()
model.fit(X_train, y_train)


# In[41]:


y_pred = model.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[42]:


import pickle
with open("nearest_centroid_model.pkl", "wb") as f:
    pickle.dump(model, f)


# In[43]:


# 9️⃣ Load model (example)
with open("nearest_centroid_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)


# In[44]:


# 10️⃣ Test loaded model
y_pred_loaded = loaded_model.predict(X_test)
print("♻️ Loaded model accuracy:", accuracy_score(y_test, y_pred_loaded))


# In[45]:


column_info = X.dtypes.reset_index()
column_info.columns = ['Column_Name', 'Data_Type']

# 3. Format the output row by row with comma separation
print("Column_Name,Data_Type")
for index, row in column_info.iterrows():
    # Use f-string to join the column name and data type with a comma
    output_line = f"{row['Column_Name']} : {row['Data_Type']}"
    print(output_line)


# In[ ]:




