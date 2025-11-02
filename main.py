#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd


# In[2]:


with open("nearest_centroid_model.pkl", "rb") as f:
    model = pickle.load(f)


# In[3]:


class InputData(BaseModel):
    perc_premium_paid_by_cash_credit: float
    age_in_days: float
    Income: float
    Count_3_6_months_late: float
    Count_6_12_months_late: float
    Count_more_than_12_months_late: float
    application_underwriting_score: float
    no_of_premiums_paid: float
    premium: float


# In[4]:


# 3️⃣ Initialize FastAPI app
# ------------------------------
app = FastAPI(title="Nearest Centroid API", description="Predict target using trained model", version="1.0")


# In[5]:


@app.get("/")
def home():
    return {"message": "✅ FastAPI is running! Use /predict endpoint."}


# In[6]:


@app.post("/predict")
def predict(data: InputData):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

#     # Apply same label encoders as training
#     for col, le in label_encoders.items():
#         if col in df.columns:
#             # handle unseen categories gracefully
#             df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
#             df[col] = le.transform(df[col])

    # Make prediction
    pred = model.predict(df)[0]

    # Return prediction
    return {"prediction": int(pred)}


# In[ ]:




