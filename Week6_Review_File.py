# Databricks notebook source
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# COMMAND ----------

dbutils.widgets.text("output", "","")
dbutils.widgets.get("output")
FilePath = getArgument("output")

# COMMAND ----------

dbutils.widgets.text("File_name", "","")
dbutils.widgets.get("File_name")
filename = getArgument("File_name")

# COMMAND ----------

storage_account_name = "simplitesting"
storage_account_access_key = "hePUhzTzQo2tXxEfEvpPrPI0Hl2TjySGy2CGxq5AwgQCbU4lA1Jx4QD9BzX8n5TvwpnF7PC6YKHNJUcCwPxUsQ=="

# COMMAND ----------

spark.conf.set(
"fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
storage_account_access_key)

# COMMAND ----------

file_location = "wasbs://reviewblob@simplitesting.blob.core.windows.net"+FilePath+"/"+filename
print(file_location)
#file_type = "csv"

# COMMAND ----------


df = spark.read.format(file_type).option("inferSchema", "true").load(file_location)
df.show()

# COMMAND ----------

train = df.toPandas()

# COMMAND ----------

train.head()

# COMMAND ----------

x=train.iloc[2:len(train),2:4].values

y=train.iloc[2:len(train),:1].values

# COMMAND ----------

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
reg = LinearRegression()
reg.fit(x_train,y_train)

y_pred= reg.predict(x_test)
print(y_pred)

# COMMAND ----------

file_name = 'House.pkl'
pkl_file = open(file_name, 'wb')
print("pickelfile",pkl_file)
model = pickle.dump(reg, pkl_file)

# COMMAND ----------

pkl_file = open(file_name, 'rb')
model_pkl = pickle.load(pkl_file)
y_pred = model_pkl.predict(x_test)
print("prediction",y_pred)

# COMMAND ----------


