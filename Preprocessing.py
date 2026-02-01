#!/usr/bin/env python
# coding: utf-8

# # <div style="font-family: 'Playfair Display', serif; font-weight:bold; letter-spacing: 0px; color:#FAF0EF; font-size:150%; text-align:center; padding:10px; background: #1db5baff; border-radius: 10px; box-shadow: 10px 10px 5px #7abcafff;">🌟 Data Preprocessing  <br></div>

# In[18]:


from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[1]:


import kagglehub
import pandas as pd
import os

# Download dataset to cache (not project folder)
path = kagglehub.dataset_download("justinas/startup-investments")

# List files to know the exact CSV name
os.listdir(path)


# In[2]:


import pandas as pd
import os

objects = pd.read_csv(os.path.join(path, "objects.csv"))
funding = pd.read_csv(os.path.join(path, "funding_rounds.csv"))
investments = pd.read_csv(os.path.join(path, "investments.csv"))
ipos = pd.read_csv(os.path.join(path, "ipos.csv"))
acq = pd.read_csv(os.path.join(path, "acquisitions.csv"))
offices = pd.read_csv(os.path.join(path, "offices.csv"))


# In[3]:


companies = objects[objects["entity_type"] == "Company"].copy()


# In[4]:


funding_agg = funding.groupby("object_id").agg(
    total_funding_usd=("raised_amount_usd", "sum"),
    nb_funding_rounds=("funding_round_id", "count")
).reset_index()

companies = companies.merge(
    funding_agg,
    left_on="id",       # ← id من companies
    right_on="object_id",
    how="left"
)

# NaN → 0
companies["total_funding_usd"] = companies["total_funding_usd"].fillna(0)
companies["nb_funding_rounds"] = companies["nb_funding_rounds"].fillna(0)


# In[5]:


investor_agg = investments.groupby("funded_object_id").agg(
    nb_investors=("investor_object_id", "nunique")
).reset_index()

companies = companies.merge(
    investor_agg,
    left_on="id",               # id من companies
    right_on="funded_object_id",  # العمود الصحيح من investments
    how="left"
)


companies["nb_investors"] = companies["nb_investors"].fillna(0)


# In[6]:


ipos["ipo"] = 1   
ipos = ipos.rename(columns={"object_id": "company_object_id"})

companies = companies.merge(
    ipos[["company_object_id", "ipo"]],
    left_on="id",
    right_on="company_object_id",
    how="left"
)

companies["ipo"] = companies["ipo"].fillna(0)


# In[7]:


acq["acquired"] = 1
acq = acq.rename(columns={"acquired_object_id": "company_object_id"})

companies = companies.merge(
    acq[["company_object_id", "acquired"]],
    left_on="id",
    right_on="company_object_id",
    how="left"
)

companies["acquired"] = companies["acquired"].fillna(0)


# In[8]:


office_agg = offices.groupby("object_id").agg(
    nb_offices=("office_id", "count"),
    country=("country_code", "first")
).reset_index().rename(columns={"object_id": "company_object_id"})

companies = companies.merge(
    office_agg,
    left_on="id",
    right_on="company_object_id",
    how="left"
)

companies["nb_offices"] = companies["nb_offices"].fillna(0)
companies["country"] = companies["country"].fillna("Unknown")


# In[9]:


import pandas as pd

def explore_dataframe(df):
   """
   Simple function to explore a DataFrame
   Shows dtypes, info, and describe
   """
   
   print("===== Data Types =====")
   print(df.dtypes)  # print the type of each column
   
   print("\n===== Info =====")
   print(df.info())  # print general info of the DataFrame
   
   print("\n===== Describe =====")
   print(df.describe())  # print statistics of numeric columns


# In[10]:


explore_dataframe(companies)


# In[ ]:


#  Remove duplicates
companies = companies.drop_duplicates()


# # 2️⃣ Fill missing values

# In[12]:


import pandas as pd

def get_numeric_columns(df):
    """
    Get numeric columns from a DataFrame
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()  # select numeric columns
    return numeric_cols


# In[13]:


numeric_cols = get_numeric_columns(companies)
print("Numeric columns:", numeric_cols)


# In[14]:


def get_categorical_columns(df):
    """
    Get categorical (non-numeric) columns from a DataFrame
    """
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()  # select non-numeric columns
    return categorical_cols


# In[15]:


categorical_cols = get_categorical_columns(companies)
print("Categorical columns:", categorical_cols)


# In[16]:


#   Handle missing values
# ===============================
# Numeric columns → fill with 0
companies[numeric_cols] = companies[numeric_cols].fillna(0)

# Categorical columns → fill with "Unknown"
companies[categorical_cols] = companies[categorical_cols].fillna("Unknown")


# # 4️⃣ Encode categorical features

# In[19]:


# 4️⃣ Encode categorical features
# ===============================
for col in categorical_cols:
    le = LabelEncoder()
    companies[col + "_encoded"] = le.fit_transform(companies[col])



# In[20]:


# ===============================
# 5️⃣ Scale numeric features
# ===============================
scaler = StandardScaler()
companies[numeric_cols] = scaler.fit_transform(companies[numeric_cols])



# In[21]:


# ===============================
# 6️⃣ Select final features for model
# ===============================
feature_cols = numeric_cols + [col + "_encoded" for col in categorical_cols]
X = companies[feature_cols]


# In[22]:


# Example target → we can predict total_funding_usd
y = companies["total_funding_usd"]

# ===============================
# 7️⃣ Check processed data
# ===============================
print(X.head())
print(y.head())


# # Exploratory Data Analysis

# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 1️⃣ Numeric columns distribution
# ===============================
numeric_cols = get_numeric_columns(companies)

plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols):
    plt.subplot(2, 3, i+1)
    sns.histplot(companies[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()





# In[24]:


# ===============================
# 2️⃣ Correlation matrix
# ===============================
corr = companies[numeric_cols].corr()  # correlation between numeric features
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# In[25]:


# ===============================
# 3️⃣ Bar plots for categorical features
# ===============================
categorical_cols = get_categorical_columns(companies)

# Show top 10 values for each categorical feature
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    top_values = companies[col].value_counts().head(10)
    sns.barplot(x=top_values.values, y=top_values.index)
    plt.title(f"Top 10 values in {col}")
    plt.show()


# In[26]:


# ===============================
# 4️⃣ Feature importance insight (simple correlation with target)
# ===============================
target = "total_funding_usd"
feature_corr = companies[numeric_cols].corrwith(companies[target]).sort_values(ascending=False)
print("Correlation of numeric features with target:")
print(feature_corr)


# # Select useful features only

# In[3]:


# Select useful features only
selected_features = [
    "nb_funding_rounds",
    "nb_investors",
    "nb_offices",
    "ipo",
    "acquired",
    "milestones",
    "relationships",
    "funding_rounds"
]

# Create X and y
X = companies[selected_features]
y = companies["total_funding_usd"]

# Check data
print(X.head())
print(y.head())


# In[4]:


selected_features = [
    "nb_funding_rounds",
    "nb_investors",
    "nb_offices",
    "ipo",
    "acquired",
    "milestones",
    "relationships",
    "funding_rounds"
]


# In[5]:


import json
# Save to JSON
with open("feature_cols.json", "w") as f:
    json.dump(selected_features, f)

print("Feature columns saved to feature_cols.json")


# # Model Training

# In[28]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[41]:


y = np.log1p(companies["funding_total_usd"])


# In[42]:


# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=400
)


# ## Regression Models (classical ML)

# In[43]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

def train_and_compare_regression_models(X_train, X_test, y_train, y_test):
    # Create models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    results = []

    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)

        # Predict test data
        y_pred = model.predict(X_test)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Save results
        results.append({
            "Model": name,
            "RMSE": rmse,
            "R2": r2
        })

    return pd.DataFrame(results)


# In[44]:


results = train_and_compare_regression_models(
    X_train, X_test, y_train, y_test
)

results.sort_values("RMSE")



# 

# In[38]:


results = train_and_compare_regression_models(
    X_train, X_test, y_train, y_test
)

results.sort_values("RMSE")


# Ridge Regression

# ## Deep Learning

# In[47]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# In[46]:


#   Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[48]:


#  Define the MLP model
model = Sequential([
    Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # regression output
])


# In[49]:


#   Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#   Train the model with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


# In[50]:


#   Evaluate the model
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)



# In[51]:


print(f"MLP RMSE: {rmse:.4f}")
print(f"MLP R2: {r2:.4f}")


# # <div style="font-family: 'Playfair Display', serif; font-weight:bold; letter-spacing: 0px; color:#FAF0EF; font-size:150%; text-align:center; padding:10px; background: rgb(70, 162, 198); border-radius: 10px; box-shadow: 10px 10px 5px rgb(82, 130, 192);">🌟 pickle <br></div>

# In[52]:


# Save the entire model (architecture + weights + optimizer state)
model.save("mlp_funding_model.h5")
print("MLP model saved as mlp_funding_model.h5")


# In[55]:


# Save weights with correct extension
model.save_weights("mlp_model.weights.h5")
print("Weights saved successfully!")


# In[57]:


import pickle
import json

# Save architecture
model_json = model.to_json()
with open("mlp_model_architecture.json", "w") as json_file:
    json_file.write(model_json)

# Save weights with correct name
model.save_weights("mlp_model.weights.h5")

# Save paths with Pickle
pickle_data = {
    "architecture_file": "mlp_model_architecture.json",
    "weights_file": "mlp_model.weights.h5"
}

with open("mlp_model_pickle.pkl", "wb") as f:
    pickle.dump(pickle_data, f)

print("MLP model saved via Pickle workaround!")

