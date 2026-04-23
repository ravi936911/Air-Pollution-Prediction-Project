import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("C:/Users/Ravi verma/Downloads/airPollution.csv") 

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# find missing values
print(df.isnull().sum())

# handle missing values with mean
df['pollutant_avg'].fillna(df['pollutant_avg'].mean(), inplace=True)
df['pollutant_max'].fillna(df['pollutant_max'].mean(), inplace=True)
df['pollutant_min'].fillna(df['pollutant_min'].mean(), inplace=True)

print("\nAfter Cleaning:")
print(df.isnull().sum())


# Normalization
num_cols = ['pollutant_min', 'pollutant_max', 'pollutant_avg']
scaler = MinMaxScaler()

df[num_cols] = scaler.fit_transform(df[num_cols])

print("Normalized Data:")
print(df[num_cols].head())

# Scatter plot: pollution_min vs pollution+avg  
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df['pollutant_min'],
    y=df['pollutant_avg']
)

plt.title("Scatter Plot: Min vs Avg Pollution")
plt.xlabel("Normalized pollutant_min")
plt.ylabel("Normalized pollutant_avg")
plt.grid(True)
plt.show()


# Scatter plot: pollution_max vs pollution+avg  
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df['pollutant_max'],
    y=df['pollutant_avg']
)

plt.title("Scatter Plot: Max vs Avg Pollution")
plt.xlabel("Normalized pollutant_max")
plt.ylabel("Normalized pollutant_avg")
plt.grid(True)
plt.show()


df["last_update"] = pd.to_datetime(df["last_update"])
# Encode categorical columns
le = LabelEncoder()
for col in ['country', 'state', 'city', 'station', 'pollutant_id']:
    df[col] = le.fit_transform(df[col])
print(df[['country','state','city','station','pollutant_id']].head())


# Linear Regression Model  
X = df.drop("pollutant_avg", axis=1)
y = df["pollutant_avg"]

# # Convert datetime to numeric
X["last_update"] = X["last_update"].astype("int64") 


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#  STATISTICAL ANALYSIS
print("Mean:\n", df.mean(numeric_only=True))
print("Median:\n", df.median(numeric_only=True))
print("Variance:\n", df.var(numeric_only=True))
print("Correlation:\n", df.corr(numeric_only=True))


# VISUALIZATION

# Histogram
df.hist(figsize=(12,10))
plt.suptitle("Histogram")
plt.show()

# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Boxplot
plt.figure(figsize=(12,6))
sns.boxplot(data=df[['pollutant_min','pollutant_max','pollutant_avg']])
plt.title("Boxplot")
plt.show()


# Top 10 Cities Avg Pollution
top_city = df.groupby("city")["pollutant_avg"].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
top_city.plot(kind='bar')
plt.title("Top 10 Polluted Cities")
plt.ylabel("Average Pollution")
plt.show()



# MACHINE LEARNING (NUMERIC PREDICTION)

model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Mean absolute error (MAE) :", mean_absolute_error(y_test, y_pred))
print("Mean square error (MSE)  :", mean_squared_error(y_test, y_pred))
print(" Root Mean square error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R-Square Score:", r2_score(y_test, y_pred))

# Actual vs Predicted
result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(result.head(10))

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Pollution")
plt.show()

