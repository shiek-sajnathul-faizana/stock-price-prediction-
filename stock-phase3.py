#importing libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

#reading the file
df = pd.read_csv(r"C:\Users\DELL\Downloads\MSFT.csv")
print(df)

#understanding the dataset
print(df.head())
print(df.describe())
print(df.isnull().sum())

#removing duplicates
df = df.drop_duplicates()
print(df)
#-- to check if any duplicates rows are removed
print(df.isnull().sum())

#DATA TRANSFORMATION - z-score standardization
scaler = MinMaxScaler()
df['Normalized_Open']=scaler.fit_transform(df[['Open']])
label_encoder = LabelEncoder()
df['Encoded_Date'] = label_encoder.fit_transform(df['Date'])
print(df)

#feature engineering - a new column(feature) High minus Low is obtained by subtracting low from high
df['High_Minus_Low'] = df['High'] - df['Low']
print(df)

#handling outliers
thresholds = {'Date': ("1986-03-19","2020-01-02")}
for col, (lower,upper) in thresholds.items():
    df=df[(df[col] >= lower ) & (df[col] <= upper)]
print(df)

#data splitting
#-- Splitting the dataset
X = df.drop('Close', axis=1)
y = df['Close']
#--Split into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
#--Display the split datasets
print("Training set:")
print(X_train)
print("Validation set:")
print(X_val)
print("Testing set:")
print(X_test)
