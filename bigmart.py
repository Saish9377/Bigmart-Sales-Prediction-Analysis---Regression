import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Bigmart Sales Prediction", layout="wide")
st.title("Bigmart Sales Prediction App")

# Upload CSV file
data_file = st.file_uploader("Upload Bigmart Train CSV File", type=["csv"])

if data_file is not None:
    df = pd.read_csv(data_file)

    st.subheader("Data Preview")
    st.write(df.head())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Identify categorical columns
    cat_col = [col for col in df.columns if df[col].dtype == 'object']
    for col in ['Item_Identifier', 'Outlet_Identifier']:
        if col in cat_col:
            cat_col.remove(col)

    st.subheader("Categorical Column Distributions")
    for col in cat_col:
        st.write(f"**{col}**")
        st.write(df[col].value_counts())

    # Fill missing Item_Weight safely
    item_weight_mean = df.pivot_table(values="Item_Weight", index='Item_Identifier')
    def fill_weight(identifier):
        try:
            return item_weight_mean.loc[identifier].values[0]
        except KeyError:
            return df['Item_Weight'].mean()
    df['Item_Weight'] = df.apply(
        lambda row: fill_weight(row['Item_Identifier']) if pd.isnull(row['Item_Weight']) else row['Item_Weight'],
        axis=1
    )

    # Fill Outlet_Size missing values with mode
    mode_outlet_size = df.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=lambda x: x.mode()[0])
    def fill_outlet_size(row):
        if pd.isnull(row['Outlet_Size']):
            try:
                return mode_outlet_size[row['Outlet_Type']]
            except KeyError:
                return df['Outlet_Size'].mode()[0]
        else:
            return row['Outlet_Size']
    df['Outlet_Size'] = df.apply(fill_outlet_size, axis=1)

    # Replace zeros in Item_Visibility
    df['Item_Visibility'].replace(0, df['Item_Visibility'].mean(), inplace=True)

    # Feature Engineering
    df['Item_Type_Combined'] = df['Item_Identifier'].apply(lambda x: x[:2]).map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
    df['Outlet_Years'] = 2025 - df['Outlet_Establishment_Year']

    # Label Encoding
    label_cols = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined']
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Drop unneeded columns
    df.drop(['Item_Type','Outlet_Establishment_Year','Item_Identifier','Outlet_Identifier'], axis=1, inplace=True)

    # Model Training
    st.subheader("Train Model and Predict")
    X = df.drop('Item_Outlet_Sales', axis=1)
    y = df['Item_Outlet_Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write(f"**R2 Score:** {r2_score(y_test, y_pred):.2f}")
    st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

    st.subheader("Prediction vs Actual")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.3)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted Sales")
    st.pyplot(fig)

else:
    st.info("👆 Please upload a CSV file to get started.")
