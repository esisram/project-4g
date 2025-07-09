import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.title("📊 Telkomsel 4G Conversion Campaign Dashboard")

# Load dataset
df = pd.read_csv("dummy_telkomsel_data.csv")
df = df[df['segment'] == 'A'].copy()

# Simulate A/B group and conversion
np.random.seed(42)
df['group'] = np.random.choice(['treatment', 'control'], size=len(df))
df['converted'] = 0
df.loc[(df['group'] == 'treatment') & (np.random.rand(len(df)) < 0.28), 'converted'] = 1
df.loc[(df['group'] == 'control') & (np.random.rand(len(df)) < 0.16), 'converted'] = 1

# Encode features
df['group_encoded'] = df['group'].map({'control': 0, 'treatment': 1})
df['device_type_encoded'] = LabelEncoder().fit_transform(df['device_type'])
features = ['age', 'monthly_usage_gb', 'tenure_months', 'device_type_encoded']
X = df[features]

# Uplift models
treat_model = RandomForestClassifier(random_state=42).fit(X[df['group_encoded'] == 1], df[df['group_encoded'] == 1]['converted'])
control_model = RandomForestClassifier(random_state=42).fit(X[df['group_encoded'] == 0], df[df['group_encoded'] == 0]['converted'])
df['uplift_score'] = treat_model.predict_proba(X)[:, 1] - control_model.predict_proba(X)[:, 1]

# Sidebar filter
st.sidebar.header("Filters")
selected_group = st.sidebar.selectbox("Group", ['All', 'treatment', 'control'])
selected_segment = st.sidebar.selectbox("Segment", ['All', 'A', 'B', 'C'])

filtered = df.copy()
if selected_group != 'All':
    filtered = filtered[filtered['group'] == selected_group]
if selected_segment != 'All':
    filtered = filtered[filtered['segment'] == selected_segment]

# Data preview
st.subheader("📄 Dataset Preview")
st.dataframe(filtered.head())

# Conversion chart
st.subheader("📈 Conversion Rate by Group")
conversion = filtered.groupby('group')['converted'].mean().reset_index()
st.bar_chart(conversion.set_index('group'))

# Uplift distribution
st.subheader("📊 Uplift Score Distribution")
fig, ax = plt.subplots()
sns.histplot(filtered['uplift_score'], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# Top uplift customers
st.subheader("🏆 Top Uplift Customers")
top_uplift = filtered.sort_values('uplift_score', ascending=False).head(10)
st.dataframe(top_uplift[['user_id', 'age', 'monthly_usage_gb', 'uplift_score']])

# Bottom uplift customers
st.subheader("👎 Bottom Uplift Customers")
bottom_uplift = filtered.sort_values('uplift_score', ascending=True).head(10)
st.dataframe(bottom_uplift[['user_id', 'age', 'monthly_usage_gb', 'uplift_score']])