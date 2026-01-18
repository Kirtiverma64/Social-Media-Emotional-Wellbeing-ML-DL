import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.set_page_config(page_title="Emotion Prediction", layout="centered")
st.title("📱 Social Media Usage & Emotional Well-Being")
st.write("Machine Learning & Deep Learning Based Prediction")

df = pd.read_csv("train.csv")
df.rename(columns={"Daily_Usage_Time (minutes)": "Daily_Usage_Time"}, inplace=True)
df.drop(columns=["User_ID"], inplace=True)

X = pd.get_dummies(df.drop(columns=["Dominant_Emotion"]), drop_first=True)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Dominant_Emotion"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- ML MODEL ----------------
ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
ml_model.fit(X_train, y_train)

# ---------------- DL MODEL ----------------
dl_model = Sequential()
dl_model.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],)))
dl_model.add(Dense(32, activation="relu"))
dl_model.add(Dense(len(np.unique(y)), activation="softmax"))

dl_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

dl_model.fit(
    X_train.values.astype("float32"),
    y_train,
    epochs=15,
    batch_size=16,
    verbose=0
)

# ---------------- UI ----------------
st.header("📝 Enter User Details")

age = st.number_input("Age", 10, 80)
gender = st.selectbox("Gender", df["Gender"].unique())
platform = st.selectbox("Platform", df["Platform"].unique())
usage = st.number_input("Daily Usage Time (minutes)", 0)
posts = st.number_input("Posts Per Day", 0)
likes = st.number_input("Likes Received Per Day", 0)
comments = st.number_input("Comments Received Per Day", 0)
messages = st.number_input("Messages Sent Per Day", 0)

user_df = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Platform": platform,
    "Daily_Usage_Time": usage,
    "Posts_Per_Day": posts,
    "Likes_Received_Per_Day": likes,
    "Comments_Received_Per_Day": comments,
    "Messages_Sent_Per_Day": messages
}])

user_df = pd.get_dummies(user_df)
user_df = user_df.reindex(columns=X.columns, fill_value=0)

# ---------------- PREDICTION ----------------
if st.button("Predict Emotion"):
    ml_pred = ml_model.predict(user_df)[0]


    dl_pred = np.argmax(
        dl_model.predict(user_df.values.astype("float32")),
        axis=1
    )[0]

    st.success(f"ML Prediction: {label_encoder.inverse_transform([ml_pred])[0]}")
    st.success(f"DL Prediction: {label_encoder.inverse_transform([dl_pred])[0]}")
