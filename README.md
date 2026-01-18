# Social Media Usage and Emotional Well-Being  
   (Machine Learning & Deep Learning Project)

# Introduction
Social media has become an essential part of modern life, influencing communication, entertainment, and emotional expression. While it offers many benefits, excessive or unhealthy social media usage can negatively affect emotional well-being, leading to stress, anxiety, boredom, or mood changes.

This project analyzes social media usage patterns and predicts the **dominant emotional state** of users using **Machine Learning (ML)** and **Deep Learning (DL)** models. An interactive web application is developed to allow users to input their social media usage details and receive real-time predictions.

# Objectives
- To study the impact of social media usage on emotional well-being  
- To predict dominant emotional states using ML and DL techniques  
- To compare Machine Learning and Deep Learning predictions  
- To build a user-friendly interface for real-time emotion prediction  

# Dataset Description
The project uses a structured dataset containing the following attributes:

- Age  
- Gender  
- Social media platform  
- Daily usage time (in minutes)  
- Posts per day  
- Likes received per day  
- Comments received per day  
- Messages sent per day  

**Target Variable:**  
- `Dominant_Emotion` (Happiness, Anxiety, Neutral, Boredom, Anger, etc.)

# Technologies Used
- **Programming Language:** Python  
- **User Interface:** Streamlit  
- **Machine Learning Model:** Random Forest Classifier  
- **Deep Learning Model:** Artificial Neural Network (ANN)  
- **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow, Keras  

# Methodology
1. Data loading and preprocessing  
2. Handling categorical data using one-hot encoding  
3. Splitting the dataset into training and testing sets  
4. Training a Machine Learning model (Random Forest)  
5. Training a Deep Learning model (ANN)  
6. Developing an interactive Streamlit web application  
7. Predicting emotional well-being based on user input  

#User Interface
The Streamlit-based interface allows users to:
- Enter social media usage details  
- Predict emotional state using both ML and DL models  
- View results instantly in a simple and interactive format  

# How to Run the Project
Follow these steps to run the project locally:

1. Open the project folder in VS Code  
2. Install required dependencies: pip install -r requirements.txt
3.Run the application: streamlit run ml_dl_app.py
The application will open automatically in your browser.


