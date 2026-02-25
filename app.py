import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
 
st.title("Naive Bayes Classifier App")
 
# Upload dataset
file = st.file_uploader("Upload CSV Dataset", type=["csv"])
 
if file:
    df = pd.read_csv(file)
    st.write("Dataset Preview")
    st.dataframe(df)
 
    # Select target column
    target = st.selectbox("Select Target Column", df.columns)
 
    X = df.drop(target, axis=1)
    y = df[target]
 
    # Encode categorical data
    X = pd.get_dummies(X)
   
    # Train test split
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
 
    # Model selection
    model_name = st.selectbox(
        "Select Naive Bayes Model",
        ["Gaussian", "Multinomial", "Bernoulli"]
    )
 
    if model_name == "Gaussian":
        model = GaussianNB()
    elif model_name == "Multinomial":
        model = MultinomialNB()
    else:
        model = BernoulliNB()
 
    if st.button("Train Model"):
 
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
 
        model.fit(X_train, y_train)
 
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
 
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
 
        st.subheader("Results")
 
        st.write("Training Accuracy:", round(train_acc, 3))
        st.write("Testing Accuracy:", round(test_acc, 3))
 
        # Confusion matrix
        cm = confusion_matrix(y_test, test_pred)
 
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
 