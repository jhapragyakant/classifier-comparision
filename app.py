import streamlit as st
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

st.title("Classifiers Comparision")
st.write("""
## Let's Explore the different classifiers
Which one is the best
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Brest Cancer", "Wine"))
# st.write(dataset_name)

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Brest Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X,y

X, y = get_dataset(dataset_name)
st.write("shape of dataset", X.shape)
st.write("No of classes", len(np.unique(y)))

def add_parameter_ui(clf_name):
    param = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        param["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        param["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        param["max_depth"] = max_depth
        param["n_estimators"] = n_estimators
    return param

params = add_parameter_ui(classifier_name)

def get_classifier(classifier_name, params):
    if classifier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors = params["K"])
    elif classifier_name == "SVM":
        clf = SVC(C = params["C"])
    else:
        clf = RandomForestClassifier(n_estimators = params["n_estimators"], max_depth = params["max_depth"], random_state = 1234)
    return clf

clf = get_classifier(classifier_name, params)

# Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"classifier = {classifier_name}")
st.write(f"Accuracy {acc}")

# Ploting the dataset
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1, x2, c =y, alpha = 0.8, cmap = "viridis")
plt.xlabel("Principle component 1")
plt.xlabel("Principle component 2")
plt.colorbar()

st.pyplot(fig)