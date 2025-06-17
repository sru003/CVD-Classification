import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_curve, auc, accuracy_score,
    recall_score, precision_score,
    f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier


# Title of the app
st.title("Cardio vascular analysis using machine Learning ")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)

    # Define categorical features (update this list based on your dataset)
    cat_feats = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # Label encoding function
    def label_encode_cat_features(data, cat_features):
        le = LabelEncoder()
        for feature in cat_features:
            data[feature] = le.fit_transform(data[feature])
        return data

    # Apply label encoding
    data = label_encode_cat_features(data, cat_feats)

    # Define target variable and features
    features = data.columns[:-1]
    X = data[features]
    y = data['target']  # Ensure the target column is named 'target'

    # Split data into training and validation sets
    seed = 0
    test_size = 0.25
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Define classifiers and their names
    names = [
        'Logistic Regression',
        'Nearest Neighbors',
        'Support Vectors',
        'Nu SVC',
        'Decision Tree',
        'Random Forest',
        'AdaBoost',
        'Gradient Boosting',
        'Naive Bayes',
        'Linear DA',
        'Quadratic DA',
        'Neural Net'
    ]

    classifiers = [
        LogisticRegression(solver="liblinear", random_state=seed),
        KNeighborsClassifier(2),
        SVC(probability=True, random_state=seed),
        NuSVC(probability=True, random_state=seed),
        DecisionTreeClassifier(random_state=seed),
        RandomForestClassifier(random_state=seed),
        AdaBoostClassifier(algorithm='SAMME', random_state=seed),
        GradientBoostingClassifier(random_state=seed),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        MLPClassifier(random_state=seed),
    ]

    # Dropdown for selecting classifier
    selected_classifier = st.selectbox("Select a Classifier", names)
    
    # Option to display overall ROC
    show_overall_roc = st.checkbox("Show Overall ROC Curve")

    # Evaluate the selected classifier
    if st.button("Evaluate"):
        clf = classifiers[names.index(selected_classifier)]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        # Display metrics
        st.write(f'Classifier: {selected_classifier}')
        st.write(f'Accuracy: {accuracy_score(y_val, y_pred)}')
        st.write(f'ROC AUC: {roc_auc_score(y_val, y_pred)}')
        st.write(f'Recall: {recall_score(y_val, y_pred)}')
        st.write(f'Precision: {precision_score(y_val, y_pred)}')
        st.write(f'F1 Score: {f1_score(y_val, y_pred)}')

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {selected_classifier}')
        st.pyplot(plt)

        # ROC Curve for the selected classifier
        if hasattr(clf, "predict_proba"):
            fpr, tpr, _ = roc_curve(y_val, clf.predict_proba(X_val)[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(10, 7))
            plt.plot(fpr, tpr, marker='.', label=f'{selected_classifier} (AUC = {roc_auc:.2f})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.grid(True)
            st.pyplot(plt)

    # Plot Overall ROC Curve for all classifiers
    if show_overall_roc and st.button("Show Overall ROC"):
        plt.figure(figsize=(12, 8))
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            y_pred_prob = clf.predict_proba(X_val)[:, 1]  # Probability estimates for the positive class
            fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Overall Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        st.pyplot(plt)

# Run the app
if __name__ == "__main__":
    st.write("Run this script using Streamlit.")


##confusion matrix

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# Define seed for reproducibility
seed = 0

# Define test size
test_size = 0.25

# Define classifiers and their names
names = [
    'Logistic Regression',
    'Nearest Neighbors',
    'Support Vectors',
    'Nu SVC',
    'Decision Tree',
    'Random Forest',
    'AdaBoost',
    'Gradient Boosting',
    'Naive Bayes',
    'Linear DA',
    'Quadratic DA',
    'Neural Net'
]

classifiers = [
    LogisticRegression(solver="liblinear", random_state=seed),
    KNeighborsClassifier(2),
    SVC(probability=True, random_state=seed),
    NuSVC(probability=True, random_state=seed),
    DecisionTreeClassifier(random_state=seed),
    RandomForestClassifier(random_state=seed),
    AdaBoostClassifier(algorithm='SAMME', random_state=seed),
    GradientBoostingClassifier(random_state=seed),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    MLPClassifier(random_state=seed),
]

# Load dataset
path = 'heart.csv'
data = pd.read_csv(path)

# Define categorical features
cat_feats = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Label encoding function
def label_encode_cat_features(data, cat_features):
    le = LabelEncoder()
    for feature in cat_features:
        data[feature] = le.fit_transform(data[feature])
    return data

# Apply label encoding
data = label_encode_cat_features(data, cat_feats)

# Define target variable and features
features = data.columns[:-1]
X = data[features]
y = data['target']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=seed)

# Function to plot confusion matrices
def plot_conf_matrix(names, classifiers):
    nrows = 4
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))
    axes = axes.flatten()
    for name, clf, ax in zip(names, classifiers, axes):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        cm = confusion_matrix(y_val, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Disease'])
        disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d', colorbar=False)
        
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    for i in range(len(names), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    st.pyplot(fig)  # Display the plot in Streamlit

# Streamlit interface
st.title("Classifier Confusion Matrices")

if st.button("Display Confusion Matrices"):
    plot_conf_matrix(names, classifiers)

import streamlit as st
import pandas as pd

# Load your score summary HTML file
#html_path = 'score_summary.html'

# Streamlit app
st.title('Machine Learning Classifier Score Summary')

# Button to display the score summary
if st.button('Show Score Summary'):
    # Read and display the HTML file
    with open(score_summary.html, 'r') as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=600)  # Adjust height as needed

# Additional code for your app can go here
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import uniform

# Load your dataset
data = pd.read_csv('heart.csv')

# Define your features and target
features = data.columns[:-1]
X = data[features]
y = data['target']

# Streamlit app
st.title('Hyperparameter Tuning for Logistic Regression')

# Button to find best hyperparameters
if st.button('Find Best Hyperparameters'):
    # Split the data
    seed = 0
    test_size = 0.25
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Initialize Logistic Regression model
    lr = LogisticRegression(tol=1e-4, max_iter=1000, random_state=seed)

    # Define the hyperparameter space
    space = dict(
        C=uniform(loc=0, scale=5),
        penalty=['l2', 'l1'],
        solver=['liblinear']
    )

    # Define RepeatedStratifiedKFold cross-validation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)

    # Initialize RandomizedSearchCV
    search = RandomizedSearchCV(
        lr,
        space,
        n_iter=10,
        random_state=seed,
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )

    # Fit the RandomizedSearchCV
    rand_search = search.fit(X_train, y_train)

    # Display best hyperparameters
    best_params = rand_search.best_params_
    st.write('Best Hyperparameters:', best_params)

    # Extract best parameters and fit the model
    best_lr = LogisticRegression(**best_params)
    best_lr.fit(X_train, y_train)

    # Predictions and classification report
    y_pred = best_lr.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True)

    # Convert the classification report to a DataFrame
    report_df = pd.DataFrame(report).transpose()
    st.write('Classification Report:')
    st.dataframe(report_df)  # Display the report as a table

    # Plot confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Heart Disease', 'Heart Disease'],
                yticklabels=['No Heart Disease', 'Heart Disease'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(fig)  # Display the confusion matrix in Streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Load your dataset
data = pd.read_csv('heart.csv')

# Define your features and target
features = data.columns[:-1]
X = data[features]
y = data['target']

# Split the data
seed = 0
test_size = 0.25
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=seed)

# Define names and classifiers for boosting methods
names_boost = [
    'CatBoost',
    'XGBoost',
    'LightGBM'
]

classifiers = [
    CatBoostClassifier(random_state=seed, verbose=0),
    XGBClassifier(objective='binary:logistic', random_state=seed),
    LGBMClassifier(random_state=seed)
]

def score_summary(names, classifiers):
    cols = ["Classifier", "Accuracy", "ROC_AUC", "Recall", "Precision", "F1"]
    data_table = pd.DataFrame(columns=cols)
    
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        accuracy = accuracy_score(y_val, pred)
        pred_proba = clf.predict_proba(X_val)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_val, pred_proba)
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(y_val, pred)
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) != 0 else 0
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) != 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) != 0 else 0

        df = pd.DataFrame([[name, accuracy * 100, roc_auc, recall, precision, f1]], columns=cols)
        data_table = pd.concat([data_table, df], ignore_index=True)
    
    return np.round(data_table.reset_index(drop=True), 2)

def display_ml_results():
    df_scores = score_summary(names_boost, classifiers)
    df_scores_sorted = df_scores.sort_values(by='Accuracy', ascending=False)

    # Display the DataFrame with styled background gradient and bars
    styled_df = df_scores_sorted.style.background_gradient(cmap='coolwarm')\
        .bar(subset=["ROC_AUC"], color='#6495ED')\
        .bar(subset=["Recall"], color='#ff355d')\
        .bar(subset=["Precision"], color='lightseagreen')\
        .bar(subset=["F1"], color='gold')

    st.write('### Model Performance Summary:')
    st.dataframe(styled_df)

    # Plot confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for i, (clf, ax) in enumerate(zip(classifiers, axes)):
        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        cm = confusion_matrix(y_val, pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
        ax.set_title(names_boost[i])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
    plt.tight_layout()
    st.pyplot(fig)  # Display confusion matrix in Streamlit

    # Hyperparameter tuning for LightGBM
    rs_params = {
        'num_leaves': [20, 100],
        'max_depth': [5, 15],
        'min_data_in_leaf': [80, 120], 
    }

    rs_cv = GridSearchCV(
        estimator=LGBMClassifier(random_state=seed, verbose=-1), 
        param_grid=rs_params, 
        cv=5
    )

    rs_cv.fit(X_train, y_train)
    params = rs_cv.best_params_
    st.write("Best hyperparameters for LightGBM:", params)

# Streamlit app layout
st.title('Machine Learning Implementation')

# Button to display machine learning implementation results
if st.button('Run Machine Learning Models'):
    display_ml_results()

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMClassifier

# Load your dataset
data = pd.read_csv('heart.csv')
features = data.columns[:-1]
X = data[features]
y = data['target']

# Split the data
seed = 0
test_size = 0.25
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=seed)

# Grid Search for LightGBM hyperparameters
rs_params = {
    'num_leaves': [20, 100],
    'max_depth': [5, 15],
    'min_data_in_leaf': [80, 120], 
}

rs_cv = GridSearchCV(
    estimator=LGBMClassifier(random_state=seed, verbose=-1), 
    param_grid=rs_params, 
    cv=5
)

rs_cv.fit(X_train, y_train)
params = rs_cv.best_params_

# Streamlit app
st.title("Machine Learning Implementation")

# Button to display best hyperparameters
if st.button("Show Best Hyperparameters for LightGBM"):
    # Create a DataFrame for the hyperparameters
    best_params_df = pd.DataFrame(list(params.items()), columns=['Hyperparameter', 'Value'])
    st.write(best_params_df)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report, ConfusionMatrixDisplay
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Load your dataset
data = pd.read_csv('heart.csv')

# Define your features and target
features = data.columns[:-1]
X = data[features]
y = data['target']

# Split the data
seed = 0
test_size = 0.25
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=seed)

# Define names and classifiers for boosting methods
names_boost = [
    'CatBoost',
    'XGBoost',
    'LightGBM'
]

classifiers = [
    CatBoostClassifier(random_state=seed, verbose=0),
    XGBClassifier(objective='binary:logistic', random_state=seed),
    LGBMClassifier(random_state=seed)
]

def score_summary(names, classifiers):
    cols = ["Classifier", "Accuracy", "ROC_AUC", "Recall", "Precision", "F1"]
    data_table = pd.DataFrame(columns=cols)
    
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        accuracy = accuracy_score(y_val, pred)
        pred_proba = clf.predict_proba(X_val)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_val, pred_proba)
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(y_val, pred)
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) != 0 else 0
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) != 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) != 0 else 0

        df = pd.DataFrame([[name, accuracy * 100, roc_auc, recall, precision, f1]], columns=cols)
        data_table = pd.concat([data_table, df], ignore_index=True)
    
    return np.round(data_table.reset_index(drop=True), 2)

# Get the scores and sort them by 'Accuracy'
df_scores = score_summary(names_boost, classifiers)
df_scores_sorted = df_scores.sort_values(by='Accuracy', ascending=False)

# Streamlit app
st.title("Machine Learning Implementation")

# Button to display all confusion matrices
if st.button("Show All Confusion Matrices"):
    fig, axes = plt.subplots(1, len(classifiers), figsize=(15, 5))
    for i, (clf, name) in enumerate(zip(classifiers, names_boost)):
        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        cm = confusion_matrix(y_val, pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot(ax=axes[i], cmap=plt.cm.Blues, colorbar=False)
        axes[i].set_title(name)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    plt.tight_layout()
    st.pyplot(fig)

# Button to display final confusion matrix and scores for LightGBM
if st.button("Show Final Confusion Matrix and Scores for LightGBM"):
    # Fit LightGBM with best parameters
    params = {'num_leaves': 20, 'max_depth': 5, 'min_data_in_leaf': 80}  # Replace with your best params
    lgbm = LGBMClassifier(**params, verbose=-1)
    lgbm.fit(X_train, y_train)
    
    # Predictions and confusion matrix
    y_pred = lgbm.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    
    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lgbm.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    ax.set_title('Final Confusion Matrix for LightGBM')
    st.pyplot(fig)

    # Calculate metrics
    report = classification_report(y_val, y_pred, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose()

    # Display metrics table
    st.write("Classification Metrics for LightGBM:")
    st.table(metrics_df[['precision', 'recall', 'f1-score']].round(2))
