from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from django.conf import settings
import os
from django.http import FileResponse

def home(request):
    return render(request, 'base/home.html')

def predict(request):
    if request.method == 'POST' and 'test_file' in request.FILES:
        test_file = request.FILES['test_file']
        fs = FileSystemStorage()
        filename = fs.save(test_file.name, test_file)
        uploaded_file_url = fs.url(filename)

        col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
            "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
            "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
            "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
            "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "level"]

        train_df = pd.read_csv("C:\\Users\\Sir Vin\\OneDrive\\Desktop\\collins\\KDDTrain+.csv", names=col_names)
        test_df = pd.read_csv(test_file, names=col_names)

        test_df = test_df.drop('attack', axis=1)
        train_df = train_df.drop('attack', axis=1)

        # One-hot encode categorical features
        categorical_features = ['protocol_type', 'service', 'flag']
        encoder = OneHotEncoder(sparse=False)
        train_df_encoded = pd.DataFrame(encoder.fit_transform(train_df[categorical_features]))
        test_df_encoded = pd.DataFrame(encoder.transform(test_df[categorical_features]))
        train_df_encoded.columns = encoder.get_feature_names_out(categorical_features)
        test_df_encoded.columns = encoder.get_feature_names_out(categorical_features)
        train_df.drop(categorical_features, axis=1, inplace=True)
        test_df.drop(categorical_features, axis=1, inplace=True)
        train_df = pd.concat([train_df, train_df_encoded], axis=1)
        test_df = pd.concat([test_df, test_df_encoded], axis=1)

        # Scale the test and train data
        scaler = RobustScaler()
        train_df_scaled = scaler.fit_transform(train_df)
        test_df_scaled = scaler.transform(test_df)

        # Define the classifier and fit the model to the training data
        clf1 = DecisionTreeClassifier(max_depth=4)
        clf2 = RandomForestClassifier(max_depth=4)
        clf3 = GaussianNB()
        ensemble = VotingClassifier(estimators=[('dt', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
        ensemble.fit(train_df_scaled, train_df['level'])

        # Make predictions on the test set
        y_pred_test = ensemble.predict(test_df_scaled)

        # Convert predicted label numbers to their corresponding names
        attack_types = {
            0: "normal",
            1: "back",
            2: "buffer_overflow",
            3: "ftp_write",
            4: "guess_passwd",
            5: "imap",
            6: "ipsweep",
            7: "land",
            8: "loadmodule",
            9: "multihop",
            10: "neptune",
            11: "nmap",
            12: "perl",
            13: "phf",
            14: "pod",
            15: "portsweep",
            16: "rootkit",
            17: "satan",
            18: "smurf",
            19: "spy",
            20: "teardrop",
           21: "warezclient",
            22: "warezmaster"

        }

        y_pred_test_names = [attack_types[i] for i in y_pred_test]

        # Save the predictions to a CSV file
        import io

        csv_data = pd.DataFrame({'label': y_pred_test_names}).to_csv(index=False)
        csv_file = io.StringIO(csv_data)
        predictions_path = fs.save('predictions.csv', csv_file)
        # predictions_path = fs.save('predictions.csv', pd.DataFrame({'label': y_pred_test}).to_csv(index=False))
        predictions_url = fs.url(predictions_path)

        # Plot the predicted labels
        unique_labels, label_counts = np.unique(y_pred_test_names, return_counts=True)

        sns.set_style("whitegrid")
        plt.figure(figsize=(8,6))
        sns.barplot(x=unique_labels, y=label_counts)
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.title('Predicted Labels')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot to a file
        # plot_path = fs.save('plot.png', plt.savefig(fs.open('plot.png', 'rb'), format='png'))
        with fs.open('plot.png', 'wb') as f:
            plt.savefig(f, format='png')
        plot_path = fs.path('plot.png')  # Get the file path
        plot_url = fs.url(plot_path)

        # Render the results template
        return render(request, 'base/results.html', {'plot_url': plot_url, 'predictions_url': predictions_url})

    # Render the input form template if the request method is not POST or if 'test_file' not in request.FILES
    return render(request, 'base/form.html')

import os

def download(request):
    file_path = os.path.join(settings.MEDIA_ROOT, request.GET.get('path', ''))
    with open(file_path, 'rb') as fh:
        response = HttpResponse(fh.read(), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=' + os.path.basename(file_path)
        return response

