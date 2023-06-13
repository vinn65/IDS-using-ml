# from django.shortcuts import render
# import csv
# import pandas as pd
# from django.http import HttpResponse
# from django.shortcuts import render
# import pickle
# from .forms import UploadForm

# # Create your views here.
# def home(request):
#     return render(request,'base/home.html')
# def Detector(request):
#     return render(request,'base/Detector.html')


# def load_model():
#     # Load the trained model from the classifier.pkl file

#     with open('web/base/classifier.pkl', 'rb') as file:
#         model = pickle.load(file)
#     return model

# def predict_labels(df, model):
#     # Perform the necessary preprocessing on the input data
#     # and use the loaded model to make predictions
#     # Replace this code with your actual prediction logic
#     predictions = model.predict(df)
#     return predictions

# def process_csv(request):
#     if request.method == 'POST':
#         form = UploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             csv_file = request.FILES['csv_file']
#             df = pd.read_csv(csv_file)

#             # Load the trained model
#             model = load_model()

#             # Make predictions using the loaded model
#             predictions = predict_labels(df, model)

#             # Generate the output CSV
#             output_df = df.head(5)
#             output_df['predicted_attack'] = predictions

#             # Create a response with the output CSV
#             response = HttpResponse(content_type='text/csv')
#             response['Content-Disposition'] = 'attachment; filename="output.csv"'

#             output_df.to_csv(path_or_buf=response, index=False, quoting=csv.QUOTE_NONNUMERIC)
#             return response
#     else:
#         form = UploadForm()
    
#     return render(request, 'base/process.html', {'form': form})
