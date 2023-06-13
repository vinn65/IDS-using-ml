from django import forms

class TestFileForm(forms.Form):
    test_file = forms.FileField()