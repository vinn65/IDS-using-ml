from django.db import models

class TestFile(models.Model):
    file = models.FileField(upload_to='tests/')