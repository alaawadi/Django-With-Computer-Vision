from django.db import models

# Create your models here.


class Card(models.Model):
    card_number = models.CharField(max_length=100)
    expiration_date = models.CharField(max_length=100)
    cvv = models.CharField(max_length=100)

    def __str__(self):
        return self.card_number



class Luhn_algorithm_card(models.Model):
    card_number = models.CharField(max_length=100)
    expiration_date = models.CharField(max_length=100)
    cvv = models.CharField(max_length=100)

    def __str__(self):
        return self.card_number 