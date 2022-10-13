



from rest_framework import serializers


from .models import Luhn_algorithm_card,Card






class Luhn_algorithm_card_serializer(serializers.ModelSerializer):
  class Meta:
        model = Luhn_algorithm_card
        fields = '__all__'


class Card_serializer(serializers.ModelSerializer):
  class Meta:
        model = Card
        fields = '__all__'
