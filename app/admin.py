from calendar import c
from django.contrib import admin
from .models import Card,Luhn_algorithm_card
from import_export.admin import ImportExportModelAdmin


@admin.register(Card)
class anyname(ImportExportModelAdmin):
     pass
admin.register(Card)
admin.register(Card)

admin.site.register(Luhn_algorithm_card)

# admin.site.unregister(groups)
# admin.site.unregister(User)