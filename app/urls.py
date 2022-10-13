from django.contrib import admin
from django.urls import path
from app.views import hand, main, face, facemeshdetector, key, qrcode, eye, power

urlpatterns = [
    # path('',include('app.urls'))
    # path('upload/',upload,name="upload"),
    # path('data/',card_data,name="data"),
    # path('bot/',bot, name="bot"),
    # path('',bb, name="bb"),
    path('',hand, name="hand"),
    path('qrcode/',qrcode, name="qrcode"),
    path('eye/',eye, name="eye"),
    path('power/',power, name="power"),
    path('main/',main, name="main"),
    path('face/',face, name="face"),
    path('facem/',facemeshdetector, name="facem"),
    path('key/',key, name="key"),
    # path('sac/',sac, name="sac"),
    # 
]
