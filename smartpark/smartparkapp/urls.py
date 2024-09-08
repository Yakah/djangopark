from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from smartparkapp import views

urlpatterns = [
    path('sysadmin/', admin.site.urls),
    path('',views.login, name='login'),
    path('login/',views.login, name='login'),
    path('monitoring/',views.monitoring, name='monitoring'),
    path('dashboard/',views.dashboard, name='dashboard'),
    path('slotsTable/',views.slotsTable, name='slotsTable'),
    path('videostream/',views.videoStream, name='videoStream'),
    path('parkingslot/',views.parkingslot, name='parkingslot'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)