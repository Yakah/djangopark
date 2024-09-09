from django.contrib import admin
from .models import Video, parkingSlot, vehicle, Item
from embed_video.admin import AdminVideoMixin

# Register your models here.
admin.site.register(Video)
admin.site.register(parkingSlot)
admin.site.register(vehicle)


class MyModelAdmin(AdminVideoMixin, admin.ModelAdmin):
    pass

admin.site.register(Item, MyModelAdmin)