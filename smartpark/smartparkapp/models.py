from django.db import models
from django import forms
from embed_video.fields import EmbedVideoField

# Create your models here.
class Video(models.Model):
    title = models.CharField(max_length=100)
    video_file = models.FileField(upload_to='videos/')
    # emb_video = EmbedVideoField(default='https://www.youtube.com/watch?v=Pv8N1PamwPQ')
    
class parkingSlot(models.Model):
    slotNo = models.CharField(max_length=5)
    slotState = models.BooleanField(default="false")
    
class vehicle(models.Model):
    plateNo =models.CharField(max_length=8)
    vehicleType = models.CharField(max_length=8)
    slotAllocated = models.CharField(max_length=8)
    entryDate = models.DateTimeField(auto_now_add=True)
    entryTime = models.DateTimeField(auto_now_add=True)
    exitTime = models.DateTimeField(auto_now=True)
    
    def __str__(self) -> str:
        return super().__str__()
    
    class Meta:
        ordering = ['-exitTime']
        
class Item(models.Model):
    video = EmbedVideoField()  # same like models.URLField()