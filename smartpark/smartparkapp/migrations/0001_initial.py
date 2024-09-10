# Generated by Django 5.1.1 on 2024-09-10 11:31

import embed_video.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Item',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('video', embed_video.fields.EmbedVideoField()),
            ],
        ),
        migrations.CreateModel(
            name='parkingSlot',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('slotNo', models.CharField(max_length=5)),
                ('slotState', models.BooleanField(default='false')),
            ],
        ),
        migrations.CreateModel(
            name='vehicle',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('plateNo', models.CharField(max_length=8)),
                ('vehicleType', models.CharField(max_length=8)),
                ('slotAllocated', models.CharField(max_length=8)),
                ('entryDate', models.DateTimeField(auto_now_add=True)),
                ('entryTime', models.DateTimeField(auto_now_add=True)),
                ('exitTime', models.DateTimeField(auto_now=True)),
            ],
            options={
                'ordering': ['-exitTime'],
            },
        ),
        migrations.CreateModel(
            name='Video',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=100)),
                ('video_file', models.FileField(upload_to='videos/')),
            ],
        ),
    ]
