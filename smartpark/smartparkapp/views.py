from http.client import HTTPResponse
from django.shortcuts import render, redirect
from django.http.response import StreamingHttpResponse
from django.template import loader, Context, Template
from .models import Video
# from smartparkingapp.streaming import VideoCamera, IPWebCam, MaskDetect, LiveWebcam, VideoFeed 
from smartparkapp.streaming import VideoFeed
# Create your views here.
def login(request):
    return render(request, 'login.html')

# function to generate the camera videos views
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def monitoring(request):
    # videos = carPark.url
        
    videos = Video.objects.all()
    # embvid = Video.objects.all()
    context = {'videofile':videos}
    # return context
    return render(request, 'monitoring.html',context)

def videoStream(request):
    return StreamingHttpResponse(gen(VideoFeed()), content_type='multipart/x-mixed-replace; boundary=frame')
    

def dashboard(request):
    return render(request, 'dashboard.html')

def slotsTable(request):
    return render(request, 'slotsTable.html')

# handling forms

def upload_video(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('video_list')
    else:
        form = VideoForm()
    return render(request, 'upload_video.html', {'form': form})

def video_list(request):
    videos = Video.objects.all()
    return render(request, 'video_list.html', {'videos': videos})

def parkingslot(request):
    # videos = carPark.url
    if request.method =='POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save() 
    else:
        form = VideoForm()
        
    videos = Video.objects.all()
    embvid = Video.objects.all()
    # context = {'video':videos}
    context = {'videofile':videos,'embvideo':embvid}
    # return context
    return render(request, 'parkingslots.html',context)

