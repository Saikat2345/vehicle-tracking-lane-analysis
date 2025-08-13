import cv2
import yt_dlp

video_url = "https://www.youtube.com/watch?v=MNn9qKG2UFI"
output_path = "traffic.mp4"

ydl_opts = {'outtmpl': output_path, 'format': 'best'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])
