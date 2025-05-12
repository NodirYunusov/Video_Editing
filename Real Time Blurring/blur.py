#!/usr/bin/env python
# coding: utf-8

# In[3]:
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
torch.cuda.is_available()

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# In[ ]:


import cv2
import argparse
from ultralytics import YOLO

class FaceBlurProcessor:
    def __init__(self, model_path="D:/Projects/video/blur_best.pt"):
        self.model = YOLO(model_path)

    def blur_faces(self, frame, results):
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                # Ensure coordinates are within frame bounds
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    blurred = cv2.GaussianBlur(face, (51, 51), 30)
                    frame[y1:y2, x1:x2] = blurred
        return frame

    def process_video(self, save_path, video_path = 0):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame, conf=0.5, verbose=False)
            frame = self.blur_faces(frame, results)

            out.write(frame)

        cap.release()
        out.release()
        print(f"[INFO] Saved blurred video to: {save_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO-based face blurring tool")
    parser.add_argument("--video_path", required=True, help="Path to input video")
    parser.add_argument("--save_path", required=True, help="Path to save output video")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    processor = FaceBlurProcessor()
    processor.process_video(args.video_path, args.save_path)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




