#!/usr/bin/env python
# coding: utf-8

# In[5]:


from ultralytics import YOLO
import cv2
import time
import os

class PersonTracker:
    def __init__(self, model_path="yolov8n.pt", tracker_config="bytetrack.yaml", save_path=None):
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.prev_time = time.time()
        self.latency = 0
        self.fps = 0
        self.save_path = save_path
        self.writer = None

    def update_metrics(self):
        current_time = time.time()
        self.latency = (current_time - self.prev_time) * 1000
        self.fps = 1.0 / (current_time - self.prev_time)
        self.prev_time = current_time

    def draw_metrics(self, frame):
        cv2.putText(frame, f"FPS: {self.fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Latency: {self.latency:.2f} ms", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        return frame

    def process_frame(self, frame):
        results = self.model.track(source=frame, persist=True, tracker=self.tracker_config, verbose=False)
        result_frame = results[0].plot(labels=False)
        self.update_metrics()
        return self.draw_metrics(result_frame)

    def run(self, video_source):
        cap = cv2.VideoCapture(video_source)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        # Initialize video writer if save path is provided
        if self.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
            self.writer = cv2.VideoWriter(self.save_path, fourcc, fps, (width, height))
            print(f" Saving output to: {self.save_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result = self.process_frame(frame)

            if self.writer:
                self.writer.write(result)

            cv2.imshow("Tracking", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()


# In[7]:


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="People Tracking with YOLOv8 + ByteTrack")
    parser.add_argument("--video_path", type=str, default=0, help="Path to input video or 0 for webcam")
    parser.add_argument("--save_path", type=str, default=None, help= "output_video")
    parser.add_argument("--model_path", type=str, default="yolov8n.pt", help="YOLOv8 model path")
    parser.add_argument("--tracker_config", type=str, default="bytetrack.yaml", help="Tracker config YAML")

    args = parser.parse_args()

    tracker = PersonTracker(
        model_path=args.model_path,
        tracker_config=args.tracker_config,
        save_path=args.save_path
    )
    tracker.run(args.video_path)


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




