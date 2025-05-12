import cv2
import time
import torch
import argparse
import os
from ultralytics import YOLO
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
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    blurred = cv2.GaussianBlur(face, (51, 51), 30)
                    frame[y1:y2, x1:x2] = blurred
        return frame

    def process_video(self, video_path, save_path):
        # Detect webcam usage
        cap = cv2.VideoCapture(0) if str(video_path) == "0" else cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        print("[INFO] Starting face blur. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame, conf=0.5, verbose=False)
            frame = self.blur_faces(frame, results)

            out.write(frame)
            # cv2.imshow("Webcam - Blurring Faces", frame)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            try:
                cv2.imshow("Webcam - Blurring Faces", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                print("[WARNING] cv2.imshow() not supported in your environment.")


        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Saved blurred video to: {save_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO-based face blurring tool")
    parser.add_argument("--video_path", required=True, help="Path to input video or '0' for webcam")
    parser.add_argument("--save_path", required=True, help="Path to save output video")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    processor = FaceBlurProcessor()
    processor.process_video(args.video_path, args.save_path)


# class Custom_detector:
#     def __init__(self, device):
#         self.device = device

#     def detector(self, source, save_path="output.mp4"):
#         device = self.device
#         model = YOLO("yolo11n.pt").to(device)
#         model.eval()

#         cap = cv2.VideoCapture(source)
#         if not cap.isOpened():
#             print(f"Video is not opened: {source}")
#             return

#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS) or 25

#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

#         print(f"Writing started: {save_path}")

#         prev_time = 0
#         frame_count = 0

#         while True:
#             start_time = time.time()
#             ret, frame = cap.read()
#             if not ret:
#                 print("Failed to read frame or reached end of video.")
#                 break

#             results = model.track(
#                 source=frame,
#                 classes=[0],
#                 tracker="bytetrack.yaml",
#                 persist=True,
#                 device=device,
#                 conf=0.4,
#                 iou=0.5,
#                 stream=True
#             )

#             for r in results:
#                 im = r.orig_img.copy()
#                 boxes = r.boxes

#                 if boxes is not None and len(boxes) > 0:
#                     for box in boxes.xyxy:
#                         x1, y1, x2, y2 = map(int, box)
#                         cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Thin rectangle

#             # FPS and latency
#             end_time = time.time()
#             latency = (end_time - start_time) * 1000
#             fps_disp = 1 / (end_time - prev_time + 1e-5)
#             prev_time = end_time

#             text = f"FPS: {fps_disp:.2f}  Latency: {latency:.2f}ms"
#             cv2.putText(im, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#             # Display the frame
#             cv2.imshow('Detection Frame', im)


#             key = cv2.waitKey(10) & 0xFF
#             if key == ord('q'): 
#                 print("Stopping execution...")
#                 break 

         
#             out.write(im)
#             frame_count += 1

#         print(f"{frame_count} frames written. File: {save_path}")

#         cap.release()
#         out.release()
#         cv2.destroyAllWindows()

#     def run(self, source, save_path):
#         self.detector(source=source, save_path=save_path)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--video_path", required=True, help="Input video path or 0 to use webcam")
#     parser.add_argument("--save_path", default="output.mp4", help="Output video file path.")
#     args = parser.parse_args()

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     detector = Custom_detector(device)
#     try:
#         source = int(args.video_path) if args.video_path == "0" else args.video_path
#     except ValueError:
#         source = args.video_path

#     detector.run(source, args.save_path)









