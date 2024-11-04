from collections import defaultdict
import cv2
import numpy as np
import requests
from ultralytics import YOLO
import threading
import time

# LINE Notify Token
token = 'hSZKWJiXYZXrjUOAw1vPn7C6hjJBq8Ai21ImGZes1Pb'

def send_line_notify(message, token, image_path=None):
    """Sends a LINE notification asynchronously with optional image attachment."""
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": message}
    
    files = {"imageFile": open(image_path, "rb")} if image_path else None
    try:
        response = requests.post(url, headers=headers, data=data, files=files)
        if files:
            files["imageFile"].close()  # Ensure the file is closed
        return response.status_code
    except Exception as e:
        print(f"Error sending notification: {e}")
        if files:
            files["imageFile"].close()
        return None

def send_line_notify_threaded(message, token, image_path=None):
    """Wrapper to send LINE notifications in a separate thread."""
    thread = threading.Thread(target=send_line_notify, args=(message, token, image_path))
    thread.start()

def track_video():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)  # Camera 0
    track_history = defaultdict(list)
    notified_track_ids = set()  # Track IDs already notified

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True)
            # boxes2 = results[0].boxes.xywh.cpu()
            boxes = results[0].boxes
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
            annotated_frame = results[0].plot()
            found_new_target = False


            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
            confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else []
            class_ids = boxes.cls.int().cpu().tolist() if boxes.cls is not None else []
            coordinates = boxes.xywh.cpu().tolist()

            img_txt = ""
            for track_id, confidence, class_id, coord in zip(track_ids, confidences, class_ids, coordinates):
                x, y, w, h = coord  # Unpack coordinates
                class_name = model.names[class_id]

                # Print the ID, class name, confidence, and bounding box details of each detected object
                print(f"ID: {track_id}, Class: {class_name}, Confidence: {confidence:.2f}, Box: ({x:.2f}, {y:.2f}, {w:.2f}, {h:.2f})")

                img_txt = f"{track_id}_{class_name}_{confidence:.2f}_{time.time()}.jpg"

                if track_id not in notified_track_ids:
                    notified_track_ids.add(track_id)
                    # found_new_target = True

                    image_path = f"data/{img_txt}"
                    cv2.imwrite(image_path, annotated_frame)





            # for idx, result in enumerate(results):
            #     for detection_idx, class_name in enumerate(result.names):
            #         track_id = track_ids[detection_idx] if detection_idx < len(track_ids) else None
            #         # Check if 'cat' (class 15) or 'person' (class 0)
            #         if class_name in [0, 15] and track_id not in notified_track_ids:
            #             notified_track_ids.add(track_id)
            #             found_new_target = True
            #             break

            # Send notification if a new 'cat' or 'person' detected
            # if found_new_target:
            #     image_path = f"data/{img_txt}"
            #     cv2.imwrite(image_path, annotated_frame)
                # send_line_notify_threaded('New target detected', token, image_path)

            # Plot the tracking history
            if track_ids:
                for box, track_id in zip(boxes.xywh.cpu(), track_ids):
                    x, y, w, h = box
                    track_history[track_id].append((float(x), float(y)))
                    if len(track_history[track_id]) > 30:
                        track_history[track_id].pop(0)
                    points = np.array(track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

            # Show the annotated frame
            cv2.imshow("YOLO Real-Time Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Run the tracking function
track_video()
