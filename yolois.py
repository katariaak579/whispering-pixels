# from ultralytics import YOLO
# import cv2
# import time

# def eyes():
#     cap = cv2.VideoCapture(0)

#     model = YOLO("yolov8n.pt")
#     names = model.names
#     nmis = ""
#     mx = 0

#     start_time = time.time()  # Record the start time

#     while True:
#         ret, frame = cap.read()

#         if not ret:
#             break
        
#         elapsed_time = time.time() - start_time
        
#         if elapsed_time >= 5:
#             break  # Break the loop after 5 seconds
        
#         results = model.predict(frame, conf=0.7, save=True, save_txt=True, save_conf=True, stream=True)
        
#         for r in results:
#             for c in r.boxes.data:
#                 class_id = int(c[5])
#                 class_name = names[class_id]
#                 conf = c[4]
#                 if conf > mx and class_name != "person":
#                     mx = conf
#                     nmis = class_name
#                     x, y, w, h = map(int, c[:4])
#                     x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
                    
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(frame, f"{class_name}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         cv2.imshow("Live Camera", frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
    
#     return nmis

# aser=eyes()

from ultralytics import YOLO
import cv2
import time

def eyes():
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    time.sleep(1)

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not capture frame.")
        cap.release() 
        exit()

    image_filename = "captured_image.jpg"
    cv2.imwrite(image_filename, frame)

    print(f"Image saved as {image_filename}")

    capture_duration = 3
    start_time = time.time()

    while time.time() - start_time < capture_duration:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not capture frame.")
            break

        cv2.imshow("Live Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
 

# model = YOLO("yolov8n.pt")
# src = r'/Users/anubhavkataria/Desktop/YOLO/Whisperis/brush.mov'
# results = model.predict(source=src, conf=0.7,save=True,save_txt=True,save_conf=True,stream=True)
# names = model.names
# # print(results.boxes.conf)
# # print(results[0].boxes.data)
# lst=[]
# max=0
# nameis=""
# si=0
# for r in results:
#     si+=1
#     for c in r.boxes.data:
#         # print(names[int(c)])
#         class_id=int(c[5])
#         class_name=names[class_id]
#         conf=c[4]*100
#         # print(f"ClassName: {class_name} Confidence: {conf}")
#         if conf>max and class_name!="person":
#             max=conf
#             nameis=class_name
#             # print(nameis)

# print(f"Name is {nameis}")
# print(f"Conf is {conf}")
# print(si)