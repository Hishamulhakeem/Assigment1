import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

def drawBbox(frame, trackH, id):
   
    points = trackH[id]
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        
        thickness = int(np.sqrt(float(i + 1)) * 1.5) 
        cv2.line(frame, points[i - 1], points[i], (0, 255, 0), thickness)

def detectionTracking(model, video=0, threshold=0.5):
   
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("Error: Could not open")
        return 0
    
    trackH = defaultdict(lambda: [])
    
    target = [0, 2, 3, 4]  
    targetName = {0:'PErson', 2:'Car', 3:'Bicycle'}
    
    count= 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame")
            break
        count+= 1
        results = model.track(frame,persist=True,conf=threshold,classes=target,verbose=False)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            
            boxes = results[0].boxes.xyxy.cpu().numpy()  
            confidences = results[0].boxes.conf.cpu().numpy()
            classids = results[0].boxes.cls.cpu().numpy().astype(int)
            trackids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, conf, clsid, trackid in zip(boxes, confidences, classids, trackids):
                x1, y1, x2, y2 = map(int, box)
                
                X = (x1 + x2) // 2
                Y = (y1 + y2) // 2
                CP = (X, Y)
                
                trackH[trackid].append(CP)
                if len(trackH[trackid]) > 30:
                    trackH[trackid].pop(0)
                
                drawBbox(frame, trackH, trackid)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                classname = targetName.get(clsid, f'class_{clsid}')
                
                label = f'ID:{trackid} {classname} {conf:.2f}'
                labelSize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                cv2.rectangle(frame,(x1, y1 - labelSize[1] - 10),(x1 + labelSize[0], y1),(0, 255, 255),-1)
                cv2.putText(frame,label,(x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 0),2)
                
        fps_text = f' Total Frame: {count}'
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Assigmnet 1', frame)
        
        if cv2.waitKey(25) & 0xff == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames: {count}")

def main():
    detectionTracking(model=YOLO('yolov8n.pt'),video='test2.mp4',threshold=0.5)

if __name__ == "__main__":
    main()