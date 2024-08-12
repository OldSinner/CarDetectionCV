import cv2
from tracker import *
class StableCameraCarDetector:
    def __init__(self, videopath) -> None:
        self.path = videopath
        self.masker = cv2.createBackgroundSubtractorMOG2(history = 200, varThreshold = 100)
        self.tracker = EuclideanDistTracker()

    def run(self) -> None:
        # Get Frames of Video
        cap = cv2.VideoCapture(self.path)
        while True:
            ret, frame = cap.read()

            # extract POI
            roi = frame[340:600, 500:700]

            # Processing Mask
            mask = self.masker.apply(roi)
            _, mask = cv2.threshold(mask,250,255, cv2.THRESH_BINARY)
            cts, _ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            detections = []
            # ShowCont
            for ct in cts:
                # area 
                area = cv2.contourArea(ct)
                if area > 200:
                    x,y,w,h = cv2.boundingRect(ct)
                    detections.append([x,y,w,h])
                    cv2.rectangle(roi,(x,y), (x+w, y+h), (0,255,0),2)

            # Object track
            boxes_id = self.tracker.update(detections)
            for box in boxes_id:
                x,y,w,h,id = box
                cv2.putText(frame,str(id),(500+ x,y + 320),1,1,(0,255,0),2)
            

            # Display
            cv2.imshow("Frame",frame)
            cv2.imshow("Mask", mask)
            # cv2.imshow("POI", roi)

            # Close Display
            key = cv2.waitKey(30)
            if key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
