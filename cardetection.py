import cv2
class StableCameraCarDetector:
    def __init__(self, videopath) -> None:
        self.path = videopath
        self.masker = cv2.createBackgroundSubtractorMOG2()

    def run(self) -> None:
        # Get Frames of Video
        cap = cv2.VideoCapture(self.path)
        while True:
            ret, frame = cap.read()

            # extract POI
            roi = frame[340:650, 500:700]

            # Processing Mask
            mask = self.masker.apply(roi)
            cts, _ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # ShowCont
            for ct in cts:
                # area 
                area = cv2.contourArea(ct)
                if area > 100:
                    cv2.drawContours(roi,[ct],-1,(255,0,0),2)

            # Display
            cv2.imshow("Frame",frame)
            # cv2.imshow("Mask", mask)
            cv2.imshow("POI", roi)


            # Close Display
            key = cv2.waitKey(30)
            if key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
