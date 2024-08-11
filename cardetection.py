import cv2
class CarDetector:
    def __init__(self, videopath) -> None:
        self.path = videopath

    def run(self) -> None:
        # Get Frames of Video
        cap = cv2.VideoCapture(self.path)
        while True:
            ret, frame = cap.read()
            cv2.imshow("Frame",frame)
            key = cv2.waitKey(30)
            if key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
