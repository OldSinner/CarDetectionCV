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
            # Processing
            mask = self.masker.apply(frame)

            # Display
            cv2.imshow("Frame",frame)
            cv2.imshow("Mask", mask)

            # Close Display
            key = cv2.waitKey(30)
            if key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
