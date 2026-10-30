import cv2

for i in range(2):

    cap = cv2.VideoCapture(i)

    if cap.isOpened():

        ret, frame = cap.read()

        if ret:
            print("Camera", i, "opened")

            cv2.imshow(f"Camera {i}", frame)

            cv2.waitKey(3000)

        cap.release()

cv2.destroyAllWindows()