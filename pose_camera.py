import cv2
import datetime

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Camera', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        now = datetime.datetime.now()
        filename = now.strftime("captura_%H%M_%d%m%Y.jpg")
        cv2.imwrite(f"captures/{filename}", frame)
        print(f"Captura guardada como {filename}")

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()