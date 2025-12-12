import cv2
import os
import time

# Create folder if it doesn't exist
if not os.path.exists("Snapshots"):
    os.makedirs("Snapshots")

cap = cv2.VideoCapture(0)
background = None

motion_detected = False
last_saved = 0
save_delay = 3  # seconds

log_file = open("events.log", "a")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    # Initialize background
    if background is None:
        background = blur.astype("float")
        continue

    # Smooth background update
    cv2.accumulateWeighted(blur, background, 0.05)
    background_uint8 = background.astype("uint8")

    # Frame delta
    frame_delta = cv2.absdiff(background_uint8, blur)

    # Threshold
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    motion_detected = False

    # Contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 1500:
            continue

        motion_detected = True

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Save snapshot & log
    if motion_detected:
        current_time = time.time()

        if current_time - last_saved > save_delay:
            filename = f"Snapshots/motion_{int(current_time)}.jpg"
            cv2.imwrite(filename, frame)

            print("Snapshot saved:", filename)

            log_file.write(f"Motion detected at {time.ctime(current_time)}\n")
            log_file.flush()

            last_saved = current_time

    # Show windows
    cv2.imshow("Color", frame)
    cv2.imshow("Delta", frame_delta)
    cv2.imshow("Thresh", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

log_file.close()
cap.release()
cv2.destroyAllWindows()
