import cv2
import os
import time

# Create folders if not exist
if not os.path.exists("Snapshots"):
    os.makedirs("Snapshots")
if not os.path.exists("Videos"):
    os.makedirs("Videos")

cap = cv2.VideoCapture(0)
background = None

motion_detected = False
last_saved = 0
save_delay = 3  # seconds

log_file = open("events.log", "a")

recording = False
video_writer = None
video_start_time = 0
video_duration = 3   # seconds

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

    # -------------------------
    # START VIDEO RECORDING
    # -------------------------
    if motion_detected and not recording:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        filename = f"Videos/video_{int(time.time())}.avi"
        video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

        recording = True
        video_start_time = time.time()
        print("Recording started:", filename)

    # WRITE FRAMES IF RECORDING
    if recording:
        video_writer.write(frame)

        # Stop after 3 seconds
        if time.time() - video_start_time > video_duration:
            recording = False
            video_writer.release()
            video_writer = None
            print("Recording stopped.")

    # -------------------------
    # SNAPSHOT + LOGGING
    # -------------------------
    if motion_detected:
        current_time = time.time()

        if current_time - last_saved > save_delay:
            snap_name = f"Snapshots/motion_{int(current_time)}.jpg"
            cv2.imwrite(snap_name, frame)

            print("Snapshot saved:", snap_name)

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
