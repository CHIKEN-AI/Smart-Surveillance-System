import cv2
import os
import time

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Create folders if not exist
if not os.path.exists("Snapshots"):
    os.makedirs("Snapshots")
if not os.path.exists("Videos"):
    os.makedirs("Videos")

cap = cv2.VideoCapture(0)
background = None

last_saved = 0
save_delay = 3

log_file = open("events.log", "a")

recording = False
video_writer = None
video_start_time = 0
video_duration = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    if background is None:
        background = blur.astype("float")
        continue

    cv2.accumulateWeighted(blur, background, 0.05)
    background_uint8 = background.astype("uint8")

    frame_delta = cv2.absdiff(background_uint8, blur)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    motion_detected = False
    motion_regions = []

    contours, _ = cv2.findContours(
        thresh.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        if cv2.contourArea(contour) < 3000:
            continue

        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_regions.append((x, y, w, h))

    # ---------------- FACE DETECTION (ROI BASED) ----------------
    faces = []

    if motion_detected:
        for (x, y, w, h) in motion_regions:
            roi_gray = gray[y:y + h, x:x + w]

            detected = face_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.05,
                minNeighbors=8,
                minSize=(60, 60)
            )

            for (fx, fy, fw, fh) in detected:
                faces.append((x + fx, y + fy, fw, fh))

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
        cv2.putText(
            frame,
            "Face Detected",
            (fx, fy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    # ---------------- VIDEO RECORDING ----------------
    if motion_detected and not recording:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        filename = f"Videos/video_{int(time.time())}.avi"
        video_writer = cv2.VideoWriter(
            filename,
            fourcc,
            20.0,
            (frame.shape[1], frame.shape[0])
        )
        recording = True
        video_start_time = time.time()

    if recording:
        video_writer.write(frame)
        if time.time() - video_start_time > video_duration:
            recording = False
            video_writer.release()
            video_writer = None

    # ---------------- SNAPSHOT + LOGGING ----------------
    if motion_detected:
        current_time = time.time()

        if current_time - last_saved > save_delay:
            snap_name = f"Snapshots/motion_{int(current_time)}.jpg"
            cv2.imwrite(snap_name, frame)
            log_file.write(f"Motion detected at {time.ctime(current_time)}\n")
            log_file.flush()
            last_saved = current_time

        if len(faces) > 0:
            log_file.write(f"Face detected at {time.ctime(current_time)}\n")

    cv2.imshow("Color", frame)
    cv2.imshow("Delta", frame_delta)
    cv2.imshow("Thresh", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

log_file.close()
cap.release()
cv2.destroyAllWindows()
