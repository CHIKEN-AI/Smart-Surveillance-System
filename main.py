import cv2

cap = cv2.VideoCapture(0)  # Open the default webcam

background = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blur = cv2.GaussianBlur(gray, (21, 21), 0)      # Smooth the image

    # First frame becomes background
    if background is None:
        background = blur
        continue

    # Difference between background and current frame
    frame_delta = cv2.absdiff(background, blur)

    # Highlight motion areas
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Show all windows
    cv2.imshow("Color", frame)
    cv2.imshow("Delta", frame_delta)
    cv2.imshow("Thresh", thresh)

    # Update background
    background = blur

    # REQUIRED for window to refresh
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
