import cv2

cap = cv2.VideoCapture(0)  # Open the default webcam
background = None  # Will store the running average background

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)       # Convert to grayscale
    blur = cv2.GaussianBlur(gray, (21, 21), 0)           # Smooth the image

    # Initialize background on first frame
    if background is None:
        background = blur.astype("float")                # store as float for accumulateWeighted
        continue

    # Update background slowly for smoother detection
    cv2.accumulateWeighted(blur, background, 0.05)       # 0.05 = smoothing factor
    background_uint8 = background.astype("uint8")        # convert back for absdiff

    # Difference between background and current frame
    frame_delta = cv2.absdiff(background_uint8, blur)

    # Threshold to highlight motion areas
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)      # make contours smoother

    # Find contours (moving objects)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 1500:              # ignore tiny movements
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show windows
    cv2.imshow("Color", frame)
    cv2.imshow("Delta", frame_delta)
    cv2.imshow("Thresh", thresh)

    # Quit on 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
