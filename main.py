import cv2

cap = cv2.VideoCapture(0)# Open the default camera

while True:
    ret, frame = cap.read()# Read a frame from the camera
    if not ret:#
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# Convert the frame to grayscale
    blur = cv2.GaussianBlur(gray, (21, 21), 0)# Apply Gaussian blur to the grayscale frame

    cv2.imshow("Color", frame)# Display the original frame
    cv2.imshow("Gray", gray)# Display the grayscale frame
    cv2.imshow("Blur", blur)# Display the blurred frame

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
