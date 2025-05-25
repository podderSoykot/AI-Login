import cv2

rtsp_url = "rtsp://admin:qwq1234.@192.168.1.110/channel1/subtype=0"

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit(-1)
while True:
    ret,frame = cap.read()
    if not ret:
        print("Camera can't receive frame(stream end?).Exiting ...")
        break
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()