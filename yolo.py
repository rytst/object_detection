import cv2
import urllib.request
import os
import time

from ultralytics import YOLO



os.environ["XDG_SESSION_TYPE"] = "xcb"
save_as = 'image.jpg'
url = 'http://192.168.128.153/capture'
model = YOLO("yolov8n.pt")

while True:
    # time.sleep(5)

    # get jpg data
    urllib.request.urlretrieve(url, save_as)

    image = cv2.imread(save_as)

    results = model(image)

    annotated_image = results[0].plot()

    cv2.imshow("Image", annotated_image)


    # press "q" to stop this loop
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()

