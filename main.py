import yolov5
"Grab the part of the screen"
import pyscreenshot as ImageGrab
import time
import pyautogui

# Grab the center of the screen 416x416

im = ImageGrab.grab()

# save image file
im.save("box.png")

model = yolov5.load('valorant-v12.pt')

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

img = 'box.png'  # or file, Path, PIL, OpenCV, numpy, list
start = time.time()
results = model(img, size=640)
print(f'Done. ({time.time() - start:.3f}s)')

# parse results
predictions = results.pred[0]
boxes = predictions[:, :4] # x1, y1, x2, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

# Find the center of the box

center = (boxes[:, 0] + boxes[:, 2]) // 2, (boxes[:, 1] + boxes[:, 3]) // 2

center = int(center[0]), int(center[1])

# Move the mouse to the center of the box
pyautogui.moveTo(center[0], center[1])




