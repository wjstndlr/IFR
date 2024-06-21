import cv2
import numpy as np
import time
from tensorflow.lite.python.interpreter import Interpreter
import matplotlib.pyplot as plt


### Define function for inferencing with TFLite model and displaying results
capture_count = 0
def tflite_detect_images(modelpath, imgpath, lblpath, min_conf=0.7):
  global capture_count
  # Grab filenames of all images in test folder
  #images = glob.glob(imgpath + '/*.jpg') + glob.glob(imgpath + '/*.JPG') + glob.glob(imgpath + '/*.png') + glob.glob(imgpath + '/*.bmp')

  # Load the label map into memory
  with open(lblpath, 'r') as f:
      labels = [line.strip() for line in f.readlines()]

  # Load the Tensorflow Lite model into memory
  interpreter = Interpreter(model_path=modelpath)
  interpreter.allocate_tensors()

  # Get model details
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]

  float_input = (input_details[0]['dtype'] == np.float32)

  input_mean = 127.5
  input_std = 127.5

  # Randomly select test images
  #images_to_test = random.sample(images, num_test_images)
  high_confidence_start_time = None
  # Loop over every image and perform detection
  while True:

      # Load image and resize to expected shape [1xHxWx3]
      success,frame = imgpath.read()
      if not success:
          break
      image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      imH, imW, _ = frame.shape
      image_resized = cv2.resize(image_rgb, (width, height))
      input_data = np.expand_dims(image_resized, axis=0)

      # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
      if float_input:
          input_data = (np.float32(input_data) - input_mean) / input_std

      # Perform the actual detection by running the model with the image as input
      interpreter.set_tensor(input_details[0]['index'],input_data)
      interpreter.invoke()

      # Retrieve detection results
      boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
      classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
      scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

      detections = []
      high_confidence= False
      # Loop over all detections and draw detection box if confidence is above minimum threshold
      for i in range(len(scores)):
          if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

              # Get bounding box coordinates and draw box
              # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
              ymin = int(max(1,(boxes[i][0] * imH)))
              xmin = int(max(1,(boxes[i][1] * imW)))
              ymax = int(min(imH,(boxes[i][2] * imH)))
              xmax = int(min(imW,(boxes[i][3] * imW)))

              cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

              # Draw label
              object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
              label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
              labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
              label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
              cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
              cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

              detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
              high_confidence = True
              if high_confidence_start_time is None:
                high_confidence_start_time = time.time()
      #cv2.imshow('frame', frame)
      cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
      cv2.imshow('frame', frame)
      cv2.waitKey(1)
      #cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
      #plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      #plt.show()
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      if high_confidence:
        if time.time() - high_confidence_start_time >= 5:
            capture_count += 1
            capture_name = f'capture_{capture_count}.jpg'
            cv2.imwrite(capture_name, frame)
            print(f'Capture saved as {capture_name}')
            high_confidence_start_time = None
      else:
            high_confidence_start_time = None
  
PATH_TO_IMAGES=cv2.VideoCapture(0)   # Path to test images folder
PATH_TO_MODEL='detect.tflite'   # Path to .tflite model file
PATH_TO_LABELS='labelmap.txt'   # Path to labelmap.txt file
min_conf_threshold=0.7   # Confidence threshold (try changing this to 0.01 if you don't see any detection results)
#images_to_test = 10   # Number of images to run detection on

# Run inferencing function!
tflite_detect_images(PATH_TO_MODEL, PATH_TO_IMAGES, PATH_TO_LABELS, min_conf_threshold)