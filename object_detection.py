import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, urllib
import base64
import io
from io import BytesIO
#from google.colab.patches import cv2_imshow
from base64 import b64decode, b64encode

def retina_net(my_img_path, score_threshold, speed, col2):

  #Import ObjectDetection class from the ImageAI library.
  from imageai.Detection import ObjectDetection 
  
  #create an instance of the class ObjectDetection
  detector = ObjectDetection()
  
  #specify the path from our input image, output image, and model
  #model_path = "/content/resnet50_coco_best_v2.1.0.h5" #download model locally
  #download model from google drive
  model_path = "/content/gdrive/MyDrive/Object-detection/resnet50_coco_best_v2.1.0.h5"
  input_path = my_img_path
  output_path = "/content/newimage.jpg" 

  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg

  #load our model
  detector.setModelTypeAsRetinaNet()
  #function which accepts a string which contains the path to the pre-trained model.
  detector.setModelPath(model_path)
  #Loads the model from the path specified above.
  detector.loadModel(speed) 

  #function returns a dictionary which contains the class names and percentage probabilities.
  #the dictionary items can be accessed by traversing through each item in the dictionary.
  detection = detector.detectObjectsFromImage(
    input_image=input_path, output_image_path=output_path, minimum_percentage_probability=score_threshold) 
  for eachItem in detection:
    st.write("Found {} Object - {}".format(eachItem["name"],eachItem["percentage_probability"]))
  
  #display the result image
  img = mpimg.imread(output_path)

  column2 = col2
  st.text("")
  column2.subheader("Output image")
  st.text("")
  plt.figure(figsize = (15,15))
  plt.imshow(img)
  column2.pyplot(use_column_width=True)

def yolov3_detection(my_img_path, score_threshold, speed, col2):

  #Import ObjectDetection class from the ImageAI library.
  from imageai.Detection import ObjectDetection 
  
  #create an instance of the class ObjectDetection
  detector = ObjectDetection()
  
  #specify the path from our input image, output image, and model
  #model_path = "/content/pretrained-yolov3.h5" #download model locally
  #download model from google drive
  model_path = "/content/gdrive/MyDrive/Object-detection/yolo.h5"
  input_path = my_img_path
  output_path = "/content/newimage.jpg" 

  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg

  #load our model
  detector.setModelTypeAsYOLOv3()
  #function which accepts a string which contains the path to the pre-trained model.
  detector.setModelPath(model_path)
  #Loads the model from the path specified above.
  detector.loadModel(speed) 

  #function returns a dictionary which contains the class names and percentage probabilities.
  #the dictionary items can be accessed by traversing through each item in the dictionary.
  detection = detector.detectObjectsFromImage(
    input_image=input_path, output_image_path=output_path, minimum_percentage_probability=score_threshold) 
  for eachItem in detection:
    st.write("Found {} Object - {}".format(eachItem["name"],eachItem["percentage_probability"]))
  
  #display the result image
  img = mpimg.imread(output_path)

  column2 = col2
  st.text("")
  column2.subheader("Output image")
  st.text("")
  plt.figure(figsize = (15,15))
  plt.imshow(img)
  column2.pyplot(use_column_width=True)  
  
def yolo_tiny(my_img_path, score_threshold, speed, col2):

  #Import ObjectDetection class from the ImageAI library.
  from imageai.Detection import ObjectDetection 
  
  #create an instance of the class ObjectDetection
  detector = ObjectDetection()
  
  #specify the path from our input image, output image, and model
  #model_path = "/content/yolo-tiny.h5" #download model file locally 
  #download model from google drive 
  model_path = "/content/gdrive/MyDrive/Object-detection/yolo-tiny.h5"
  input_path = my_img_path
  output_path = "/content/newimage.jpg" 

  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg

  #load our model
  detector.setModelTypeAsTinyYOLOv3()
  #function which accepts a string which contains the path to the pre-trained model.
  detector.setModelPath(model_path)
  #Loads the model from the path specified above.
  detector.loadModel(speed) 

  #function returns a dictionary which contains the class names and percentage probabilities.
  #the dictionary items can be accessed by traversing through each item in the dictionary.
  detection = detector.detectObjectsFromImage(
    input_image=input_path, output_image_path=output_path, minimum_percentage_probability=score_threshold) 
  for eachItem in detection:
    st.write("Found {} Object - {}".format(eachItem["name"],eachItem["percentage_probability"]))
  
  #display the result image
  img = mpimg.imread(output_path)

  column2 = col2
  st.text("")
  column2.subheader("Output image")
  st.text("")
  plt.figure(figsize = (15,15))
  plt.imshow(img)
  column2.pyplot(use_column_width=True)  


# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


def to_jpeg(file):
  # Encode your PIL Image as a JPEG without writing to disk
  buffer = io.BytesIO()
  file.save(buffer, format='JPEG', quality=75)

  desiredObject = buffer.getbuffer()
  return desiredObject

def show_input(my_img):
  st.set_option('deprecation.showPyplotGlobalUse', False)
  column1, column2 = st.beta_columns(2)
  column1.subheader("Input image")
  st.text("")
  plt.figure(figsize = (16,16))
  plt.imshow(my_img)
  column1.pyplot(use_column_width=True)
  return column1, column2


def main():
    
    st.sidebar.header("Что бы вы хотели?")

    #score_threshold = st.sidebar.slider("Confidence_threshold", 0.00,1.00,0.5,0.01)
    #nms_threshold = st.sidebar.slider("NMS_threshold", 0.00, 1.00, 0.4, 0.01)

    choice = st.sidebar.radio(
      "", ("Просмотр информации", "Посмотреть пример", "Загрузить своё изображение", "Посмотреть код"))

    if choice == "Просмотр информации":
      st.header("Распознавание объектов :)")
      st.subheader("*Используя модели YOLOv3, YOLO tiny и RetinaNet*")
      #st.write("  ")

    elif choice == "Загрузить своё изображение":
      model_name = st.sidebar.selectbox('*Выберите модель*', ('YOLOv3', 'YOLO tiny', 'RetinaNet'))
      image_file = st.file_uploader("Upload", type=['jpg','png','jpeg'])
      score_threshold = st.sidebar.slider("Коэффициент уверенности (%)", 0, 100, 50, 1, '%d')
      speed = st.sidebar.selectbox('Скорость распознавания', ('normal', 'fast', 'faster', 'fastest', 'flash'))
      clicked = st.button('Найти объекты')

      if image_file is not None:
        my_img = Image.open(image_file)
        my_img.save("out.jpg", "JPEG")
        img_path = "/content/out.jpg"
        image = to_jpeg(my_img)
        col1, col2 = show_input(my_img)
        
        if clicked:
          st.set_option('deprecation.showfileUploaderEncoding', False)
          #image_file = st.file_uploader("Upload", type=['jpg','png','jpeg'])

        #if image_file is not None:
          #my_img = Image.open(image_file)
          #my_img.save("out.jpg", "JPEG")
          #img_path = "/content/out.jpg"
          #image = to_jpeg(my_img)
          #col1, col2 = show_input(my_img)

          if model_name == "YOLOv3":
            yolov3_detection(img_path, score_threshold, speed, col2)
          if model_name == "YOLO tiny":
            yolo_tiny(img_path, score_threshold, speed, col2)
          if model_name == "RetinaNet":
            retina_net(img_path, score_threshold, speed, col2)
        
      else:
          st.warning("Не удалось загрузить изображение. Попробуйте ещё раз.")

    elif choice == "Посмотреть пример":
      model_name = st.sidebar.selectbox('Выберите модель', ('YOLOv3', 'YOLO tiny', 'RetinaNet'))
      score_threshold = st.sidebar.slider("Коэффициент уверенности (%)",  0, 100, 50, 1, '%d')
      speed = st.sidebar.selectbox('Скорость распознавания', ('normal', 'fast', 'fastest', 'flash')) 
      clicked = st.sidebar.button('Найти объекты')

      img_path = '/content/gdrive/MyDrive/Object-detection/input.jpg'
      my_img = Image.open(img_path)
      col1, col2 = show_input(my_img)

      if clicked:
        #img_path = "/content/example_img.jpg"
        #img_path = '/content/gdrive/MyDrive/Object-detection/input.jpg'
        #my_img = Image.open(img_path)
        #col1, col2 = show_input(my_img)

        if model_name == "YOLOv3":
          yolov3_detection(img_path, score_threshold, speed, col2)
        if model_name == "YOLO tiny":
          yolo_tiny(img_path, score_threshold, speed, col2)
          #st.balloons()
        if model_name == "RetinaNet":
            retina_net(img_path, score_threshold, speed, col2)
    
    elif choice == "Посмотреть код":
      st.code(get_file_content_as_string("/content/yolo_app3.py"))

if __name__ == '__main__':
    main()
