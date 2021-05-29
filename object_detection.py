import streamlit as st
import time
import matplotlib.pyplot as plt
from PIL import Image
import urllib
import gluoncv
from gluoncv import model_zoo, data, utils
import mxnet as mx
import matplotlib.image as mpimg
import matplotlib.patches as patches

def load_rcnn():
  #load pretrained model
  net = model_zoo.get_model('faster_rcnn_fpn_resnet50_v1b_coco', pretrained=True)
  return net

def load_ssd():
  #load pretrained model
  net_ssd = model_zoo.get_model('ssd_512_vgg16_atrous_coco', pretrained=True)
  return net_ssd

def mxnet_run(net_, img_path, min_precision, col2):
  net = net_

  #get and transform image
  im_fname = img_path
  x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)

  thresh = min_precision / 100

  #detect objects and display results
 
  box_ids, scores, bboxes = net(x)
  ax = utils.viz.plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], thresh, class_names=net.classes, linewidth=2, fontsize=9)
  
  column2 = col2
  st.text("")
  column2.subheader("Результат распознавания")
  st.text("")
  column2.pyplot(use_column_width=True)

  bboxes_= bboxes[0]
  scores = scores[0]
  labels = box_ids[0]
  class_names=net.classes
  flag = 0
  thresh = min_precision / 100
  
  if isinstance(bboxes_, mx.nd.NDArray):
    bboxes_ = bboxes_.asnumpy()
  if isinstance(labels, mx.nd.NDArray):
    labels = labels.asnumpy()
  if isinstance(scores, mx.nd.NDArray):
    scores = scores.asnumpy()

  for i, bbox in enumerate(bboxes_):
    if scores is not None and scores.flat[i] < thresh:
      continue
    if labels is not None and labels.flat[i] < 0:
      continue
    cls_id = int(labels.flat[i]) if labels is not None else -1
       
    if class_names is not None and cls_id < len(class_names):
      class_name = class_names[cls_id]
    else:
      class_name = str(cls_id) if cls_id >= 0 else ''
    score = '{:.3f}'.format(scores.flat[i]*100) if scores is not None else ''
    if class_name or score:
      st.write("Найден объект класса '{:s}' - с уверенностью {:s} %".format(class_name, score))
      flag = flag + 1
    
  if flag == 0:
    st.write("Ни одного объекта не обнаружено.")
    st.warning("Возможно, объекты на изображении не принадлежат ни к одному из классов COCO датасета")


@st.cache(show_spinner=False)
def load_model(model_type, speed):
  #Import ObjectDetection class from the ImageAI library
  from imageai.Detection import ObjectDetection  
  
  #create an instance of the class ObjectDetection
  detector = ObjectDetection()  

  if model_type == 'retina_net':
    # dowload model file from google drive
    model_path = "/content/gdrive/MyDrive/Object-detection/resnet50_coco_best_v2.1.0.h5"
    # dowload model locally 
    #model_path = "/content/resnet50_coco_best_v2.1.0.h5" 

    #set the type of pretrained model
    detector.setModelTypeAsRetinaNet() 

  elif model_type == 'yolov3':
    # dowload model file from google drive
    model_path = "/content/gdrive/MyDrive/Object-detection/yolo.h5"
    # dowload model locally
    #model_path = "/content/pretrained-yolov3.h5"

    #set the type of pretrained model
    detector.setModelTypeAsYOLOv3() 
     
  elif model_type == 'yolo_tiny':
    # dowload model file from google drive
    model_path = "/content/gdrive/MyDrive/Object-detection/yolo-tiny.h5"
    # dowload model locally
    #model_path = "/content/yolo-tiny.h5"

    #set the type of pretrained model
    detector.setModelTypeAsTinyYOLOv3()

  else:
    print("Error: wrong model type") 
    
  #set the path to the model 
  detector.setModelPath(model_path)
  #load the model from the path specified and specify speed
  detector.loadModel(speed)

  return detector

def run_detection(detector_, my_img_path, score_threshold, col2):

  input_path = my_img_path
  output_path = "/content/newimage.jpg" # //specify the path from our input image, output image, and model.

  #load any image and display 

  #function returns a dictionary which contains the names and percentage probabilities of all the objects detected in the image.
  #the dictionary items can be accessed by traversing through each item in the dictionary.
  detector = detector_
  detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path, minimum_percentage_probability=score_threshold)
  if len(detection) > 0: 
    for eachItem in detection:
      st.write("Найден объект класса '{}' - с уверенностью {} %".format(eachItem["name"],round(eachItem["percentage_probability"], 3)))
  else:
    st.write("Ни одного объекта не обнаружено.")
    st.warning("Возможно, объекты на изображении не принадлежат ни к одному из классов COCO датасета")
  
  img = mpimg.imread(output_path)

  column2 = col2
  st.text("")
  column2.subheader("Результат распознавания")
  st.text("")
  plt.figure(figsize = (15,15))
  plt.imshow(img)
  column2.pyplot(use_column_width=True)

@st.cache(show_spinner=False)
def load_ssd_0():
  precision = 'fp32'
  ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)

  utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

  ssd_model.to('cuda')
  ssd_model.eval()

  return ssd_model, utils

def ssd_detection(model, utils, img_path, min_precision, col2):
  
  precision = 'fp32'
  uris = [img_path]
  inputs = [utils.prepare_input(uri) for uri in uris]
  tensor = utils.prepare_tensor(inputs, precision == 'fp16')

  with torch.no_grad():
    detections_batch = model(tensor)
  
  results_per_input = utils.decode_results(detections_batch)
  best_results_per_input = [utils.pick_best(results, min_precision) for results in results_per_input]

  classes_to_labels = utils.get_coco_object_dictionary()

  for image_idx in range(len(best_results_per_input)):

    fig, ax = plt.subplots(1)
    # Show original, denormalized image...
    image = inputs[image_idx] / 2 + 0.5
    img = Image.fromarray(image, 'RGB')
    img.save("img_ssd.jpg", "JPEG")
    ax.imshow(image)
    # ...with detections
    bboxes, classes, confidences = best_results_per_input[image_idx]
    for idx in range(len(bboxes)):
      left, bot, right, top = bboxes[idx]
      x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
      rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
      ax.add_patch(rect)
      ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
      if len(bboxes) > 0:
        st.write("Найден объект класса '{}' - с уверенностью {} %".format(classes_to_labels[classes[idx] - 1], round(confidences[idx]*100, 3)))
      else:
        st.write("Ни одного объекта не обнаружено.")
        st.warning("Возможно, объекты на изображении не принадлежат ни к одному из классов COCO датасета")

    column2 = col2
    st.text("")
    column2.subheader("Результат распознавания")
    st.text("")
    column2.pyplot(fig, use_column_width=True)

def to_jpeg(file):
  # Encode PIL Image as a JPEG without writing to disk
  buffer = io.BytesIO()
  file.save(buffer, format='JPEG', quality=75)

  desiredObject = buffer.getbuffer()
  return desiredObject

def set_speed(model_name):
  if model_name == 'YOLOv3' or model_name == 'YOLO tiny':
    speed = 'fast'
  elif model_name == 'RetinaNet':
    speed = 'flash'
  else:
    speed = 'None'
  return speed

def preprocess_models(model_name):
  if model_name == 'SSD':
    net_ = load_ssd()
  
  elif model_name == 'Faster RCNN':
    net_ = load_rcnn()

  else:
    net_ = None
  return net_

#@st.cache(show_spinner=False, suppress_st_warning=True)
def show_image(my_img, col1):
  with st.spinner('Идёт загрузка данных...'):
    column1 = col1
    column1.image(my_img, use_column_width=True)


def show_input(my_img):
  st.set_option('deprecation.showPyplotGlobalUse', False)
  column1, column2 = st.beta_columns(2)
  column1.subheader("Исходное изображение")
  plt.figure(figsize = (15,15))
  with st.spinner('Идёт загрузка данных...'):
    plt.imshow(my_img)
    column1.pyplot() #use_column_width=True
  return column1, column2

#download a single file and make its content available as a string
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
  url = path
  response = urllib.request.urlopen(url)
  return response.read().decode("utf-8")

def image_detection(model_name, speed, img_path, score_threshold, col2, net_):
  #set the chosen model type
  if model_name == "YOLOv3":
    model_type = 'yolov3'

  if model_name == "YOLO tiny":
    model_type = 'yolo_tiny'

  if model_name == "RetinaNet":
    model_type = 'retina_net'

  if model_name == "SSD":
    model_type = 'ssd'

  if model_name == "Faster RCNN":
    model_type = 'faster_rcnn'

  if model_type == 'ssd':
    #model, utils = load_ssd()
    #score_threshold_ = score_threshold / 100
    #ssd_detection(model, utils, img_path, score_threshold_, col2)
    net = net_
    mxnet_run(net, img_path, score_threshold, col2) 
  
  elif model_type == 'faster_rcnn':
    net = net_
    mxnet_run(net, img_path, score_threshold, col2)

  else:
    #load the model settings and file
    detector = load_model(model_type, speed)
    #detect the objects in the image
    run_detection(detector, img_path, score_threshold, col2)

def main():
    
    st.set_page_config(
        page_title="App",
        page_icon= "random",
        layout="centered", #centered or wide
        initial_sidebar_state="expanded") #auto or expanded or collapsed
    
    st.sidebar.header("Что бы вы хотели?")
    choice = st.sidebar.radio("", ("Посмотреть информацию", "Посмотреть пример", "Загрузить своё изображение", "Посмотреть код"))

    if choice == "Посмотреть информацию":
      
      st.header("Распознавание объектов на изображениях")
      st.subheader("*Используя модели YOLOv3, SSD, Faster RCNN, RetinaNet и YOLO tiny*")
      st.text("................................................................................")
      st.write(get_file_content_as_string("https://raw.githubusercontent.com/BelkinaKate/Object-detection/main/info"))

    elif choice == "Загрузить своё изображение":
    
      st.empty()
      st.header("Обнаружение объектов на Вашем изображении")
      st.markdown("Загрузите своё изображение, настройте модель для распознавания и минимальный процент уверенности предсказаний.")
      model_name = st.sidebar.selectbox('Выберите модель', ('SSD', 'YOLOv3', 'Faster RCNN', 'RetinaNet'))
      image_file = st.file_uploader("Загрузите изображение с помощью формы", type=['jpg','jpeg'])
      score_threshold = st.sidebar.slider("Коэффициент уверенности (%)", 0, 100, 50, 1, '%d')
      #speed = st.sidebar.selectbox('Скорость распознавания', ('normal', 'fast', 'faster', 'fastest', 'flash'))
      speed = set_speed(model_name)
      net_ = preprocess_models(model_name)

      if image_file is not None:
        
        my_img = Image.open(image_file)

        my_img.save("out.jpg", "JPEG")
        img_path = "/content/out.jpg"
        image = to_jpeg(my_img)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        column1, column2 = st.beta_columns(2)
        column1.subheader("Исходное изображение")
        show_image(my_img, column1)
        col1, col2 = column1, column2
        #col1, col2 = show_input(my_img)

        clicked = st.button('Найти объекты')
        
        if clicked:
          with st.spinner('Идёт обработка. Пожалуйста подождите...'):
            start_time = time.time()
            image_detection(model_name, speed, img_path, score_threshold, col2, net_)
            st.info("Время распознавания = %s с " % (time.time() - start_time))
        
      else:
          st.info("Загрузите изображение для распознавания.")

    elif choice == "Посмотреть пример":
      st.empty()
      st.header("Пример распознавания объектов на изображении")
      st.markdown("Попробуйте распознавание с помощью разных моделей глубокого обучения.")
      model_name = st.sidebar.selectbox('Выберите модель', ('SSD', 'YOLOv3', 'Faster RCNN', 'RetinaNet'))
      score_threshold = st.sidebar.slider("Коэффициент уверенности (%)",  0, 100, 50, 1, '%d')
      #speed = st.sidebar.selectbox('Скорость распознавания', ('normal', 'fast', 'faster','fastest', 'flash'))
      speed = set_speed(model_name)

      img_path = '/content/gdrive/MyDrive/Object-detection/street1_1.jpg'
      my_img = Image.open(img_path)

      st.set_option('deprecation.showPyplotGlobalUse', False)
      column1, column2 = st.beta_columns(2)
      column1.subheader("Исходное изображение")
      show_image(my_img, column1)
      #col1, col2 = show_input(my_img)

      net_ = preprocess_models(model_name)

      if st.button('Найти объекты'):
        with st.spinner('Идёт обработка. Пожалуйста подождите...'):
          start_time = time.time()
          image_detection(model_name, speed, img_path, score_threshold, column2, net_)
          st.info("Время распознавания = %s с " % round(time.time() - start_time, 3))
    
    elif choice == "Посмотреть код":
      st.header("Код веб-приложения для распознавания объектов")
      st.code(get_file_content_as_string("https://raw.githubusercontent.com/BelkinaKate/Object-detection/main/object_detection.py"))

if __name__ == '__main__':
    main()
