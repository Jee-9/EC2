import json
import os
import random
from decimal import Decimal
import glob
import numpy as np
import boto3
import onnxruntime
import cv2


# new video name
new_video = ""
# bring new video from s3
s3 = boto3.client('s3')
s3.download_file('youtubepj-v3',new_video, '/tmp/'+ new_video + '.webm')


# snapshot sampling
def extract_sanpshots(file_path):
  # extract 10 snapshots
  cap = cv2.VideoCapture(file_path)

  sample_count = 0
  captured_count = 0

  while cap.isOpened():
    if not cap.isOpened():
      break

    ret, frame = cap.read()
    sample_count += 1

    if not ret:
      break
      
    if sample_count == 120 :
      captured_count += 1
      files_name = str(new_video) + '_'+ str(captured_count) +'.jpg'
      cv2.imwrite('/image/' + files_name, frame)
      sample_count = 0

  cap.release()

  # random sampling 10 images
  image_list = glob.glob('./image/*.jpg')
  sample_images = random.sample(image_list, 10)

  # delete other images
  for f in image_list:
    if f not in sample_images:
      os.remove(f)

  # delete new_video
  os.remove(file_path)

  return sample_images

img_list = extract_sanpshots(newvideo)

# preprocessing
class ImagePreProcessing:
  def __init__(self, jpg):
    self.image_data = jpg

  def __call__(self): 
    # decode
    encoded_bi_img = np.frombuffer(self.image_data, dtype = np.uint8)  # buffer means byte
    img_ar = cv2.imdecode(encoded_bi_img, cv2.IMREAD_COLOR)  # 1D-array encoded_binary_img 3D-array로 
    # 동일한 사이즈로
    img_ar = cv2.resize(img_ar, (224,224))
    # 0~255 사이의 데이터를 0~1까지로 변환
    img_ar = img_ar/255
    # zero-centered
    img_ar = (img_ar-0.5)/1   # 표준편차 계산해보기
    # pytoroch tensor에 형태 맞춰주기
    img_ar = np.transpose(img_ar, (2, 0, 1))
    # to float
    img_ar = img_ar.astype(np.float64)
    # 배치형태를 위해 4차원으로 변형시키기
    img_ar = np.expand_dims(img_ar, axis = 0)
    
    return img_ar

def img_list_to_ndarray(image_list):
  x_data = np.zeros(shape = (1, 3, 224, 224))

  for f in image_list:
    with open(f, 'rb') as f:  
        new_data = f.read()
    img = ImagePreProcessing(new_data)()
    x_data = np.append(x_data, img, axis = 0)

  x_data = np.delete(x_data, 0, 0)

  return x_data

# make x_data
x_data = img_list_to_ndarray(img_list)

# onnx model 
ort_model = onnxruntime.InferenceSession('new_final_resnet50.onnx')

# result 
ort_input = {ort_session.get_inputs()[0].name: x_data}
## prediction
ort_out_vectors, ort_out_classes = ort_session.run(None, ort_inputs)

## feature vector
video_vec = np.mean(ort_out_vectors, 0, keepdims = True)

to_db = video_vec.tobytes()

# predicted class
dataset_class = {0 : 'education_image',
                 1 : 'game_image',
                 2 : 'kpop_image',
                 3 : 'mukbang_image'}
sum_class_mat = np.sum(ort_out_classes, axis = 0)
video_class = int(np.where(sum_class_mat == np.max(sum_class_mat))[0])
class_name = dataset_class[video_class]


# save result data in DynamoDB
dynamodb = boto3.client('dynamodb')
response = dynamodb.put_item(
  TableName = 'youtube_table_v2',
  Item = {
    'pk' : 'class#' + str(class_name),
    'video' : str(new_video),
    'vector' : to_db
    }
)