from ultralytics import YOLOv10
from wandb.integration.ultralytics import add_wandb_callback
import wandb
import os
import pandas as pd
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import cv2
import glob

IMG_DIR = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images'

FOLD = 0
OD_INPUT_SIZE = 384
STD_BOX_SIZE = 20
BATCH_SIZE = 16
EPOCHS = 100

SAMPLE = None
CONDITIONS = ['Spinal Canal Stenosis']
SEVERITIES = ['Normal/Mild', 'Moderate', 'Severe']
LEVELS = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']

DATA_DIR = f'data_fold{FOLD}'

train_val_df = pd.read_csv('rsna-2024-lumbar-spine-degenerative-classification/train.csv')
train_xy = pd.read_csv('rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv')
train_des = pd.read_csv('rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv')

if SAMPLE:
    train_val_df = train_val_df.sample(SAMPLE, random_state=2698)

fold_df = pd.read_csv('/kaggle/input/lsdc-fold-split/5folds.csv')


def get_level(text):
    for lev in ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']:
        if lev in text:
            split = lev.split('_')
            split[0] = split[0].capitalize()
            split[1] = split[1].capitalize()
            return '/'.join(split)
    raise ValueError('Level not found ' + lev)


def get_condition(text):
    split = text.split('_')
    for i in range(len(split)):
        split[i] = split[i].capitalize()
    split = split[:-2]
    return ' '.join(split)


label_df = {'study_id': [], 'condition': [], 'level': [], 'label': []}

for i, row in train_val_df.iterrows():
    study_id = row['study_id']
    for k, label in row.iloc[1:].to_dict().items():
        level = get_level(k)
        condition = get_condition(k)
        label_df['study_id'].append(study_id)
        label_df['condition'].append(condition)
        label_df['level'].append(level)
        label_df['label'].append(label)

label_df = pd.DataFrame(label_df)
label_df = label_df.merge(fold_df, on='study_id')

train_xy = train_xy.merge(train_des, how='inner', on=['study_id', 'series_id'])
label_df = label_df.merge(train_xy, how='inner', on=['study_id', 'condition', 'level'])


def query_train_xy_row(study_id, series_id=None, instance_num=None):
    if series_id is not None and instance_num is not None:
        return label_df[(label_df.study_id == study_id) & (label_df.series_id == series_id) &
                        (label_df.instance_number == instance_num)]
    elif series_id is None and instance_num is None:
        return label_df[(label_df.study_id == study_id)]
    else:
        return label_df[(train_xy.study_id == study_id) & (label_df.series_id == series_id)]


def read_dcm(src_path):
    dicom_data = pydicom.dcmread(src_path)
    image = dicom_data.pixel_array
    image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
    image = np.stack([image] * 3, axis=-1).astype('uint8')
    return image


def get_accronym(text):
    split = text.split(' ')
    return ''.join([x[0] for x in split])


ex = label_df.sample(1).iloc[0]
study_id = ex.study_id
series_id = ex.series_id
instance_num = ex.instance_number

WIDTH = 10

path = os.path.join(IMG_DIR, str(study_id), str(series_id), f'{instance_num}.dcm')
img = read_dcm(path)

tmp_df = query_train_xy_row(study_id, series_id, instance_num)
for i, row in tmp_df.iterrows():
    lbl = f'{get_accronym(row['condition'])}_{row['level']}'
    x, y = row['x'], row['y']
    x1 = int(x - WIDTH)
    x2 = int(x + WIDTH)
    y1 = int(y - WIDTH)
    y2 = int(y + WIDTH)
    color = None
    if row['label'] == 'Normal/Mild':
        color = (0, 255, 0)
    elif row['label'] == 'Moderate':
        color = (255, 255, 0)
    elif row['label'] == 'Severe':
        color = (255, 0, 0)

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    thickness = 1
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, lbl, (x1, y1), fontFace, fontScale, color, thickness, cv2.LINE_AA)


def read_dcm(src_path):
    dicom_data = pydicom.dcmread(src_path)
    image = dicom_data.pixel_array
    image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
    image = np.stack([image] * 3, axis=-1).astype('uint8')
    return image


filtered_df = label_df[label_df.condition.map(lambda x: x in CONDITIONS)]
label2id = {}
id2label = {}
i = 0
for cond in CONDITIONS:
    for level in LEVELS:
        for severity in SEVERITIES:
            cls_ = f'{cond.lower().replace(' ', '_')}_{level}_{severity.lower()}'
            label2id[cls_] = i
            id2label[i] = cls_
            i += 1

train_df = filtered_df[filtered_df.fold != FOLD]
val_df = filtered_df[filtered_df.fold == FOLD]

_IM_DIR = f'{DATA_DIR}/images/train'
_ANN_DIR = f'{DATA_DIR}/labels/train'
name = np.random.choice(os.listdir(_IM_DIR))[:-4]

im = plt.imread(os.path.join(_IM_DIR, name+'.jpg')).copy()
H, W = im.shape[:2]
anns = np.loadtxt(os.path.join(_ANN_DIR, name+'.txt')).reshape(-1, 5)

for _cls, x, y, w, h in anns.tolist():
    x *= W
    y *= H
    w *= W
    h *= H
    x1 = int(x - w / 2)
    x2 = int(x + w / 2)
    y1 = int(y - h / 2)
    y2 = int(y + h / 2)
    label = id2label[_cls]

    c = (0, 255, 255)

    im = cv2.rectangle(im, (x1, y1), (x2, y2), c, 2)
    cv2.putText(im, label, (x1, y1), fontFace, 0.3, c, 1, cv2.LINE_AA)


plt.imshow(im)

for k, v in id2label.items():
    print(f'{k}: {v}')

HOME = os.getcwd()

secret_value_0 = 'i said no'
wandb.login(key=secret_value_0)

wandb.init(
    project='lsdc_yolov10x',
    group=';'.join(CONDITIONS)
)

model = YOLOv10(f'{HOME}/weights/yolov10x.pt')
add_wandb_callback(model, enable_model_checkpointing=True)

model.train(
    project='lsdc_yolov10',
    data='yolo_scs.yaml',
    epochs=EPOCHS,
    imgsz=OD_INPUT_SIZE,
    batch=BATCH_SIZE,
    optimizer='AdamW',
    seed=52,
    cos_lr=True,
    device=[0, 1],
    #     box=2.5,
    #     cls=4.5,
    #     dfl=2.5,
    #     dropout=0.05
)

wandb.finish()

_IM_DIR = f'{DATA_DIR}/images/val'
_ANN_DIR = f'{DATA_DIR}/labels/val'
name = np.random.choice(os.listdir(_IM_DIR))[:-4]

path = os.path.join(_IM_DIR, name + '.jpg')

im = plt.imread(path).copy()
H, W = im.shape[:2]
anns = np.loadtxt(os.path.join(_ANN_DIR, name + '.txt')).reshape(-1, 5)

for _cls, x, y, w, h in anns.tolist():
    x *= W
    y *= H
    w *= W
    h *= H
    x1 = int(x - w / 2)
    x2 = int(x + w / 2)
    y1 = int(y - h / 2)
    y2 = int(y + h / 2)
    label = id2label[_cls]
    print(label)

    c = (0, 255, 255)

    im = cv2.rectangle(im, (x1, y1), (x2, y2), c, 2)
    cv2.putText(im, label, (x1, y1), fontFace, 0.3, c, 1, cv2.LINE_AA)

model = YOLOv10(glob.glob('lsdc_yolov10/*/weights/best.pt')[0])

out = model.predict([path], save=True, conf=0.2)

wandb.finish()
