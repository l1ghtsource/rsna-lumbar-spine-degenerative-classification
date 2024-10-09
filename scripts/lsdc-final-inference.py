import pandas as pd
import time
from torch.cuda.amp import autocast
import glob
from spacecutter.callbacks import *
from spacecutter.models import *
from spacecutter.losses import *
from spacecutter import *
import timm_3d
import torch.nn as nn
import matplotlib.pyplot as plt
import torchio as tio
from torch.utils.data import Dataset, DataLoader
import copy
import math
from pydicom import dcmread
import open3d as o3d
from collections import defaultdict
from ultralytics import YOLOv10
from tqdm.auto import tqdm
import cv2
import torch
from multiprocessing import Pool, cpu_count
import numpy as np
import pydicom
import os

IMG_DIR = '/images'
FOLD = 0
SEVERITIES = ['Normal/Mild', 'Moderate', 'Severe']
LEVELS = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']

SCS_WEIGHTS = ['best_scs.pt']
SS_WEIGHTS = ['best_ss.pt']
NFN_WEIGHTS = ['best_nfn.pt']
TRAIN_PATH = 'rsna-2024-lumbar-spine-degenerative-classification/train.csv'
DESC_PATH = 'rsna-2024-lumbar-spine-degenerative-classification/test_series_descriptions.csv'
INPUT_DIRECTORY = 'rsna-2024-lumbar-spine-degenerative-classification/test_images'

train_val_df = pd.read_csv(TRAIN_PATH)
des = pd.read_csv(DESC_PATH)


def read_dcm(src_path):
    dicom_data = pydicom.dcmread(src_path)
    image = dicom_data.pixel_array
    image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
    return image


def convert_dcm_to_jpg(file_path):
    try:
        image_array = read_dcm(file_path)

        relative_path = os.path.relpath(file_path, start=input_directory)
        output_path = os.path.join(output_directory, relative_path)
        output_path = output_path.replace('.dcm', '.jpg')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cv2.imwrite(output_path, image_array)

        return output_path
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def process_files(dcm_files):
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(convert_dcm_to_jpg, dcm_files), total=len(dcm_files)))


def get_dcm_files(directory):
    dcm_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.dcm'):
                dcm_files.append(os.path.join(root, file))
    return dcm_files


output_directory = IMG_DIR
dcm_files = get_dcm_files(INPUT_DIRECTORY)
process_files(dcm_files)

test_df = os.listdir(INPUT_DIRECTORY)
test_df = pd.DataFrame(test_df, columns=['study_id'])
test_df['study_id'] = test_df['study_id'].astype(int)
test_df = test_df.merge(des, on=['study_id'])


def gen_label_map(CONDITIONS):
    label2id = {}
    id2label = {}
    i = 0
    for cond in CONDITIONS:
        for level in LEVELS:
            for severity in SEVERITIES:
                cls_ = f"{cond.lower().replace(' ', '_')}_{level}_{severity.lower()}"
                label2id[cls_] = i
                id2label[i] = cls_
                i += 1
    return label2id, id2label


scs_label2id, scs_id2label = gen_label_map(['Spinal Canal Stenosis'])
ss_label2id, ss_id2label = gen_label_map(['Left Subarticular Stenosis', 'Right Subarticular Stenosis'])
nfn_label2id, nfn_id2label = gen_label_map(['Left Neural Foraminal Narrowing', 'Right Neural Foraminal Narrowing'])

scs_models = []
for weight in SCS_WEIGHTS:
    scs_models.append(YOLOv10(weight))

ss_models = []
for weight in SS_WEIGHTS:
    ss_models.append(YOLOv10(weight))

nfn_models = []
for weight in NFN_WEIGHTS:
    nfn_models.append(YOLOv10(weight))

all_label_set = train_val_df.iloc[0, 1:].index.tolist()
scs_label_set = all_label_set[:5]
nfn_label_set = all_label_set[5:15]
ss_label_set = all_label_set[15:]

mu = 32768 * 4

settings = [
    ('Sagittal T2/STIR', scs_models, scs_id2label, scs_label_set, 0.01 / mu),
    ('Axial T2', ss_models, ss_id2label, ss_label_set, 0.01 / mu),
    ('Sagittal T1', nfn_models, nfn_id2label, nfn_label_set, 0.1 / mu)
]

pred_rows = []
for modality, models, id2label, label_set, thresh in settings:
    mod_df = test_df[test_df.series_description == modality]

    for study_id, group in tqdm(mod_df.groupby('study_id')):
        predictions = defaultdict(list)
        for i, row in group.iterrows():
            series_dir = os.path.join(IMG_DIR, str(row['study_id']), str(row['series_id']))
            for model in models:
                results = model(series_dir, conf=thresh, verbose=False, augment=True)
                for res in results:
                    for pred_class, conf in zip(res.boxes.cls, res.boxes.conf):
                        pred_class = pred_class.item()
                        conf = conf.item()
                        _class = id2label[pred_class]
                        predictions[_class].append(conf)

        for condition in label_set:
            res_dict = {'row_id': f'{study_id}_{condition}'}

            score_vec = []
            for severity in SEVERITIES:
                severity = severity.lower()
                key = f'{condition}_{severity}'
                if len(predictions[key]) > 0:
                    score = np.max(predictions[key])
                else:
                    score = thresh
                score_vec.append(score)

            score_vec = torch.tensor(score_vec)
            score_vec = score_vec / score_vec.sum()

            for idx, severity in enumerate(SEVERITIES):
                res_dict[severity.replace('/', '_').lower()] = score_vec[idx].item()

            pred_rows.append(res_dict)

pred_df = pd.DataFrame(pred_rows)
pred_df

pred_df.to_csv('submission_yolo.csv', index=False)

DATA_PATH = 'rsna-2024-lumbar-spine-degenerative-classification/'


def retrieve_test_data(data_path):
    test_df = pd.read_csv(data_path + 'test_series_descriptions.csv')

    return test_df


retrieve_test_data(DATA_PATH)


def retrieve_image_paths(base_path, study_id, series_id):
    series_dir = os.path.join(base_path, str(study_id), str(series_id))
    images = os.listdir(series_dir)
    image_paths = [os.path.join(series_dir, img) for img in images]
    return image_paths


def read_study_as_pcd(dir_path, series_types_dict=None, downsampling_factor=1, img_size=(256, 256)):
    pcd_overall = o3d.geometry.PointCloud()

    for path in glob.glob(os.path.join(dir_path, '**/*.dcm'), recursive=True):
        dicom_slice = dcmread(path)

        series_id = os.path.basename(os.path.dirname(path))
        _ = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if series_types_dict is None or int(series_id) not in series_types_dict:
            series_desc = dicom_slice.SeriesDescription
        else:
            series_desc = series_types_dict[int(series_id)]
            series_desc = series_desc.split(" ")[-1]

        x_orig, y_orig = dicom_slice.pixel_array.shape
        img = np.expand_dims(cv2.resize(dicom_slice.pixel_array, img_size, interpolation=cv2.INTER_AREA), -1)
        x, y, z = np.where(img)

        downsampling_factor_iter = max(downsampling_factor, int(math.ceil(len(x) / 6e6)))

        index_voxel = np.vstack((x, y, z))[:, ::downsampling_factor_iter]
        grid_index_array = index_voxel.T
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid_index_array.astype(np.float64)))

        vals = np.expand_dims(img[x, y, z][::downsampling_factor_iter], -1)
        if series_desc == 'T1':
            vals = np.pad(vals, ((0, 0), (0, 2)))
        elif series_desc == 'T2':
            vals = np.pad(vals, ((0, 0), (1, 1)))
        elif series_desc == 'T2/STIR':
            vals = np.pad(vals, ((0, 0), (2, 0)))
        else:
            raise ValueError(f'Unknown series desc: {series_desc}')

        pcd.colors = o3d.utility.Vector3dVector(vals.astype(np.float64))

        dX, dY = dicom_slice.PixelSpacing
        dZ = dicom_slice.SliceThickness

        X = np.array(list(dicom_slice.ImageOrientationPatient[:3]) + [0]) * dX
        Y = np.array(list(dicom_slice.ImageOrientationPatient[3:]) + [0]) * dY

        for z in range(int(dZ)):
            pos = list(dicom_slice.ImagePositionPatient)
            if series_desc == 'T2':
                pos[-1] += z
            else:
                pos[0] += z
            S = np.array(pos + [1])

            transform_matrix = np.array([X, Y, np.zeros(len(X)), S]).T
            transform_matrix = transform_matrix @ np.matrix(
                [[0, y_orig / img_size[1], 0, 0],
                 [x_orig / img_size[0], 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]
            )

            pcd_overall += copy.deepcopy(pcd).transform(transform_matrix)

    return pcd_overall


def read_study_as_voxel_grid(dir_path, series_type_dict=None, downsampling_factor=1, img_size=(256, 256)):
    pcd_overall = read_study_as_pcd(dir_path,
                                    series_types_dict=series_type_dict,
                                    downsampling_factor=downsampling_factor,
                                    img_size=img_size)
    box = pcd_overall.get_axis_aligned_bounding_box()

    max_b = np.array(box.get_max_bound())
    min_b = np.array(box.get_min_bound())

    pts = (np.array(pcd_overall.points) - (min_b)) * (
        (img_size[0] - 1, img_size[0] - 1, img_size[0] - 1) / (max_b - min_b))
    coords = np.round(pts).astype(np.int32)
    vals = np.array(pcd_overall.colors, dtype=np.float16)

    grid = np.zeros((3, img_size[0], img_size[0], img_size[0]), dtype=np.float16)
    indices = coords[:, 0], coords[:, 1], coords[:, 2]

    np.maximum.at(grid[0], indices, vals[:, 0])
    np.maximum.at(grid[1], indices, vals[:, 1])
    np.maximum.at(grid[2], indices, vals[:, 2])

    return grid


CONDITIONS = {
    'Sagittal T2/STIR': ['Spinal Canal Stenosis'],
    'Axial T2': ['Left Subarticular Stenosis', 'Right Subarticular Stenosis'],
    'Sagittal T1': ['Left Neural Foraminal Narrowing', 'Right Neural Foraminal Narrowing'],
}


class PatientLevelTestset(Dataset):
    def __init__(self,
                 base_path: str,
                 dataframe: pd.DataFrame,
                 transform_3d=None):
        self.base_path = base_path

        self.dataframe = (dataframe[['study_id', "series_id", "series_description"]]
                          .drop_duplicates())

        self.subjects = self.dataframe[['study_id']].drop_duplicates().reset_index(drop=True)
        self.series_descs = {e[0]: e[1]
                             for e in self.dataframe[['series_id', 'series_description']].drop_duplicates().values}

        self.transform_3d = transform_3d

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index):
        curr = self.subjects.iloc[index]
        study_path = os.path.join(self.base_path, str(curr['study_id']))

        study_images = read_study_as_voxel_grid(study_path, self.series_descs)

        if self.transform_3d is not None:
            study_images = self.transform_3d(torch.FloatTensor(study_images))
            return study_images.to(torch.half), str(curr['study_id'])

        return torch.HalfTensor(study_images), str(curr['study_id'])


transform_3d = tio.Compose([
    tio.RescaleIntensity([0, 1]),
])


def create_subject_level_testset_and_loader(df: pd.DataFrame,
                                            transform_3d,
                                            base_path: str,
                                            batch_size=1,
                                            num_workers=0):
    testset = PatientLevelTestset(base_path, df, transform_3d=transform_3d)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return testset, test_loader


data = retrieve_test_data(DATA_PATH)
dataset, dataloader = create_subject_level_testset_and_loader(
    data,
    transform_3d,
    os.path.join(DATA_PATH, 'test_images')
)

grid = dataset[0][0]
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


class CNN_Model_3D_Multihead(nn.Module):
    def __init__(self,
                 backbone="efficientnet_lite0",
                 in_chans=1,
                 out_classes=5,
                 cutpoint_margin=0.15,
                 pretrained=False):
        super(CNN_Model_3D_Multihead, self).__init__()
        self.out_classes = out_classes

        self.encoder = timm_3d.create_model(
            backbone,
            features_only=False,
            drop_rate=0,
            drop_path_rate=0,
            pretrained=pretrained,
            in_chans=in_chans,
            global_pool='max'
        )
        if 'efficientnet' in backbone:
            head_in_dim = self.encoder.classifier.in_features
            self.encoder.classifier = nn.Sequential(
                nn.LayerNorm(head_in_dim),
                nn.Dropout(0),
            )

        elif 'vit' in backbone:
            self.encoder.head.drop = nn.Dropout(0)
            head_in_dim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()

        self.heads = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(head_in_dim, 1),
                LogisticCumulativeLink(3)
            ) for _ in range(out_classes)]
        )

        self.ascension_callback = AscensionCallback(margin=cutpoint_margin)

    def forward(self, x):
        feat = self.encoder(x)
        return torch.swapaxes(torch.stack([head(feat) for head in self.heads]), 0, 1)

    def _ascension_callback(self):
        for head in self.heads:
            self.ascension_callback.clip(head[-1])


model = CNN_Model_3D_Multihead(backbone='maxvit_rmlp_tiny_rw_256', in_chans=3, out_classes=25).to(device)
model.load_state_dict(torch.load(
    'rsna-2024/pytorch/vit_voxel_v2/6/maxvit_rmlp_tiny_rw_256_256_v2_fold_3_32.pt'
))

CONDITIONS = {
    'Sagittal T2/STIR': ['spinal_canal_stenosis'],
    'Axial T2': ['left_subarticular_stenosis', 'right_subarticular_stenosis'],
    'Sagittal T1': ['left_neural_foraminal_narrowing', 'right_neural_foraminal_narrowing'],
}

ALL_CONDITIONS = sorted(
    [
        'spinal_canal_stenosis',
        'left_subarticular_stenosis',
        'right_subarticular_stenosis',
        'left_neural_foraminal_narrowing',
        'right_neural_foraminal_narrowing'
    ]
)

LEVELS = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']

results_df = pd.DataFrame({'row_id': [], 'normal_mild': [], 'moderate': [], 'severe': []})


study_ids = glob.glob('rsna-2024-lumbar-spine-degenerative-classification/test_images/*')
study_ids = [os.path.basename(e) for e in study_ids]

results_df = pd.DataFrame({'row_id': [], 'normal_mild': [], 'moderate': [], 'severe': []})
for study_id in study_ids:
    for condition in ALL_CONDITIONS:
        for level in LEVELS:
            row_id = f'{study_id}_{condition}_{level}'
            results_df = results_df._append(
                {
                    'row_id': row_id,
                    'normal_mild': 1 / 3,
                    'moderate': 1 / 3,
                    'severe': 1 / 3
                },
                ignore_index=True
            )

start_time = time.time()

with torch.no_grad():
    with autocast(dtype=torch.float16):
        model.eval()
        for images, study_id in dataloader:
            output = model(images.to(device))
            for i, batch_out in enumerate(output):
                batch_out = output.cpu().numpy()[i]
                for index, level in enumerate(batch_out):
                    row_id = f"{study_id[i]}_{ALL_CONDITIONS[index // 5]}_{LEVELS[index % 5]}"
                    results_df.loc[results_df.row_id == row_id, 'normal_mild'] = level[0]
                    results_df.loc[results_df.row_id == row_id, 'moderate'] = level[1]
                    results_df.loc[results_df.row_id == row_id, 'severe'] = level[2]

print('--- %s seconds ---' % (time.time() - start_time))

results_df.to_csv('submission_vit.csv', index=False)

yolo = pd.read_csv('submission_yolo.csv')
vit = pd.read_csv('submission_vit.csv')

sms = pd.merge(yolo, vit, on=['row_id'])
sms['normal_mild'] = sms['normal_mild_x'] * 0.65 + sms['normal_mild_y'] * 0.35
sms['moderate'] = sms['moderate_x'] * 0.65 + sms['moderate_y'] * 0.35
sms['severe'] = sms['severe_x'] * 0.65 + sms['severe_y'] * 0.35

sub = sms[['row_id', 'normal_mild', 'moderate', 'severe']]
sub.to_csv('submission.csv', index=False, float_format='%.7f')
