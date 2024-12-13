import json
import os

import numpy as np
import re
import torch
import torchio
from numpy import ndarray
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torchio import RandomFlip, RandomAffine, RandomElasticDeformation, RandomNoise, SubjectsDataset, SubjectsLoader, \
    Resample, ZNormalization

from library.classifier import MriClassifier
from library.dataset import MriDataset
from library.db_module import Db
from library.queries import get_patients_with_head_mri_images

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

TRANSFORM_PIPELINE = torchio.Compose([
        Resample((1.0, 1.0, 1.0)),  # Resample to isotropic spacing
        ZNormalization(),
        RandomFlip(axes=('lf',), flip_probability=0.5),  # Left-right flipping
        RandomAffine(degrees=15, scales=(0.9, 1.1)),  # Rotation and scaling
        RandomElasticDeformation(num_control_points=7, max_displacement=(7, 7, 7)),  # Elastic deformation
        RandomNoise(mean=0, std=0.05)  # Add Gaussian noise
    ])

def get_image_ids(json_dir: str, patient_data: list[dict]) -> list[dict]:
    final_results = []

    for result in patient_data:
        image_id = None
        first_name = result['FIRST'].replace(" ", "_")
        last_name = result['LAST'].replace(" ", "_")
        full_filepath = os.path.join(json_dir, f"{first_name}_{last_name}_{result['Id']}.json")
        if os.path.exists(full_filepath):
            patient_data = json.load(open(full_filepath))
        else:
            print(f"Patient {first_name} {last_name} does not have FHIR data at {full_filepath}")
            if not os.path.exists(json_dir):
                print(f"Directory {json_dir} does not exist")
            continue
        for resource in patient_data['entry']:
            if resource['resource']['resourceType'] == 'ImagingStudy':
                if resource['resource']['series'][0]['instance'][0]['title'] == "MRI Image of Brain":
                    image_id = resource['resource']['identifier'][0]['value'][8:]
                    break
        if image_id is None:
            continue
        result['FIRST'] = first_name
        result['LAST'] = last_name
        result['image_id'] = image_id
        final_results.append(result)

    return final_results

def train_evaluate_model(train_dataset: SubjectsDataset, test_dataset: SubjectsDataset, batch_size: int,
                         epochs: int, checkpoint_dir: str, checkpoint_file: str = None) -> tuple[MriClassifier, ndarray]:
    # Create dataloaders for training and testing
    train_loader = SubjectsLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = SubjectsLoader(test_dataset, batch_size=batch_size, shuffle=False)
    start_epoch = 0
    start_batch = -1
    model = MriClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    if checkpoint_file != None:
        if os.path.exists(checkpoint_file):
            # Checkpoint format:
            # checkpoint_{epoch_num}_{batch_num}.pth     0-indexed for both
            match = re.match(r"checkpoint_(\d+)_(\d+).pth", checkpoint_file)

            if match:
                start_epoch = int(match.group(1))
                start_batch = int(match.group(2))
                checkpoint = torch.load(checkpoint_file)
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print("Invalid checkpoint file format. Starting from scratch")
        else:
            print("Checkpoint file invalid. Starting from scratch.")
    
    loss = torch.nn.BCEWithLogitsLoss()

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        for i, (subjects_batch) in enumerate(train_loader):
            while (i <= start_batch):
                continue
            print(f"Batch {i + 1}/{len(train_loader)} for epoch {epoch + 1}/{epochs}")
            optimizer.zero_grad()
            images = subjects_batch['mri'][torchio.DATA].to(DEVICE)
            labels = torch.tensor(subjects_batch['stroke_flag']).to(DEVICE)
            outputs = model(images)
            loss_val = loss(outputs, labels.unsqueeze(1).float())
            loss_val.backward()
            optimizer.step()
            train_loss += loss_val.item()
            if i % 5 == 0:
                if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
                    check = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                    check_file = checkpoint_dir.rstrip('/') + '/checkpoint_' + str(epoch) + '_' +  str(i) + '.pth'
                    torch.save(check, check_file)
        start_batch = -1
        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss}")

    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for i, (subjects_batch) in enumerate(test_loader):
            images = subjects_batch['mri'][torchio.DATA].to(DEVICE)
            labels = torch.tensor(subjects_batch['stroke_flag']).to(DEVICE)
            outputs = model(images)
            predicted = torch.round(torch.sigmoid(outputs))  # Assuming binary classification
            all_predictions.extend(predicted.cpu().numpy().flatten())  # Flatten to 1D array
            all_labels.extend(labels.cpu().numpy())  # Collect ground truth labels

    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(conf_matrix)

    return model, conf_matrix

def create_datasets(db_fpath: str, dicom_dir: str, json_dir: str, test_split: float) -> tuple[SubjectsDataset, SubjectsDataset]:
    print("Querying patient data from database")
    with Db(db_fpath) as db:
        query_results = db.query(get_patients_with_head_mri_images())
    print()
    print("Getting image IDs from JSON files")
    final_results = get_image_ids(json_dir, query_results)

    print()
    print("Splitting data into training and testing sets")
    labels: list[int] = [result['stroke_flag'] for result in final_results]
    x_train, x_test, y_train, y_test = train_test_split(final_results, labels, test_size=test_split,
                                                        random_state=42, stratify=labels)
    train_dataset = MriDataset(x_train, dicom_dir, TRANSFORM_PIPELINE).to_subject_dataset()
    test_dataset = MriDataset(x_test, dicom_dir).to_subject_dataset()

    return train_dataset, test_dataset

def create_datasets_and_model(db_fpath: str, dicom_dir: str, json_dir: str, checkpoint_dir: str, checkpoint_file: str = None, test_split: float = 0.2,
                              epochs: int = 10, batch_size: int = 4) -> tuple[SubjectsDataset, SubjectsDataset, MriClassifier, ndarray]:
    train_dataset, test_dataset = create_datasets(db_fpath, dicom_dir, json_dir, test_split)

    print()
    print("Training and evaluating model")
    model, cf_matrix = train_evaluate_model(train_dataset, test_dataset, batch_size, epochs, checkpoint_dir, checkpoint_file)

    return train_dataset, test_dataset, model, cf_matrix

def test_model(db_fpath: str, dicom_dir: str, json_dir: str, model_file: str, batch_size: int = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MriClassifier().to(device)
    model.load_state_dict(torch.load("model_file.pth"))
    model.eval()

    train_dataset, test_dataset = create_datasets(db_fpath, dicom_dir, json_dir, test_split)

    test_loader = SubjectsLoader(test_dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for i, (subjects_batch) in enumerate(test_loader):
            images = subjects_batch['mri'][torchio.DATA].to(DEVICE)
            labels = torch.tensor(subjects_batch['stroke_flag']).to(DEVICE)
            outputs = model(images)
            print(outputs)
            print(labels)
            return -1
            predicted = torch.max(outputs, 1)  # Assuming binary classification
            all_predictions.extend(predicted.cpu().numpy().flatten())  # Flatten to 1D array
            all_labels.extend(labels.cpu().numpy())  # Collect ground truth labels

if __name__ == "__main__":
    sqlite_fpath = "/Users/arodriguez/Desktop/FA24-high-risk-project-ai-healthcare/db_dir/coherent_data.db"
    mri_dir = "/Users/arodriguez/Downloads/coherent-11-07-2022/dicom"
    fhir_dir = "/Users/arodriguez/Downloads/coherent-11-07-2022/fhir"
    checkpoint_dir = "/Users/arodriguez/Desktop/FA24-high-risk-project-ai-healthcare/checkpoints"
    create_datasets_and_model(sqlite_fpath, mri_dir, fhir_dir, checkpoint_dir, epochs=1)