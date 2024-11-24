import json
import os

import numpy as np
import torch
import torchio
from numpy import ndarray
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torchio import RandomFlip, RandomAffine, RandomElasticDeformation, RandomNoise, SubjectsDataset, SubjectsLoader

from library.classifier import MriClassifier
from library.dataset import MriDataset
from library.db_module import Db
from library.queries import get_patients_with_head_mri_images

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

TRANSFORM_PIPELINE = torchio.Compose([
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
        try:
            patient_data = json.load(open(os.path.join(json_dir, f"{first_name}_{last_name}_{result['Id']}.json")))
        except FileNotFoundError:
            print(f"Patient {first_name} {last_name} does not have FHIR data")
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
                         epochs: int) -> tuple[MriClassifier, ndarray]:
    # Create dataloaders for training and testing
    train_loader = SubjectsLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = SubjectsLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MriClassifier().to(DEVICE)
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, (subjects_batch) in enumerate(train_loader):
            print(f"Batch {i + 1}/{len(train_loader)} for epoch {epoch + 1}/{epochs}")
            optimizer.zero_grad()
            images = subjects_batch['mri'][torchio.DATA].to(DEVICE)
            labels = torch.tensor(subjects_batch['stroke_flag']).to(DEVICE)
            outputs = model(images)
            loss_val = loss(outputs, labels.unsqueeze(1).float())
            loss_val.backward()
            optimizer.step()
            train_loss += loss_val.item()
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
                                                        random_state=42)
    train_dataset = MriDataset(x_train, dicom_dir, TRANSFORM_PIPELINE).to_subject_dataset()
    test_dataset = MriDataset(x_test, dicom_dir).to_subject_dataset()

    return train_dataset, test_dataset

def create_datasets_and_model(db_fpath: str, dicom_dir: str, json_dir: str, test_split: float = 0.2,
         epochs: int = 10, batch_size: int = 4) -> tuple[SubjectsDataset, SubjectsDataset, MriClassifier, ndarray]:
    train_dataset, test_dataset = create_datasets(db_fpath, dicom_dir, json_dir, test_split)

    print()
    print("Training and evaluating model")
    model, cf_matrix = train_evaluate_model(train_dataset, test_dataset, batch_size, epochs)

    return train_dataset, test_dataset, model, cf_matrix


if __name__ == "__main__":
    sqlite_fpath = "/Users/arodriguez/Desktop/FA24-high-risk-project-ai-healthcare/db_dir/coherent_data.db"
    mri_dir = "/Users/arodriguez/Downloads/coherent-11-07-2022/dicom"
    fhir_dir = "/Users/arodriguez/Downloads/coherent-11-07-2022/fhir"
    create_datasets_and_model(sqlite_fpath, mri_dir, fhir_dir, epochs=1)