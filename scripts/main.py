import json
import os

import torch
import torchio
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
                         epochs: int, pos_weight: float) -> None:
    # Create dataloaders for training and testing
    train_loader = SubjectsLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = SubjectsLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MriClassifier().to(DEVICE)
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, (subjects_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            images = subjects_batch['mri'][torchio.DATA].to(DEVICE)
            labels = torch.tensor(subjects_batch['stroke_flag']).to(DEVICE)
            outputs = model(images)
            loss_val = loss(outputs, labels.unsqueeze(1).float())
            loss_val.backward()
            optimizer.step()
            train_loss += loss_val.item()
        train_loss /= len(train_loader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            predicted = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
    print(f"Accuracy: {correct / total}")
    print()



def main(db_fpath: str, dicom_dir: str, json_dir: str, test_split: float = 0.1,
         epochs: int = 10, batch_size: int = 32) -> None:
    with Db(db_fpath) as db:
        query_results = db.query(get_patients_with_head_mri_images())

    final_results = get_image_ids(json_dir, query_results)

    labels: list[int] = [result['stroke_flag'] for result in final_results]

    x_train, x_test, y_train, y_test = train_test_split(final_results, labels, test_size=test_split,
                                                        random_state=42)

    train_dataset = MriDataset(x_train, dicom_dir, TRANSFORM_PIPELINE).to_subject_dataset()
    test_dataset = MriDataset(x_test, dicom_dir, TRANSFORM_PIPELINE).to_subject_dataset()

    pos_weight = (len(labels) - sum(labels)) / sum(labels)

    train_evaluate_model(train_dataset, test_dataset, batch_size, epochs, pos_weight)


if __name__ == "__main__":
    sqlite_fpath = "/Users/arodriguez/Desktop/FA24-high-risk-project-ai-healthcare/db_dir/coherent_data.db"
    mri_dir = "/Users/arodriguez/Downloads/coherent-11-07-2022/dicom"
    fhir_dir = "/Users/arodriguez/Downloads/coherent-11-07-2022/fhir"
    main(sqlite_fpath, mri_dir, fhir_dir, batch_size=8)