import os
from typing import List
import torchio as tio
import torch
from pydicom import dcmread


class MriDataset:
    def __init__(self, patient_data: List[dict], dicom_dir: str, transforms=None):
        self.dicom_dir = dicom_dir
        self.patients = patient_data
        self.transforms = transforms
        self.subjects = self._create_subjects()

    def _create_subjects(self) -> List[tio.Subject]:
        subjects = []
        for patient in self.patients:
            potential_filename = f"{patient['FIRST']}_{patient['LAST']}_{patient['Id']}{patient['image_id']}.dcm"
            dicom_path = os.path.join(self.dicom_dir, potential_filename)

            if not os.path.exists(dicom_path):
                print(f"Patient {patient['FIRST']} {patient['LAST']} does not have MRI images"
                                 f" at {dicom_path}")
                continue


            # Load DICOM image and convert to tensor
            ds = dcmread(dicom_path)
            image = ds.pixel_array
            image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension

            # Create TorchIO `Image` object
            mri_image = tio.ScalarImage(tensor=image_tensor)

            # Create a `Subject` with the image and metadata
            subject = tio.Subject(
                mri=mri_image,
                stroke_flag=patient['stroke_flag']
            )
            subjects.append(subject)
        return subjects

    def to_subject_dataset(self) -> tio.SubjectsDataset:
        return tio.SubjectsDataset(self.subjects, transform=self.transforms)