import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from PIL import Image
from torchvision import transforms

def transform_image(image) -> torch.Tensor:
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),  # Convertit l'image PIL en tensor et la normalise entre 0 et 1
        # Vous pouvez ajouter d'autres transformations ici, par exemple :
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    return transform_pipeline(image)

def read_image_with_torch(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    #array_image = np.array(image)
    #return torch.from_numpy(array_image)
    return image

def read_images_labels(annotations_file: str, train: bool) -> pd.DataFrame:
    img_labels = pd.read_csv(annotations_file)
    img_labels = img_labels[img_labels["train"] == train]
    return img_labels

class AnimalsDataset(Dataset):
    
    def __init__(self, annotations_file, images_folder, train, transform=None, target_transform=None):
        self.img_labels = read_images_labels(annotations_file, train)
        self.images_folder = images_folder
        self.transform = transform
        self.target_transform = target_transform
        
        # mapping label -> entier
        unique_labels = self.img_labels.iloc[:, 1].unique()
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
   
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Récupérer le chemin relatif depuis le CSV et supprimer les espaces superflus
        relative_path = self.img_labels.iloc[idx, 0].strip()
        label = self.img_labels.iloc[idx, 1]
    
        # Définir le préfixe à retirer (celui présent dans le CSV)
        prefix = os.path.normpath("data/animals")
    
        # Normaliser le chemin relatif du CSV
        relative_norm = os.path.normpath(relative_path)
    
        # Si le chemin relatif commence par le préfixe, le retirer
        if relative_norm.startswith(prefix):
            # Supprime le préfixe et le séparateur éventuel
            relative_norm = relative_norm[len(prefix):].lstrip(os.sep)
    
        # Concaténer le dossier des images avec le chemin nettoyé
        img_path = os.path.join(self.images_folder, relative_norm)
    
        # Lire l'image depuis le chemin construit
        image = read_image_with_torch(img_path)
    
        if self.transform:
            image = self.transform(image)
    
        if self.target_transform:
            label = self.target_transform(label)
            
        label_idx = self.label_to_idx[label]
        label_idx = torch.tensor(label_idx, dtype=torch.long)
    
        return image, label_idx

