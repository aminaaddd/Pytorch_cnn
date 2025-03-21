from model_animals import EnsembleModel
from Dataset_animals import AnimalsDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
from datetime import datetime

def evaluate(net, test_dataloader, device):
    net.eval()
    total_correct = 0
    total_loss = 0.0
    total_images = 0
    with torch.no_grad():
        for x, y in tqdm(test_dataloader, desc="Évaluation"):
            x = x.to(device)
            y = y.to(device)
            y_pred = net(x)
            loss = F.cross_entropy(y_pred, y, reduction="sum").item()
            total_loss += loss
            preds = y_pred.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_images += y.size(0)
    accuracy = total_correct / total_images
    avg_loss = total_loss / total_images
    print(f"Loss éval.: {avg_loss:.4f}  |  Accuracy: {accuracy:.4f}")
    net.train()
    return accuracy, avg_loss

if __name__ == "__main__":
    # Sélection du device (GPU si disponible, sinon CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Utilisation du device:", device)
    
    # Hyperparamètres
    learning_rate = 0.001
    batch_size = 64
    epochs = 20
    num_classes = 10  # Adaptez selon votre dataset
    
    # Création d'un nom pour le run avec la date et l'heure
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"ensemble_animals_{current_time}"
    
    # Transformations pour les images :
    # - Redimensionnement à 224x224 pour ResNet
    # - Conversion en tenseur
    # - Si l'image n'a pas 3 canaux, on répète le canal existant pour obtenir 3 canaux
    # - Normalisation avec les moyennes et écarts-types d'ImageNet
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x if x.size(0) == 3 else x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Chemins vers vos données (à adapter)
    annotations_file = r"/home/amina/pytorch_cnn/data/animals/dataset.csv"
    images_folder = "/home/amina/pytorch_cnn/data/animals"
    
    # Création des datasets et des DataLoader 
    train_dataset = AnimalsDataset(annotations_file, images_folder, train=True, transform=data_transforms)
    test_dataset = AnimalsDataset(annotations_file, images_folder, train=False, transform=data_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # j'initialise le modèle en ensemble (ResNet + SimpleCNN) et envoi sur le device
    model = EnsembleModel(num_classes=num_classes, feature_extract=True)
    model = model.to(device)
    
    # Optimiseur et loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    # Initialisation du SummaryWriter pour TensorBoard
    summary_writer = SummaryWriter(log_dir=f"./runs/{path}")
    
    # entraînement
    for epoch in range(epochs):
        model.train()
        accumulated_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, (x, y) in enumerate(progress_bar):
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            
            accumulated_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs} - Loss: {accumulated_loss/(i+1):.4f}")
        
        avg_train_loss = accumulated_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}: Loss entraînement: {avg_train_loss:.4f}")
        
        # Évaluation 
        eval_acc, eval_loss = evaluate(model, test_dataloader, device)
        
        # Journalisation dans TensorBoard
        summary_writer.add_scalars("Loss", {"train": avg_train_loss, "eval": eval_loss}, epoch+1)
        summary_writer.add_scalars("Accuracy", {"eval": eval_acc}, epoch+1)
        
        # Je sauvegarde le modèle après chaque époque
        torch.save(model.state_dict(), f"./runs/{path}/model_epoch_{epoch+1}.pth")
    
    summary_writer.close()
