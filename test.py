import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 📂 Chemins
BASE_FOLDER = r"C:\Users\Admin\Desktop\MGSI\MGSI 4\PROJET(like pfa) TRAITEMENT & SEGMENTATION D IMAGES-MALADIES RENALES\IMPLEMENTATION"
IMAGE_FOLDER = os.path.join(BASE_FOLDER, "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")  # Dossier contenant les sous-dossiers "Cyst", "Normal", "Stone", "Tumor"
CSV_PATH = os.path.join(BASE_FOLDER, "kidneyData.csv")

# 1️⃣ Charger le CSV
df = pd.read_csv(CSV_PATH)

# 📌 Dossier contenant les images
LABELS = ["Cyst", "Normal", "Stone", "Tumor"]  # Les étiquettes des sous-dossiers

# 🏗️ Générer le vrai chemin des images
def get_image_paths():
    image_paths = []
    labels = []
    for label in LABELS:
        label_folder = os.path.join(IMAGE_FOLDER, label)  # Accéder au sous-dossier
        for filename in os.listdir(label_folder):
            if filename.endswith(".jpg"):  # Assurez-vous que ce sont des fichiers .jpg
                image_paths.append(os.path.join(label_folder, filename))
                labels.append(label)  # Ajouter l'étiquette correspondant au dossier
    return image_paths, labels

# Récupérer tous les chemins d'images et les étiquettes
image_paths, labels = get_image_paths()

# 🔄 Préparer les images (Redimensionnement et Normalisation)
IMAGE_SIZE = (128, 128)  # Taille des images

def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Lire l'image
    img = cv2.resize(img, IMAGE_SIZE)  # Redimensionner
    img = img / 255.0  # Normaliser [0,1]
    return img

# 🔄 Conversion en tableaux NumPy
X = []
y = []

for image_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
    img_array = preprocess_image(image_path)
    X.append(img_array)
    y.append(label)  # L'étiquette est la classe

X = np.array(X)
y = np.array(y)

# 📊 Séparer en Train & Test
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Convertir les étiquettes en valeurs numériques

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Dataset prêt : {X_train.shape[0]} images pour l'entraînement, {X_test.shape[0]} pour le test.")

# 💾 Sauvegarde des données
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("✅ Données sauvegardées !")
