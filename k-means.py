import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Fonction pour appliquer K-Means à une image et segmenter en deux clusters
def kmeans_segmentation(image, n_clusters=2):
    # Vérifier si l'image est en float64, et la convertir en uint8
    if image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    # Redimensionner l'image pour accélérer le traitement
    image_resized = cv2.resize(image, (128, 128))

    # Convertir l'image en RGB (OpenCV utilise BGR par défaut)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Aplatir l'image en une liste de pixels (chaque pixel est un point de données)
    pixels = image_rgb.reshape((-1, 3))

    # Appliquer K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels)

    # Obtenir les labels des clusters pour chaque pixel
    segmented_image = kmeans.labels_.reshape(image_resized.shape[:2])

    return segmented_image, image_rgb

# Visualisation de quelques résultats de segmentation pour le train set
def visualize_segmentation(X, y, num_images=5):
    plt.figure(figsize=(15, 15))

    for i in range(num_images):
        # Sélectionner une image du dataset
        image = X[i]

        # Appliquer K-Means pour la segmentation
        segmented_image, original_image = kmeans_segmentation(image)

        # Vérifier si l'image est bien lue
        if original_image is None or segmented_image is None:
            print(f"Erreur: L'image {i+1} n'a pas été correctement lue.")
            continue

        # Afficher l'image originale et l'image segmentée
        plt.subplot(num_images, 2, 2*i + 1)
        plt.imshow(original_image)
        plt.title(f"Image {i+1} - Originale")
        plt.axis('off')

        plt.subplot(num_images, 2, 2*i + 2)
        plt.imshow(segmented_image, cmap='viridis')
        plt.title(f"Image {i+1} - Segmentée")
        plt.axis('off')

    plt.show()

# Charger les données sauvegardées
X_train = np.load(r"C:\Users\Admin\Desktop\MGSI\MGSI 4\PROJET(like pfa) TRAITEMENT & SEGMENTATION D IMAGES-MALADIES RENALES\X_train.npy")

y_train = np.load(r"C:\Users\Admin\Desktop\MGSI\MGSI 4\PROJET(like pfa) TRAITEMENT & SEGMENTATION D IMAGES-MALADIES RENALES\y_train.npy")


visualize_segmentation(X_train, y_train, num_images=5)  # Afficher les résultats pour 5 images du dataset d'entraînement
