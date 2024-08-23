# Importación de librerías necesarias
import itk  # Toolkit de procesamiento de imágenes médicas
import numpy as np  # Biblioteca para operaciones numéricas
import pydicom  # Biblioteca para manejar archivos DICOM
from sklearn.preprocessing import StandardScaler  # Para normalizar datos
from sklearn.cluster import KMeans  # Algoritmo de clustering K-means
from sklearn.mixture import GaussianMixture  # Algoritmo de clustering GMM
from sklearn.decomposition import PCA  # Para reducción de dimensionalidad
from sklearn.metrics import silhouette_score  # Métrica para evaluar clustering
import matplotlib.pyplot as plt  # Para visualización
import os  # Para operaciones del sistema de archivos
import traceback  # Para manejo detallado de excepciones
import csv  # Para manejar archivos CSV

# Definición de la clase para extraer características de textura
class ImprovedTextureFeatureExtractor:
    def __init__(self, image_path, mask_path=None):
        # Inicializa el extractor con la ruta de la imagen y opcionalmente una máscara
        self.image_path = image_path
        self.mask_path = mask_path

    def extract_features(self, num_bins=8, pixel_value_min=0, pixel_value_max=255, neighborhood_radius=1):
        # Configura los parámetros para la extracción de características
        Dimension = 2
        InputPixelType = itk.ctype('signed short')
        InputImageType = itk.Image[InputPixelType, Dimension]

        # Lee la imagen DICOM
        dicom = pydicom.dcmread(self.image_path)
        np_array = dicom.pixel_array.astype(np.float32)
        itk_image = itk.GetImageFromArray(np_array)

        # Maneja la máscara si se proporciona
        if self.mask_path: # Definimos la ruta de la máscara
            MaskPixelType = itk.ctype('unsigned char') #definimos el tipo de pixel 
            MaskImageType = itk.Image[MaskPixelType, Dimension] #Definimos la imagen de la máscara 
            maskReader = itk.ImageFileReader[MaskImageType].New()
            maskReader.SetFileName(self.mask_path)
            mask = maskReader.GetOutput()
        else:
            mask = None

        # Configura el filtro de características de textura
        filtr = itk.CoocurrenceTextureFeaturesImageFilter.New(itk_image)
        if mask:
            filtr.SetMaskImage(mask)
        filtr.SetNumberOfBinsPerAxis(num_bins)
        filtr.SetHistogramMinimum(pixel_value_min)
        filtr.SetHistogramMaximum(pixel_value_max)
        filtr.SetNeighborhoodRadius([neighborhood_radius, neighborhood_radius])

        # Ejecuta el filtro
        filtr.Update()

        # Obtiene el resultado como un array de NumPy
        result = itk.GetArrayFromImage(filtr.GetOutput())

        return result

    def analyze_features(self, features):
        # Analiza y guarda estadísticas de las características extraídas
        results = []
        results.append("Estadísticas de las características de textura:")
        results.append(f"  Forma: {features.shape}")
        results.append(f"  Tipo de datos: {features.dtype}")
        results.append(f"  Rango de valores: [{np.min(features)}, {np.max(features)}]")
        
        # Calcula estadísticas para cada característica
        for i in range(features.shape[-1]):
            feature = features[..., i].flatten()
            results.append(f"  Característica {i}:")
            results.append(f"    Rango: [{np.min(feature)}, {np.max(feature)}]")
            results.append(f"    Media: {np.mean(feature)}")
            results.append(f"    Desviación estándar: {np.std(feature)}")
        
        # Imprime los resultados en la consola
        for line in results:
            print(line)
        
        # Guarda los resultados en un archivo
        with open('resultados/feature_statistics.txt', 'w') as f:
            for line in results:
                f.write(line + '\n')

# Definición de la clase para realizar clustering de textura
class TextureClustering:
    def __init__(self, features):
        # Inicializa el objeto de clustering con las características extraídas
        self.features = features
        self.n_rows, self.n_cols, self.n_features = features.shape
        self.flattened_features = self.features.reshape(-1, self.n_features)

    def preprocess(self):
        # Normaliza las características
        scaler = StandardScaler()
        self.normalized_features = scaler.fit_transform(self.flattened_features)

    def apply_pca(self, n_components=2):
        # Aplica PCA para reducir la dimensionalidad
        self.pca = PCA(n_components=n_components)
        self.pca_features = self.pca.fit_transform(self.normalized_features)
        variance_explained = np.sum(self.pca.explained_variance_ratio_)
        print(f"Varianza explicada por los primeros {n_components} componentes: {variance_explained:.2f}")
        return variance_explained

    def cluster_kmeans(self, n_clusters=5):
        # Aplica el algoritmo K-means
        if n_clusters < 2:
            print("Warning: K-means requiere al menos 2 clusters. Usando 2 clusters.")
            n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_labels = kmeans.fit_predict(self.normalized_features)

    def cluster_gmm(self, n_components=5):
        # Aplica el algoritmo GMM
        if n_components < 1:
            print("Warning: GMM requiere al menos 1 componente. Usando 1 componente.")
            n_components = 1
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        self.cluster_labels = gmm.fit_predict(self.normalized_features)
        self.gmm_model = gmm

    def evaluate_clustering(self):
        # Evalúa el clustering usando el coeficiente de silueta
        if len(np.unique(self.cluster_labels)) < 2:
            print("Warning: Se necesitan al menos 2 clusters para calcular el coeficiente de silueta.")
            return None
        silhouette_avg = silhouette_score(self.normalized_features, self.cluster_labels)
        return silhouette_avg

    def visualize_clusters(self, title='Clusters de textura', filename='clusters.png'):
        # Visualiza los clusters
        cluster_image = self.cluster_labels.reshape(self.n_rows, self.n_cols)
        plt.figure(figsize=(10, 10))
        plt.imshow(cluster_image, cmap='viridis')
        plt.title(title)
        plt.colorbar(label='Cluster')
        plt.savefig(filename)
        plt.close()

    def visualize_pca(self, title='Visualización PCA de clusters', filename='pca_visualization.png'):
        # Visualiza los resultados de PCA
        if not hasattr(self, 'pca_features'):
            raise ValueError("PCA no ha sido aplicado. Llame a apply_pca() primero.")

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(self.pca_features[:, 0], self.pca_features[:, 1], c=self.cluster_labels, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel('Primer componente principal')
        plt.ylabel('Segundo componente principal')
        plt.savefig(filename)
        plt.close()

    def visualize_gmm_contours(self, title='Contornos GMM', filename='gmm_contours.png'):
        # Visualiza los contornos del modelo GMM
        if not hasattr(self, 'pca_features') or not hasattr(self, 'gmm_model') or not hasattr(self, 'pca'):
            raise ValueError("PCA y GMM deben ser aplicados primero.")

        x = self.pca_features[:, 0]
        y = self.pca_features[:, 1]
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(x, y, c=self.cluster_labels, cmap='viridis', alpha=0.5)
        
        # Crea una malla en el espacio PCA
        xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 100),
                            np.linspace(y.min(), y.max(), 100))
        XY_pca = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Transforma la malla PCA de vuelta al espacio original
        XY_original = self.pca.inverse_transform(XY_pca)
        
        # Predice las probabilidades usando el modelo GMM
        Z = self.gmm_model.predict_proba(XY_original)
        Z = Z.reshape(xx.shape + (self.gmm_model.n_components,))
        
        # Dibuja contornos solo si hay más de un componente
        if self.gmm_model.n_components > 1:
            for i in range(self.gmm_model.n_components):
                plt.contour(xx, yy, Z[:, :, i], levels=[0.5], colors='k', linewidths=0.5)
        
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel('Primer componente principal')
        plt.ylabel('Segundo componente principal')
        plt.savefig(filename)
        plt.close()

def plot_silhouette_comparison(cluster_numbers, csv_file):
    # Crea un gráfico comparando los coeficientes de silueta de K-means y GMM
    kmeans_scores = []
    gmm_scores = []
    
    with open(csv_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Salta la cabecera
        for row in csvreader:
            if row[0] == 'K-means':
                kmeans_scores.append(float(row[2]))
            elif row[0] == 'GMM':
                gmm_scores.append(float(row[2]))
    
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_numbers, kmeans_scores, marker='o', label='K-means')
    plt.plot(cluster_numbers, gmm_scores, marker='s', label='GMM')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Coeficiente de Silueta')
    plt.title('Comparación de Coeficientes de Silueta')
    plt.legend()
    plt.grid(True)
    plt.savefig('resultados/silhouette_comparison.png')
    plt.close()

def create_composite_image(cluster_numbers):
    # Crea una imagen compuesta con todos los resultados de clustering
    fig, axs = plt.subplots(len(cluster_numbers), 3, figsize=(20, 6*len(cluster_numbers)))
    fig.suptitle("Resultados de Clustering para Diferentes Números de Clusters", fontsize=16)

    for i, n_clusters in enumerate(cluster_numbers):
        # K-means clusters
        kmeans_img = plt.imread(f'resultados/kmeans_clusters_{n_clusters}.png')
        axs[i, 0].imshow(kmeans_img)
        axs[i, 0].set_title(f'K-means (n={n_clusters})')
        axs[i, 0].axis('off')

        # GMM clusters
        gmm_img = plt.imread(f'resultados/gmm_clusters_{n_clusters}.png')
        axs[i, 1].imshow(gmm_img)
        axs[i, 1].set_title(f'GMM (n={n_clusters})')
        axs[i, 1].axis('off')

        # GMM contours
        gmm_contours_img = plt.imread(f'resultados/gmm_contours_{n_clusters}.png')
        axs[i, 2].imshow(gmm_contours_img)
        axs[i, 2].set_title(f'GMM Contornos (n={n_clusters})')
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('resultados/composite_clustering_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        # Crea el directorio para guardar los resultados
        os.makedirs('resultados', exist_ok=True)

        # Define la ruta de la imagen DICOM
        image_path = 'images_dicom/IM-0001-0001.dcm'
        
        # Crea el extractor de características
        extractor = ImprovedTextureFeatureExtractor(image_path)
        
        # Extrae características
        features = extractor.extract_features(num_bins=16, pixel_value_min=0, pixel_value_max=3079, neighborhood_radius=2)
        
        # Analiza las características y guarda resultados
        extractor.analyze_features(features)

        # Guarda las características
        np.save('resultados/texture_features.npy', features)
        print("Características guardadas en 'resultados/texture_features.npy'")

        # Crea una instancia de TextureClustering
        tc = TextureClustering(features)

        # Preprocesa las características
        tc.preprocess()

        # Aplica PCA y guarda resultados
        variance_explained = tc.apply_pca(n_components=2)
        with open('resultados/pca_results.txt', 'w') as f:
            f.write(f"Varianza explicada por los primeros 2 componentes: {variance_explained:.2f}\n")

        # Experimenta con diferentes números de clusters/componentes
        cluster_numbers = [2, 5, 10, 20, 30]
        
        # Abre un archivo CSV para guardar los resultados
        with open('resultados/clustering_results.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Método', 'Número de Clusters', 'Coeficiente de Silueta'])

            for n_clusters in cluster_numbers:
                print(f"\nExperimentando con {n_clusters} clusters/componentes:")
                
                # K-means
                print("K-means:")
                tc.cluster_kmeans(n_clusters=n_clusters)
                tc.visualize_clusters(title=f'K-means: Clusters de textura (n={n_clusters})',
                                      filename=f'resultados/kmeans_clusters_{n_clusters}.png')
                tc.visualize_pca(title=f'K-means: Visualización PCA (n={n_clusters})',
                                 filename=f'resultados/kmeans_pca_{n_clusters}.png')
                silhouette_avg_kmeans = tc.evaluate_clustering()
                if silhouette_avg_kmeans is not None:
                    print(f"K-means - Coeficiente de silueta promedio: {silhouette_avg_kmeans:.4f}")
                    csvwriter.writerow(['K-means', n_clusters, silhouette_avg_kmeans])

                # GMM
                print("\nGaussian Mixture Model:")
                tc.cluster_gmm(n_components=n_clusters)
                tc.visualize_clusters(title=f'GMM: Clusters de textura (n={n_clusters})',
                                      filename=f'resultados/gmm_clusters_{n_clusters}.png')
                tc.visualize_pca(title=f'GMM: Visualización PCA (n={n_clusters})',
                                 filename=f'resultados/gmm_pca_{n_clusters}.png')
                tc.visualize_gmm_contours(title=f'GMM: Contornos (n={n_clusters})',
                                          filename=f'resultados/gmm_contours_{n_clusters}.png')
                silhouette_avg_gmm = tc.evaluate_clustering()
                if silhouette_avg_gmm is not None:
                    print(f"GMM - Coeficiente de silueta promedio: {silhouette_avg_gmm:.4f}")
                    csvwriter.writerow(['GMM', n_clusters, silhouette_avg_gmm])

# Visualización adicional: Gráfico de líneas para comparar coeficientes de silueta
        plot_silhouette_comparison(cluster_numbers, 'resultados/clustering_results.csv')

        # Generar una imagen compuesta de todos los resultados
        create_composite_image(cluster_numbers)

        print("\nAnálisis completado. Todos los resultados han sido guardados en el directorio 'resultados'.")

    except Exception as e:
        print(f"Se produjo un error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()