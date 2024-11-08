import pandas as pd
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import scipy.stats as stats
import numpy as np
from sklearn.metrics import silhouette_score

# Charger les données
df = pd.read_csv('DataSetClients.csv', sep=';', encoding='iso-8859-1')

# Supprimer les lignes avec des valeurs négatives ou égales à zéro ou inférieures à 5000 dans 'PRIMETOTALE'
df = df[df['PRIMETOTALE'] > 5000]

# Sauvegarder la colonne CODEASSU avant les filtrages
codeassu_original = df['CODEASSU'].copy()

# Supprimer les colonnes non nécessaires
df_reduc = df.drop(columns=['CODEASSU'])

# Convertir les colonnes aux types appropriés
df_reduc['PRIMETOTALE'] = df_reduc['PRIMETOTALE'].astype(float)
df_reduc['PRIMENETTE'] = df_reduc['PRIMENETTE'].astype(float)
df_reduc['CODEINTE'] = df_reduc['CODEINTE'].astype(str)
df_reduc['CODECATE'] = df_reduc['CODECATE'].astype(str)

# Supprimer les lignes avec des valeurs manquantes dans 'CODECATE'
df_reduc = df_reduc.dropna(subset=['CODECATE'])

# Supprimer les doublons
df_reduc = df_reduc.drop_duplicates()

# Vérifier les valeurs manquantes
print("Valeurs manquantes par colonne :")
print(df_reduc.isnull().sum())

# Vérifier les types de données des colonnes
print("Types de données des colonnes :")
print(df_reduc.dtypes)

# Supprimer les doublons
print("Nombre de doublons :")
print(df_reduc.duplicated().sum())

# Décrire les statistiques des colonnes numériques pour vérifier les valeurs aberrantes
print("Statistiques descriptives des colonnes numériques :")
print(df_reduc.describe())

# Visualisation de l'asymétrie (skewness) de PRIMETOTALE
sns.histplot(df_reduc['PRIMETOTALE'], kde=True)
plt.title('Distribution de PRIMETOTALE')
plt.show()

# Appliquer une transformation logarithmique pour réduire l'impact des valeurs extrêmes (outliers)
df_reduc['PRIMETOTALE_log'] = np.log1p(df_reduc['PRIMETOTALE'])
df_reduc['PRIMENETTE_log'] = np.log1p(df_reduc['PRIMENETTE'])

# Calculer la matrice de corrélation pour les variables numériques
corr_matrix = df_reduc[['PRIMETOTALE_log', 'PRIMENETTE_log']].corr()

# Afficher la matrice de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matrice de corrélation')
plt.show()

# Supprimer PRIMENETTE après la matrice de corrélation
df_reduc = df_reduc.drop(columns=['PRIMENETTE', 'PRIMENETTE_log'])

# Standardiser la variable numérique PRIMETOTALE
scaler = StandardScaler()
df_reduc[['PRIMETOTALE_scaled']] = scaler.fit_transform(df_reduc[['PRIMETOTALE_log']])

# Préparer les données pour l'algorithme k-prototype
data = df_reduc.copy()

# Supprimer la colonne originale PRIMETOTALE et PRIMETOTALE_log pour éviter les doublons dans l'algorithme
data = data.drop(columns=['PRIMETOTALE', 'PRIMETOTALE_log'])

# Extraire les indices des colonnes catégorielles
categorical_columns = [data.columns.get_loc(col) for col in list(data.select_dtypes('object').columns)]
print('Colonnes catégorielles :', list(data.select_dtypes('object').columns))
print('Position des colonnes catégorielles :', categorical_columns)

# Convertir les données en numpy array pour le clustering
data = data.to_numpy()

# Calculer les distorsions et les coefficients de silhouette pour différentes valeurs de k
distortions = []
silhouette_scores = []  # Stocker les coefficients de silhouette
K = range(2, 10)  # Tester des valeurs de k de 2 à 9
for k in K:
    try:
        kproto = KPrototypes(n_jobs=-1, n_clusters=k, init='Huang', n_init=10, random_state=0, verbose=1)
        clusters = kproto.fit_predict(data, categorical=categorical_columns)
        distortions.append(kproto.cost_)
        
        # Calcul du Silhouette Score pour chaque K
        silhouette_avg = silhouette_score(data, clusters, metric='euclidean')
        silhouette_scores.append(silhouette_avg)
        print(f"Silhouette Score for k={k}: {silhouette_avg}")
        
    except ValueError as e:
        print(f"Error for k={k}: {e}")
        distortions.append(None)

# Filtrer les valeurs None et s'assurer que valid_K et valid_distortions sont de même longueur
valid_K = [k for k, d in zip(K, distortions) if d is not None]
valid_distortions = [d for d in distortions if d is not None]
valid_silhouette_scores = [s for k, s in zip(K, silhouette_scores) if s is not None]

# Tracer la courbe des distorsions
plt.figure(figsize=(8, 6))
plt.plot(valid_K, valid_distortions, 'bx-')
plt.xlabel('Nombre de clusters K')
plt.ylabel('Distorsion')
plt.title('La méthode du coude pour déterminer le nombre optimal de clusters')
plt.show()

# Récapitulatif des Silhouette Scores
print("\nRécapitulatif des Silhouette Scores pour chaque K :")
for k, s in zip(valid_K, valid_silhouette_scores):
    print(f"K={k}: Silhouette Score = {s}")

# Trouver le K avec le meilleur Silhouette Score
optimal_k = valid_K[np.argmax(valid_silhouette_scores)]
print(f"Le K optimal selon le Silhouette Score est : {optimal_k}")

# Demander à l'utilisateur d'entrer le K optimal
optimal_k = int(input("Veuillez entrer le K optimal basé sur le Silhouette Score ou la courbe du coude : "))



# Exécuter l'algorithme k-prototype avec le nombre optimal de clusters
kproto = KPrototypes(n_clusters=optimal_k, init='Huang', n_init=20, verbose=1, gamma=0.5)

clusters = kproto.fit_predict(data, categorical=categorical_columns)

# Ajouter les clusters au dataframe original
df_reduc['Cluster'] = clusters

# Réintégrer la colonne CODEASSU avec les mêmes lignes restantes
df_reduc['CODEASSU'] = codeassu_original[df_reduc.index].values

# Étape 1: Restaurer les valeurs standardisées à partir de PRIMETOTALE_scaled
df_reduc['PRIMETOTALE_scaled_inverse'] = scaler.inverse_transform(df_reduc[['PRIMETOTALE_scaled']])

# Étape 2: Appliquer l'inverse de la transformation logarithmique pour retrouver les valeurs originales de PRIMETOTALE
df_reduc['PRIMETOTALE'] = np.expm1(df_reduc['PRIMETOTALE_scaled_inverse'])

# Étape 3: Arrondir les valeurs de PRIMETOTALE
df_reduc['PRIMETOTALE'] = df_reduc['PRIMETOTALE'].round(2)

# Supprimer les colonnes inutiles
df_reduc = df_reduc.drop(columns=['PRIMETOTALE_scaled', 'PRIMETOTALE_scaled_inverse', 'PRIMETOTALE_log'])


# Afficher les premières lignes avec les valeurs d'origine restaurées et les clusters
print(df_reduc[['CODEINTE', 'CODECATE', 'PRIMETOTALE', 'Cluster']].head())

# Visualiser le nombre de clients par cluster sous forme de graphique en barres
df_reduc['Cluster'].value_counts().plot(kind='bar')

# Calculer la moyenne uniquement pour les colonnes numériques par cluster
numeric_columns = df_reduc.select_dtypes(include=[np.number])
print("Moyennes des colonnes numériques par cluster :")
print(df_reduc.groupby('Cluster')[numeric_columns.columns].mean())

# Visualiser les clusters avec PRIMETOTALE et Cluster
plt.figure(figsize=(8, 6))
plt.scatter(df_reduc['PRIMETOTALE'], df_reduc['Cluster'], c=df_reduc['Cluster'], cmap='viridis')
plt.title(f'Visualisation des clusters K-Prototypes avec PRIMETOTALE')
plt.xlabel('PRIMETOTALE')
plt.ylabel('Cluster')
plt.colorbar()
plt.show()


# Calculer les valeurs les plus fréquentes (mode) des colonnes catégorielles par cluster
print("Mode des colonnes catégorielles par cluster :")
print(df_reduc.groupby(['Cluster']).agg(lambda x: pd.Series.mode(x).iat[0])[['CODEINTE', 'CODECATE']])

# Analyser les valeurs uniques des colonnes catégorielles par cluster
print("\nValeurs uniques des colonnes catégorielles par cluster :")
for col in df.select_dtypes(include=['object']).columns:
    for cluster_num in range(optimal_k):
        print(f"Valeurs uniques de la colonne {col} dans le cluster {cluster_num} :")
        print(df_reduc[df_reduc['Cluster'] == cluster_num][col].unique())
        print()


# Enregistrer les résultats dans un fichier CSV
df_reduc.to_csv('ClustersDataSetClientsNewk2.csv', index=False)