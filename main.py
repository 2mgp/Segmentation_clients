import os

# Exécuter le script de clustering
os.system('python clustering.py')

# Exécuter le script de sauvegarde dans la base de données
os.system('insert_cluster_BD.py')
