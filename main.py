import os

# Récuper les données
os.system('python recup_data.py')

# Exécuter le script de clustering
os.system('python clustering.py')

# Exécuter le script de sauvegarde dans la base de données
os.system('insertClustersDWH.py')
