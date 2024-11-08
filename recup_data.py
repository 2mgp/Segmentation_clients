import pandas as pd
import pyodbc

# Configuration de la connexion à la base de données SQL Server
server = 'localhost'
port = '53716'
database = 'test'
username = 'sa'
password = 'intelligente20@'
driver = 'ODBC Driver 17 for SQL Server'
instance = 'SQLEXPRESS'

# Créer la chaîne de connexion
connection_string = f'DRIVER={{{driver}}};SERVER={server}\\{instance},{port};DATABASE={database};UID={username};PWD={password}'

# Établir la connexion
conn = pyodbc.connect(connection_string)
cursor = conn.cursor()

# La requête SQL pour recuperer les données
query = """
SELECT CODEASSU, CODEINTE, CODECATE, PRIMENETTE, PRIMETOTALE
FROM dbo.QUITTANCE
"""

# Exécution de la requête et chargement des résultats dans un DataFrame pandas
df = pd.read_sql(query, conn)

# Fermeture de la connexion à la base de données
cursor.close()
conn.close()

# Enregistrement des résultats dans un fichier CSV avec les en-têtes
df.to_csv('DataSetClients.csv', sep=';', index=False)
print("Les résultats ont été enregistrés dans 'DataSetClients.csv'")


