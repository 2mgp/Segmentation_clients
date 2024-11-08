import pandas as pd
import pyodbc

# Charger le DataFrame avec les clusters
df = pd.read_csv('ClustersDataSetClientsNew.csv')

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

# Créer une table pour stocker les données (si nécessaire)
cursor.execute("""
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='DBO_CLUSTERS' and xtype='U')
CREATE TABLE DBO_CLUSTERS (
    ID_DBOCLUS INT IDENTITY(1,1) PRIMARY KEY,
    CODEASSU INT,
    CODEINTE INT,
    CODECATE INT,
    PRIMETOTALE FLOAT,
    ClusterID INT
)
""")
conn.commit()

# Sauvegarder le DataFrame dans la table SQL
for index, row in df.iterrows():
    cursor.execute("INSERT INTO DBO_CLUSTERS ( CODEASSU, CODEINTE, CODECATE, PRIMETOTALE,ClusterID) VALUES (?,?, ?, ?, ?)",
                     row['CODEASSU'], row['CODEINTE'], row['CODECATE'],row['PRIMETOTALE'],row['Cluster'])
conn.commit()

print("Les données ont été sauvegardées avec succès dans la base de données.")

# Fermer la connexion
cursor.close()
conn.close()
