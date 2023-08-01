import pandas as pd

class Dataset():
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.df = None
        self.X = None
        self.y = None
        self.version = None
       
    def load(self, target):
        """Load dataset."""
        self.df = pd.read_csv(self.path)
        self.X = self.df.drop(columns=[target])
        self.y = self.df[target]

class OatsDataset():
    def __init__(self):
        self.name = 'Oats'
        self.path = 'data/processed/from_notebooks/dataset_v1.csv'
        self.version = 1
        self.df = None
        self.X = None
        self.y = None
       
    def load(self, target='Rendement'):
        """Load dataset."""
        cols = ['DAYS_WITH_VALID_SUNSHINE', 'DAYS_WITH_VALID_PRECIP', 
        'MEAN_TEMPERATURE', 'MIN_TEMPERATURE', 'DAYS_WITH_PRECIP_GE_1MM', 'TOTAL_SNOWFALL',
        'MAX_TEMPERATURE', 'TOTAL_PRECIPITATION', 'Annee', 'LATITUDE', 'LONGITUDE']
        df = pd.read_csv(self.path, index_col=0)
        df = df.loc[df.Rendement > 0, :].reset_index()
        df = df.iloc[:,1:]
        self.df = df
        self.X = self.df.loc[:, cols]  # Caractéristiques (température, precipitation, etc.)
        self.y = self.df[target]  # Valeurs cibles (rendement)
        
