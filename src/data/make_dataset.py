import os
import re
import logging
import datetime
import pandas as pd
import argparse
import requests
import validators


def extract_regions_words(regions):
    '''
    Extraire les mots administratif comme District, County, etc 
    des noms de régions
    '''
    words = []
    for region in regions:
        found = [r for r in regions if region in r]
        if len(found) > 1:
            words += [r.replace(region, '') for r in found if r != region]
    words.remove('Grand ')
    words = sorted(list(set(words)))
    words.append(' Municipality')
    return words


def standardize_region_name(regions):
    ''' Standardize region names'''
    words = extract_regions_words(regions)
    regions = [clean_region(r, words) for r in regions]
    return regions


def clean_region(region, words):
    ''' Retirer les mots administratifs dans les noms de régions'''
    for word in words:
        region = region.replace(word, '')
    return region


def extract_year(sheetname):
    ''' Récupération de l'année à partir du nom de l'onglet'''
    year = int(re.search(r'\d+', sheetname).group())
    if year == 201:
        year = 2010
    return year


def replace_missing_values(df):
    ''' Remplacer les valeurs manquantes par 0 '''
    for char in ['-', 'x']:
        df.replace(char, 0, inplace=True)
    return df


def set_columns_names(df):
    ''' Vérification des noms des colonnes '''
    cols = ['Region', 'Ensemencee', 'Recoltee', "Rendement",
            "Production_boiseaux", "Production_tonnes", "Annee", "Onglet"]

    word_in_column = df.columns.str.contains("Production", regex=True).any()
    if word_in_column is False:
        header = df.iloc[0, :].values
        df = df.iloc[1:, :]
        df.columns = header
    df.columns = cols
    return df


def load_and_clean(filepath):
    ''' Charger et nettoyer les données provenant Ontario Open Data Portal '''

    f = pd.ExcelFile(filepath)
    data = pd.DataFrame()

    for sheet in f.sheet_names:
        df = f.parse(sheet, skiprows=1)
        df.loc[:, 'Annee'] = extract_year(sheet)
        df.loc[:, 'Onglet'] = sheet
        df = set_columns_names(df)

        last_row_index = df.Region.tolist().index('Ontario')
        df = df.iloc[:last_row_index, :]
        data = pd.concat([data, df], axis=0, ignore_index=True)

    # Remplacer les valeurs manquantes par 0
    data = replace_missing_values(data)

    # Standardiser les noms de régions
    data.loc[data.Region == 'Sudbury Regional Municipality', 'Region'] = 'Grand Sudbury'

    data.loc[:, 'Region'] = standardize_region_name(data.Region)
    print(len(data.Region.unique()))

    # Exclure les lignes qui ne sont pas des régions
    data = data.loc[~data.Region.str.contains('Région'), :]
    data = data.loc[~data.Region.str.contains('Ontario'), :]
    data = data.drop_duplicates()

    return data


def filter_data(data, n_threshold=3):      
    ''' Filtre pour garder les régions avec au moins n_threshold années de données '''
    mx = data.pivot_table(index="Annee", columns="Region", values="Rendement")
    ix_columns = mx.apply(lambda x: x == 0).sum() < n_threshold
    regions_to_keep = mx.columns[ix_columns]
    print(f'Nous conservons {len(regions_to_keep)} sur un total de {mx.shape[1]} régions')
    return data.loc[data.Region.isin(regions_to_keep), :]


def save_file(url, destdir):
    ''' Sauver un fichier à partir d'un url '''
    resp = requests.get(url)
    filename = url.split('/')[-1]

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    saved_filepath = os.path.join(destdir, filename)
    with open(saved_filepath, 'wb') as output:
        output.write(resp.content)
    return saved_filepath


def process(filepath):
    global data, logger
    '''
    Exécute le code de traitement des données pour transformer les données brutes (dans ../raw) 
    en données nettoyées prêtes à être analysées (enregistrées dans ../ processed).
    '''

    today_str = datetime.datetime.now().strftime("%Y%m%d")
    
    logger.info('Cleaning data from Ontario Open Data Portal')

    filename = os.path.basename(filepath)
    datadir = os.path.dirname(filepath)

    rawpath = filepath
    processed_filename = '.'.join(filename.split('.')[:-1]) + f'_processed.{today_str}.csv'
    outpath = os.path.join(datadir.replace('raw', 'processed'), processed_filename)

    logger.info('Reading the file from %s', rawpath)
    data = load_and_clean(rawpath)
    logger.info(F'Original shape {data.shape}')

    n = 3
    logger.info(F'Filtering data to keep only regions with at least {n} years of data')
    data_filtered = filter_data(data, n)
    logger.info(F'Filtered shape {data_filtered.shape}')

    data_filtered.to_csv(outpath, index=False)

    data_wide = data_filtered.pivot_table(index="Annee", columns="Region", values="Rendement")
    data_wide.to_csv(outpath.replace('.csv', '_wide.csv'))


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(prog=os.path.basename(__file__))
    parser.add_argument('-f', '--filepath', help='Path or url of file to clean',  action='store', default=os.getcwd())
    args = parser.parse_args()
    filepath = args.filepath

    filepath = "https://data.ontario.ca/dataset/e30dc044-5f75-4f33-b63e-6326f8769bea/resource/5fe1b9cc-5c3b-4f8f-99fc-2cacf41fcf9d/download/ctyoats_f.xlsx"
    # filepath = "https://data.ontario.ca/dataset/e30dc044-5f75-4f33-b63e-6326f8769bea/resource/ba7cbd73-c6a5-4533-b3e2-608a8c7a548b/download/ctysoy_f.xlsx"

    if validators.url(filepath):
        # datadir serait normalement définie dans un fichier de configuration
        # datadir représente le répertoire racine du projet où se trouve votre projet
        datadir = "rendement_avoine"
        rawdir = os.path.join(datadir, "data/raw")
        filepath = save_file(filepath, rawdir)

    process(filepath)


