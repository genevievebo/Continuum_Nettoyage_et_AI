import seaborn as sns
import matplotlib.pyplot as plt
from . import capture_figure
sns.set(font_scale=0.9, style='whitegrid')
colors = ["#40BEAD", "#248CC4", "#308E81", "#576AC9"] # IVADO palette

# --------- Visualisation des jeux de données --------------------------
def plot_points_for_station(dat, station_name, var):
    '''
    Cette fonction crée un graphique en points de la température maximale en fonction du temps
    '''
    mx = dat.loc[dat.STATION_NAME.str.contains(station_name),:]
    fig, ax = plt.subplots(1)
    for station in mx.STATION_NAME.unique():
        mx.loc[mx.STATION_NAME==station,:].plot('LOCAL_DATE', var, figsize=(16,5), marker='o', label=station, ax=ax, alpha=0.6)
    plt.title(F'{var} en fonction de la date')
    plt.ylabel(var)
    plt.xlabel('Date')
    plt.legend()
    plt.show()
    
    
def get_first_last_date(dat, station_name):
    '''
    Obtenir la première et la dernière date pour une station donnée
    '''
    mx = dat.loc[dat.STATION_NAME.str.contains(station_name),:]
    t = mx.loc[:,['STATION_NAME', 'x', 'y','LOCAL_DATE']]
    t0 = t.drop_duplicates('STATION_NAME', keep='first')
    t1 = t.drop_duplicates('STATION_NAME', keep='last')
    t1 = t1.drop(['x', 'y'], axis=1)
    t = t0.merge(t1, on=['STATION_NAME'])
    t.columns = ['STATION_NAME', 'x', 'y', 'FIRST_DATE', 'LAST_DATE', ]
    return t


def plot_geomap(dat, year, var, with_labels=True, color='Oranges'):
    '''
    Cette fonction crée une carte géographique représentant la variable var pour une année donnée
    '''
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.axis('off')
    ax.set_title('', fontdict={'fontsize': '5', 'fontweight' : '1'})
    vmin = 0
    vmax = dat[var].max()*1.05
    sm = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm)
    cbar.ax.tick_params(labelsize=10)
    mx = dat.loc[dat['LOCAL_YEAR']==year,:].drop_duplicates()
    mx.plot(var, cmap=color, linewidth=0.8, ax=ax, edgecolor='0.8', figsize=(10,8))
    if with_labels:
        tmp = mx.loc[:,['x', 'y', 'Region', 'STATION_NAME']].drop_duplicates()
        for idx, row in tmp.iterrows():
            plt.annotate(text=F"{row['Region']}-{row['STATION_NAME']}", xy=(row['x'], row['y']),
                        horizontalalignment='center', fontsize='small', color='black', wrap=True)
    plt.title(F'{year} - {var}')
    plt.show()
      
def plot_subset(data, x, y, values, values_col, y_units='', x_units='', capture_plot=False):            
    '''
    Graphique de y en fonction x pour les valeurs spécifiées
    exmple : plot_subset(data, x='Annee', y='Rendement', values=['Sudbury', 'Grand Sudbury'],
                values_col='Region', y_units='boiseaux/acre')
    '''
    _, ax = plt.subplots(figsize=(10, 3))
   
    for value in values:
        tmp = data.loc[data[values_col]==value, :]
        tmp[x] = tmp[x].astype(int)
        tmp.plot(x=x, y=y, label=value, marker='o', ax=ax)
        
    plt.xticks(data[x].unique(), rotation=90)
    plt.ylim(data[y].min() - 5, data[y].max() + 5)
    plt.ylabel(f'{y.title()} ({y_units})')
    plt.xlabel(f'{x.title()} ({x_units})')
    plt.title(f'{y.title()} par {values_col.lower()} en fonction de(s) {x.lower()}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if capture_plot:
        return capture_figure()
    plt.show()
    
    
def heatmap(data, index, columns, values, capture_plot=False, figsize=(10,5)):
    '''
    Heatmap du rendement en fonction des années et des régions
    exemple: heatmap(data, index='Annee', values='Rendement', columns='Region')
    '''
    _, ax = plt.subplots(figsize=figsize)
    mx = data.pivot_table(index=index, columns=columns, values=values)
    sns.heatmap(mx, annot=True, ax=ax, fmt=".0f")
    plt.title(f'{values.title()} par {index} et par {columns}')
    if capture_plot:
        return capture_figure()
    plt.show()
    
    
def boxplot(data, x, y, title='', subtitle='', capture_plot=False):
    '''
    Boxplot x en fonction de y
    exemple : suptitle = f"Les données proviennent de {len(data.Region.unique())} régions",
    boxplot(data, 'Annee', 'Rendement', suptitle=suptitle)
    '''
    sns.boxplot(data, y=y, x=x, color='grey')
    plt.xticks(rotation=90)
    if title == '':
        title = f"{y.title()} par {x.title()}"
    plt.suptitle(title)
    plt.title(subtitle, fontsize=10)
    if capture_plot:
        return capture_figure()
    plt.show()
    
    
    
    
# --------- Visualisation des résultats des modèles --------------------------
def plot_pred_vs_true(y_test, y_pred, score, model_name, y_train=None, y_train_pred=None):
    plt.figure(figsize=(4,3))
    if y_train is not None and y_train_pred is not None:
        plt.scatter(y_train, y_train_pred, label='Training data', color=colors[1])
        
    plt.scatter(y_test, y_pred, label='Test data', color=colors[0])
    plt.title(F'{model_name} [test score={score:2.2}]')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.legend()
    plt.ylim(y_test.min(), y_test.max())
    plt.show()