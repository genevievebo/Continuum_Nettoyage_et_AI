import matplotlib.pyplot as plt
from io import BytesIO

def capture_figure():
    '''
    Capture le graphique dans un objet BytesIO
    '''
    f = BytesIO()
    plt.savefig(f, format='png')
    return f