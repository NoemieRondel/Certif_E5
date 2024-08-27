import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
import logging
from flask import Flask, render_template, request
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import plotly.io as pio
import flask_monitoringdashboard as dashboard
from keras.models import load_model

from src.get_data import GetData
from src.utils import create_figure, prediction_from_model 

app = Flask(__name__)

# Configure Flask logging
app.logger.setLevel(logging.INFO)  # Set log level to INFO
handler = logging.FileHandler('app.log')  # Log to a file
app.logger.addHandler(handler)



dashboard.config.init_from(file='config.cfg')
data_retriever = GetData(url="https://data.rennesmetropole.fr/api/explore/v2.1/catalog/datasets/etat-du-trafic-en-temps-reel/exports/json?lang=fr&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B")
data = data_retriever()



try:
    app.logger.info('Tentative de chargement du modèle...')
    model = load_model('model.h5')
    app.logger.info('Modèle chargé avec succès.')
except Exception as e:
    app.logger.error(f"Erreur lors du chargement du modèle : {e}")
    model = None



@app.route('/', methods=['GET', 'POST'])
def index():
    app.logger.info('Requête reçue à la route index.')
    if request.method == 'POST':
        try:
            fig_map = create_figure(data)
            graph_json = pio.to_json(fig_map)
            selected_hour = request.form['hour']
            cat_predict = prediction_from_model(model, selected_hour)
            app.logger.info(f'Prédiction pour l\'heure {selected_hour}: {cat_predict}')
            color_pred_map = {0: ["Prédiction : Libre", "green"], 
                              1: ["Prédiction : Dense", "orange"], 
                              2: ["Prédiction : Bloqué", "red"]}
            return render_template('index.html', 
                                   graph_json=graph_json, 
                                   text_pred=color_pred_map[cat_predict][0], 
                                   color_pred=color_pred_map[cat_predict][1])
        except Exception as e:
            app.logger.error(f"Erreur dans la route POST : {e}")
            return 'Erreur dans la demande POST', 500
    else:
        try:
            fig_map = create_figure(data)
            graph_json = pio.to_json(fig_map)
            app.logger.info('Affichage de la page d\'accueil.')
            return render_template('index.html', graph_json=graph_json)
        except Exception as e:
            app.logger.error(f"Erreur dans la route GET : {e}")
            return 'Erreur dans la demande GET', 500

@app.errorhandler(500)
def server_error(error):
    app.logger.exception('An exception occurred during a request.')
    return 'Internal Server Error', 500

#dashboard.config.enable_logging = True
dashboard.bind(app)
dashboard.config.monitor_level = 3


if __name__ == '__main__':
    app.run(debug=True)