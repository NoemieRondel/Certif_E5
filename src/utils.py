import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import plotly.express as px
import numpy as np


def create_figure(data):
    data['trafficstatus_label'] = data['trafficstatus'].map({
        'freeFlow': 'Libre',
        'heavy': 'Dense',
        'congested': 'Congestionné',
        'unknown': 'Inconnu'
    })
    fig_map = px.scatter_mapbox(
            data,
            title="Trafic en temps réel",
            color="trafficstatus_label",
            lat="lat",
            lon="lon",
            color_discrete_map={'Libre': 'green', 'Dense': 'orange', 'Congestionné': 'red', 'Inconnu': 'gray'},
            labels={'trafficstatus_label': 'État du trafic'},
            zoom=10,
            height=500,
            mapbox_style="carto-positron"
    )

    return fig_map


def prediction_from_model(model, hour_to_predict):

    input_pred = np.array([0]*24)
    input_pred[int(hour_to_predict)] = 1

    cat_predict = np.argmax(model.predict(np.array([input_pred])))

    return cat_predict
