from fastapi import FastAPI
from pydantic import BaseModel
import requests
import mlflow
from mangum import Mangum

#uvicorn api:app --reload --port 8000


    

class wines(BaseModel):
    fixed_acidity: float 
    volatile_acidity: float 
    citric_acid: float 
    residual_sugar: float 
    chlorides: float 
    free_sulfur_dioxide: float 
    total_sulfur_dioxide: float 
    density: float 
    pH: float 
    sulphates: float 
    alcohol: float 
    
app = FastAPI()

handler = Mangum(app)


@app.get('/experiments')
def get_experiment():
    # api do mllflow
    url = 'http://localhost:5000/api/2.0/preview/mlflow/experiments/list'
    response = requests.request('GET', url=url)
    dados = response.json()

    return dados

@app.post('/model')
def predict(wines: wines):
    mlflow.set_tracking_uri(uri='http://0.0.0.0:5000/')
    PATH = 'models:/wine_quality/Production'
    classes = ['Bad wine', 'Good wine']
    loaded_model = mlflow.sklearn.load_model(PATH)
    
    dados = [[i[1] for i in wines]]
    
    label = loaded_model.predict(dados) 

    resultado = classes[int(label[0])]
    return {'class': resultado}
