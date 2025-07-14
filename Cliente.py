# Basado en el cliente oficial de la campaña MentalRiskES de IberLEF 2025 con las modificaciones necesarias para entrega la entrega y el funcionamiento deseado
# Como acumulacion de mensajes o el uso de un umbral entre otros
import requests, zipfile, io
from requests.adapters import HTTPAdapter, Retry
from typing import List, Dict
import random
import json
import os
import pandas as pd
import numpy as np
import time
import torch
from codecarbon import EmissionsTracker

import main

URL = "http://s3-ceatic.ujaen.es:8036"
TOKEN = ""

ENDPOINT_DOWNLOAD_TRIAL = URL+"/{TASK}/download_trial/{TOKEN}"
ENDPOINT_DOWNLOAD_TRAIN = URL+"/{TASK}/download_train/{TOKEN}"
ENDPOINT_DOWNLOAD_TEST = URL+"/{TASK}/download_test/{TOKEN}"

ENDPOINT_GET_MESSAGES_TRIAL = URL+"/{TASK}/getmessages_trial/{TOKEN}"
ENDPOINT_SUBMIT_DECISIONS_TRIAL = URL+"/{TASK}/submit_trial/{TOKEN}/{RUN}"

ENDPOINT_GET_MESSAGES = URL+"/{TASK}/getmessages/{TOKEN}"
ENDPOINT_SUBMIT_DECISIONS = URL+"/{TASK}/submit/{TOKEN}/{RUN}"

def download_messages_trial(task: str, token: str):
    """ Allows you to download the trial data of the task.
        Args:
          task (str): task from which the data is to be retrieved
          token (str): authentication token
    """

    response = requests.get(ENDPOINT_DOWNLOAD_TRIAL.format(TASK=task, TOKEN=token))

    if response.status_code != 200:
        print("Trial - Status Code " + task + ": " + str(response.status_code) + " - Error: " + str(response.text))
    else:
      z = zipfile.ZipFile(io.BytesIO(response.content))
      os.makedirs("./data/{task}/trial/".format(task=task))
      z.extractall("./data/{task}/trial/".format(task=task))

def download_messages_test(task: str, token: str):  
    response = requests.get(ENDPOINT_DOWNLOAD_TEST.format(TASK=task, TOKEN=token))

    if response.status_code != 200:
        print("Trial - Status Code " + task + ": " + str(response.status_code) + " - Error: " + str(response.text))
    else:
      z = zipfile.ZipFile(io.BytesIO(response.content))
      os.makedirs("./data/{task}/test/".format(task=task))
      z.extractall("./data/{task}/test/".format(task=task))
def download_messages_train(task: str, token: str):
    """ Allows you to download the train data of the task.
        Args:
          task (str): task from which the data is to be retrieved
          token (str): authentication token
    """
    response = requests.get(ENDPOINT_DOWNLOAD_TRAIN.format(TASK=task, TOKEN=token))

    if response.status_code != 200:
        print("Train - Status Code " + task + ": " + str(response.status_code) + " - Error: " + str(response.text))
    else:
      z = zipfile.ZipFile(io.BytesIO(response.content))
      os.makedirs("./data/{task}/train/".format(task=task),exist_ok=True)
      z.extractall("./data/{task}/train/".format(task=task))
      
      
class Client_task1_2:
    """ Client communicating with the official server.
        Attributes:
            token (str): authentication token
            number_of_runs (int): number of systems. Must be 3 in order to advance to the next round.
            tracker (EmissionsTracker): object to calculate the carbon footprint in prediction

    """
    def __init__(self, task:str, token: str, number_of_runs: int, tracker: EmissionsTracker):
        self.task = task
        self.token = token
        self.number_of_runs = number_of_runs
        self.tracker = tracker
        self.relevant_cols = ['duration', 'emissions', 'cpu_energy', 'gpu_energy',
                                'ram_energy','energy_consumed', 'cpu_count', 'gpu_count',
                                'cpu_model', 'gpu_model', 'ram_total_size','country_iso_code']


    def get_messages(self, retries: int, backoff: float) -> Dict:
        """ Allows you to download the test data of the task by rounds.
            Here a GET request is sent to the server to extract the data.
            Args:
                retries (int): number of calls on the server connection
                backoff (float): time between retries
        """
        session = requests.Session()
        retries = Retry(
                        total = retries,
                        backoff_factor = backoff,
                        status_forcelist = [500, 502, 503, 504]
                        )
        session.mount('https://', HTTPAdapter(max_retries=retries))

        response = session.get(ENDPOINT_GET_MESSAGES.format(TASK=self.task, TOKEN=self.token)) # ENDPOINT

        if response.status_code != 200:
            print("GET - Task {} - Status Code {} - Error: {}".format(self.task, str(response.status_code), str(response.text)))
            return []
        else:
            return json.loads(response.content)

    def submit_decission(self, messages: List[Dict], emissions_dict: Dict, retries: int, backoff: float, prediccion: Dict, tipo: Dict, prediccion_0: Dict, tipo_0: Dict, tipo_votos: Dict):
        """ Allows you to submit the decisions of the task by rounds.
            The POST requests are sent to the server to send predictions and carbon emission data
            Args:
                messages (List[Dict]): Message set of the current round
                emissions_dict (Dict): dictionary containing emissions for each model
                retries (int): number of calls on the server connection
                backoff (float): time between retries
        """
        decisions_run0 = {}
        decisions_run1 = {}
        decisions_run2 = {}
        type_addiction_decision = {}
        type_addiction_decision_0 = {}
        type_addiction_decision_votos = {}

        for usuario in self.usuarios_riesgo.keys():
            decisions_run0[usuario] = 1 if prediccion[usuario] >= 2 + (messages[0]["round"]//20) else 0
            decisions_run1[usuario] = 1 if prediccion_0[usuario] >= 2 + (messages[0]["round"]//20) else 0
            decisions_run2[usuario] = 1 if prediccion_0[usuario] >= 3 + (messages[0]["round"]/6) else 0
            type_addiction_decision[usuario] = tipo[usuario]
            type_addiction_decision_0[usuario] = tipo_0[usuario]
            type_addiction_decision_votos[usuario] = tipo_votos[usuario]
            print("round: ", messages[0]["round"])

        print("decisions_run0: ",decisions_run0)
        print("type_addiction_decision: ",type_addiction_decision)
        
        emissions_model1 = emissions_dict.get('model1', {})
        emissions_model0 = emissions_dict.get('model0', {})
        
        data1_run0 = {
            "predictions": decisions_run0,
            "emissions": emissions_model1
        }
        data1_run1 = {
            "predictions": decisions_run1,
            "emissions": emissions_model0
        }
        data1_run2 = {
            "predictions": decisions_run2,
            "emissions": emissions_model0
        }
        data2_run0 = {
            "predictions": decisions_run0,
            "types":type_addiction_decision,
            "emissions": emissions_model1
        }
        data2_run1 = {
            "predictions": decisions_run1,
            "types":type_addiction_decision_votos,
            "emissions": emissions_model1
        }
        data2_run2 = {
            "predictions": decisions_run2,
            "types":type_addiction_decision_0,
            "emissions": emissions_model0
        }

        data1 = []
        data1.append(json.dumps(data1_run0))
        data1.append(json.dumps(data1_run1))
        data1.append(json.dumps(data1_run2))

        data2 = []
        data2.append(json.dumps(data2_run0))
        data2.append(json.dumps(data2_run1))
        data2.append(json.dumps(data2_run2))

        # Session to POST request
        session = requests.Session()
        retries = Retry(
                        total = retries,
                        backoff_factor = backoff,
                        status_forcelist = [500, 502, 503, 504]
                        )
        session.mount('https://', HTTPAdapter(max_retries=retries))

        for run in range(0, self.number_of_runs):
            # For each run, new decisions
            response1 = session.post(ENDPOINT_SUBMIT_DECISIONS.format(TASK='task1', TOKEN=self.token, RUN=run), json=[data1[run]]) # ENDPOINT
            if response1.status_code != 200:
                print("POST - Task1 - Status Code {} - Error: {}".format(str(response1.status_code), str(response1.text)))
                return
            else:
                print("POST - Task1 - run {} - Message: {}".format(run, str(response1.text)))

            response2 = session.post(ENDPOINT_SUBMIT_DECISIONS.format(TASK='task2', TOKEN=self.token, RUN=run), json=[data2[run]]) # ENDPOINT
            if response2.status_code != 200:
                print("POST - Task2 - Status Code {} - Error: {}".format(str(response2.status_code), str(response2.text)))
                return
            else:
                print("POST - Task2 - run {} - Message: {}".format(run, str(response2.text)))
            os.makedirs('./data/preds/task1/', exist_ok=True)
            os.makedirs('./data/preds/task2/', exist_ok=True)
            with open('./data/preds/task1/round{}_run{}.json'.format(messages[0]["round"], run), 'w+', encoding='utf8') as json_file:
                json.dump(data1[run], json_file, ensure_ascii=False)
            with open('./data/preds/task2/round{}_run{}.json'.format(messages[0]["round"], run), 'w+', encoding='utf8') as json_file:
                json.dump(data2[run], json_file, ensure_ascii=False)
    def run_task1_2(self, retries: int, backoff: float):
        """ Main thread
            Args:
                retries (int): number of calls on the server connection
                backoff (float): time between retries
        """
        # Get messages for task1_2
        messages = self.get_messages(retries, backoff)

        predictor = modeloLongformers.cargar_modelo_guardado("./modelos_entrenados_finales")
        predictor_solo0 = modeloLongformers.cargar_modelo_guardado("./modelos_entrenados_finales_0")
        
        tipos_a = {'betting': 0, 'lootboxes': 1, 'onlinegaming': 2, 'trading': 3}

        if not hasattr(self, "usuarios_riesgo"):
            self.usuarios_riesgo = {}
            self.usuarios_clases = {}
            self.mensaje_por_usuario = {}
            self.usuarios_riesgo_0 = {}
            self.usuarios_clases_0 = {}
            self.usuarios_clases_votos = {}
            try:
                os.makedirs('./data/rounds/', exist_ok=True)
                
                round_files = [f for f in os.listdir('./data/rounds/') if f.startswith('round') and f.endswith('.json')]
                if round_files and messages and len(messages) > 0 and messages[0]["round"] > 1:
                    latest_round = max([int(f.replace('round', '').replace('.json', '')) for f in round_files])
                    with open(f'./data/rounds/round{latest_round}.json', 'r', encoding='utf8') as f:
                        prev_data = json.load(f)
                        if "user_risk_predictions" in prev_data:
                            self.usuarios_riesgo = {k: float(v) for k, v in prev_data.get("user_risk_predictions", {}).items()}
                            self.usuarios_riesgo_0 = {k: float(v) for k, v in prev_data.get("user_risk_predictions_0", {}).items()}
                            self.usuarios_clases = prev_data.get("user_addiction_types", {})
                            self.mensaje_por_usuario = prev_data.get("accumulated_messages", {})
                            
                            print(f"Restored previous state from round {latest_round}")
                            print(f"Users tracked: {len(self.usuarios_riesgo)}")
                            print(f"Messages history loaded: {sum(len(msgs) for msgs in self.mensaje_por_usuario.values())}")
            except Exception as e:
                print(f"Error recovering previous data: {e}")

        if len(messages) == 0:
            print("All rounds processed")
            return

        while len(messages) > 0:
            emissions_dict = {'model0': {}, 'model1': {}}
            
            print("----------------------- Processing round {}".format(messages[0]["round"]))

            for mensaje in messages:
                user_id = mensaje["nick"]
                texto = f"{mensaje.get('date', '')} {mensaje.get('message', '')}"
                if user_id not in self.usuarios_riesgo:
                    self.usuarios_riesgo[user_id] = 0
                    self.usuarios_riesgo_0[user_id] = 0
                    self.usuarios_clases[user_id] = None
                    self.usuarios_clases_0[user_id] = None
                    self.mensaje_por_usuario[user_id] = []
                    self.usuarios_clases_votos[user_id] = None
                
                self.mensaje_por_usuario[user_id].append(texto)
            #model1
            self.tracker.start()
            for user_id in self.usuarios_riesgo:
                if user_id in self.mensaje_por_usuario and self.mensaje_por_usuario[user_id]:
                    texto_acumulado_hasta_ahora = " ".join([str(mensaje) for mensaje in self.mensaje_por_usuario[user_id]])
                    
                    resultado = predictor.predecir_regresion([texto_acumulado_hasta_ahora])[0]
                    prob_binaria = resultado['valor_prediccion']
                    self.usuarios_riesgo[user_id] += prob_binaria
                    
                    result_tipos = predictor.predecir_multiclase([texto_acumulado_hasta_ahora])
                    if result_tipos:
                        max_index = np.argmax(result_tipos[0]['probabilidades_clase'])
                        for addic_tipo, idx in tipos_a.items():
                            if idx == max_index:
                                self.usuarios_clases[user_id] = addic_tipo
                                break
                    

                    if not hasattr(self, "votos_acumulados"):
                        self.votos_acumulados = {}
                    if user_id not in self.votos_acumulados:
                        self.votos_acumulados[user_id] = {tipo: 0 for tipo in tipos_a.keys()}
                    
                    if result_tipos:
                        probs = result_tipos[0]['probabilidades_clase']
                        for addic_tipo, idx in tipos_a.items():
                            self.votos_acumulados[user_id][addic_tipo] += probs[idx]
                        
                        max_votos = max(self.votos_acumulados[user_id].values())
                        for addic_tipo, votos in self.votos_acumulados[user_id].items():
                            if votos == max_votos:
                                self.usuarios_clases_votos[user_id] = addic_tipo
                                break
            
            #model1
            self.tracker.stop()
            df = pd.read_csv("emissions.csv")
            emissions_dict['model1'] = df.iloc[-1][self.relevant_cols].to_dict()
            
            #model0
            self.tracker.start()
            for user_id in self.usuarios_riesgo.keys():
                if user_id in self.mensaje_por_usuario and self.mensaje_por_usuario[user_id]:
                    texto_acumulado_hasta_ahora = " ".join([str(mensaje) for mensaje in self.mensaje_por_usuario[user_id]])
                    
                    resultado_solo0 = predictor_solo0.predecir_regresion([texto_acumulado_hasta_ahora])[0]
                    prob_binaria_0 = resultado_solo0['valor_prediccion']
                    self.usuarios_riesgo_0[user_id] += prob_binaria_0
                    
                    result_tipos_0 = predictor_solo0.predecir_multiclase([texto_acumulado_hasta_ahora])
                    if result_tipos_0 and user_id in self.usuarios_clases:
                        max_index = np.argmax(result_tipos_0[0]['probabilidades_clase'])
                        for addic_tipo, idx in tipos_a.items():
                            if idx == max_index:
                                self.usuarios_clases_0[user_id] = addic_tipo
                                break
            
            #model0
            self.tracker.stop()
            df = pd.read_csv("emissions.csv")
            emissions_dict['model0'] = df.iloc[-1][self.relevant_cols].to_dict()
            
            os.makedirs('./data/rounds/', exist_ok=True)
            
            results_data = {
                "messages": messages,
                "user_risk_predictions": self.usuarios_riesgo,
                "user_risk_predictions_0": self.usuarios_riesgo_0,
                "user_addiction_types": self.usuarios_clases,
                "accumulated_messages": self.mensaje_por_usuario
            }

            with open(f'./data/rounds/round{messages[0]["round"]}.json', 'w+', encoding='utf8') as json_file:
                json.dump(results_data, json_file, ensure_ascii=False)

            self.submit_decission(messages, emissions_dict, retries, backoff, self.usuarios_riesgo, self.usuarios_clases, self.usuarios_riesgo_0, self.usuarios_clases_0, self.usuarios_clases_votos)

            messages = self.get_messages(retries, backoff)

        print("All rounds processed")
        
def download_data(task: str, token: str):
    download_messages_test(task, token)

def get_post_data(task: str, token: str):
    # Emissions Tracker Config
    config = {
        "save_to_file": True,
        "log_level": "WARNING",
        "tracking_mode": "process",
        "output_dir": ".",
        "allow_multiple_runs": True
    }
    tracker = EmissionsTracker(**config)

    number_runs = 3 # Max: 3

    # Prediction period
    client_task1_2 = Client_task1_2(task, token, number_runs, tracker)
    client_task1_2.run_task1_2(5, 0.1)
    
if __name__ == '__main__':
    download_data("task2", TOKEN)
    #get_post_data("task1",TOKEN)
    