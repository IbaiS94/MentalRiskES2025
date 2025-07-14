#This file has been developed by the SINAI research group for its usage in the MentalRiskES evaluation campaign at IberLEF 2024
# Modificado por Ibai para la evaluación en la fase de pruebas
import json
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import statistics
from typing import Dict



class multiclassClassification():
    def __init__(self, json_data, gold: Dict, labid):
        
        data = json_data
        
        predictions = data["predictions"]
        types = data["type"]
        for key, value in types.items():
            if isinstance(value, np.int64):
                types[key] = int(value)

        subjects = list(types.keys())
        pred_values = [predictions.get(subject, "") for subject in subjects]
        
        type_values = [types.get(subject, "") for subject in subjects]
  
        df = pd.DataFrame({
            'nick': subjects,
            'pred_b': pred_values,
            'pred': type_values,
            'round': [0] * len(subjects)  
        })
        
        self.run_results = df.sort_values(by=['nick'])
        self.gold_b = gold
        
        self.gold_label = {}
        for user, label in gold.items():
                self.gold_label[user] = label
        
        for user, label in self.gold_label.items():
            if label in labid:
                self.gold_label[user] = labid[label]
        if "emissions" in data:  # Acceder a emissions desde data[0]
            self.emissions = data["emissions"]

    def penalty(self, delay):
        p = 0.0314  # = -(LN(0,33)/(AVG_NUM_POSTS_PER_USER-1))
        pen = -1.0 + 2.0/(1+np.exp(-p*(delay-1)))
        return(pen)

    def n_pos(self):
        total_pos = 0
        for key in self.gold_b:
            total_pos += self.gold_b[key]
        return(total_pos)

    def eval_performance(self):
        print("===================================================")
        print("Predicciones VS Gold:")
        for index, row in self.run_results.iterrows():
            nick = row['nick']
            pred = row['pred']
            gold = self.gold_label.get(nick, "N/A")
            print(f"User: {nick} | Prediction: {pred} | Gold: {gold}")
        print("===================================================")
        print("Evaluacion:") 
        
        # Crear listas organizadas que aseguren correspondencia entre predicciones y valores reales
        y_pred = []
        y_true = []
        
        # Recorrer las filas ordenadas del DataFrame para asegurar correspondencia
        for index, row in self.run_results.iterrows():
            nick = row['nick']
            if nick in self.gold_label:
                y_pred.append(row['pred'])
                y_true.append(self.gold_label[nick])
        
        # Métricas multiclase (para clasificación con 4 clases)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        macro_precision = metrics.precision_score(y_true, y_pred, average='macro')
        macro_recall = metrics.recall_score(y_true, y_pred, average='macro')
        macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')
        micro_precision = metrics.precision_score(y_true, y_pred, average='micro')
        micro_recall = metrics.recall_score(y_true, y_pred, average='micro')
        micro_f1 = metrics.f1_score(y_true, y_pred, average='micro')

        print("MULTICLASS METRICS (4 clases): =============================")
        print("Accuracy:"+str(accuracy))
        print("Macro precision:"+str(macro_precision))
        print("Macro recall:"+str(macro_recall))
        print("Macro f1:"+str(macro_f1))
        print("Micro precision:"+str(micro_precision))
        print("Micro recall:"+str(micro_recall))
        print("Micro f1:"+str(micro_f1))

        """
        # MÉTRICAS BASADAS EN LATENCIA (COMENTADAS)
        # Estas métricas evalúan no solo la precisión de las predicciones,
        # sino también qué tan rápido se identifican las instancias positivas.
        # Nota: Estas son métricas binarias (para clasificación binaria),
        # que utilizan las columnas pred_b (valores 0 o 1) en lugar de las
        # etiquetas multiclase de 4 clases.
        
        total_pos = self.n_pos()
        erdes5 = np.zeros(len(self.run_results))
        erdes30 = np.zeros(len(self.run_results))
        ierdes = 0
        true_pos = 0
        false_pos = 0
        latency_tps = list()
        penalty_tps = list()

        for index, r in self.run_results.iterrows():
            try:
                if (self.gold_b[r['nick']] == r['pred_b']):
                    if (r['pred_b'] == 1):  # True positive
                        true_pos += 1
                        erdes5[ierdes] = 1.0 - (1.0/(1.0+np.exp((r["round"]+1) - 5.0)))
                        erdes30[ierdes] = 1.0 - (1.0/(1.0+np.exp((r["round"]+1) - 30.0)))
                        latency_tps.append(r["round"]+1)
                        penalty_tps.append(self.penalty(r["round"]+1))
                    else:  # True negative
                        erdes5[ierdes] = 0
                        erdes30[ierdes] = 0
                else:
                    if (r['pred_b'] == 1):  # False positive
                        false_pos += 1
                        erdes5[ierdes] = float(total_pos) / float(len(self.gold_b))
                        erdes30[ierdes] = float(total_pos) / float(len(self.gold_b))
                    else:  # False negative
                        erdes5[ierdes] = 1
                        erdes30[ierdes] = 1
            except KeyError:
                print("User does not appear in the gold:"+r['nick'])
            ierdes += 1

        _speed = 1-np.median(np.array(penalty_tps)) if penalty_tps else 0
        if true_pos != 0:
            precision = float(true_pos) / float(true_pos+false_pos)    
            recall = float(true_pos) / float(total_pos)
            f1_erde = 2 * (precision * recall) / (precision + recall)
            _latencyweightedF1 = f1_erde*_speed
        else:
            _latencyweightedF1 = 0
            _speed = 0
            
        print("LATENCY-BASED METRICS (Binary): =============================")
        print("ERDE_5:"+str(np.mean(erdes5)))
        print("ERDE_30:"+str(np.mean(erdes30)))
        print("Median latency:"+str(np.median(np.array(latency_tps)) if latency_tps else 0)) 
        print("Speed:"+str(_speed)) 
        print("latency-weightedF1:"+str(_latencyweightedF1))
        
        latency_metrics = {
            'ERDE5': np.mean(erdes5),
            'ERDE30': np.mean(erdes30), 
            'latencyTP': np.median(np.array(latency_tps)) if latency_tps else 0, 
            'speed': _speed, 
            'latency-weightedF1': _latencyweightedF1
        }
        """
        
        # Devolver solo métricas multiclase
        return {
            'Accuracy': accuracy, 
            'Macro_P': macro_precision, 
            'Macro_R': macro_recall,
            'Macro_F1': macro_f1,
            'Micro_P': micro_precision, 
            'Micro_R': micro_recall,
            'Micro_F1': micro_f1
        }

###########################################################################
class binaryClassification():
    """ Calculado de metricas para la clasificación binaria
            json_path (str): ruta al JSON. Tiene esta estructura: 
                [
                    {
                        "predictions": 
                        {
                            "subject1": 0,
                            "subject10": 1,
                            ...
                        },
                        "type": 
                        {
                            "subject1": "lootboxes",
                            "subject10": "onlinegaming",
                            ...
                        },
                        "emissions": 
                        {
                            "duration": 0.01,
                            "emissions": 3.67552e-08,
                            ...
                        }
                    }
                ]
            gold (Dict): dict con golds
    """
    primera_deteccion = {}

    def __init__(self, json_data, gold: Dict, round: int):
        data = json_data
        print(self.primera_deteccion)
        predictions = data["predictions"]
        #types = data[0]["type"]
        
        subjects = list(predictions.keys())
        pred_values = list(predictions.values())
        #type_values = [types.get(subject, "") for subject in subjects]
        
        rounds = round
        
        if rounds == 1:
            self.primera_deteccion.clear()
        
        df = pd.DataFrame({
            'nick': subjects,
            'pred_b': pred_values,
            #'pred': type_values,
            'round': rounds
        })
        
        self.run_results = df.sort_values(by=['nick'])
        self.gold_b = gold
        
        if "emissions" in data:
            self.emissions = data["emissions"]

    def penalty(self, delay):
        p = 0.0199
        pen = -1.0 + 2.0/(1+np.exp(-p*(delay-1)))
        return(pen)

    def n_pos(self):
        total_pos = 0
        for key in self.gold_b:
            total_pos += self.gold_b[key]
        return(total_pos)

    def eval_performance(self):
        print("===================================================")
        print("Evaluacion:") 
        total_pos = self.n_pos()
        
        for index, r in self.run_results.iterrows():
            nick = r['nick']
            pred_b = r['pred_b']
            round_num = r['round']
            
            if pred_b == 1 and nick not in self.primera_deteccion:
                self.primera_deteccion[nick] = {
                    'round': round_num,
                    'pred_b': pred_b
                }
        
        erdes5 = np.zeros(len(self.gold_b))
        erdes30 = np.zeros(len(self.gold_b))
        ierdes = 0
        true_pos = 0
        false_pos = 0
        latency_tps = list()
        penalty_tps = list()
        
        y_true = []
        y_pred_b = []
        
        print("Predicciones VS Gold:")
        
        for nick, gold_value in self.gold_b.items():
            y_true.append(gold_value)
            
            if nick in self.primera_deteccion:
                detection = self.primera_deteccion[nick]
                pred_b = detection['pred_b']
                round_num = detection['round']
                y_pred_b.append(pred_b)
                
               # print(f"User: {nick} | Prediction: {pred_b} | Gold: {gold_value}")
                
                if gold_value == pred_b:
                    if pred_b == 1:  # True positive
                        true_pos += 1
                        erdes5[ierdes] = 1.0 - (1.0/(1.0+np.exp((round_num+1) - 5.0)))
                        erdes30[ierdes] = 1.0 - (1.0/(1.0+np.exp((round_num+1) - 30.0)))
                        latency_tps.append(round_num+1)
                        penalty_tps.append(self.penalty(round_num+1))
                    else:  # True negative
                        erdes5[ierdes] = 0
                        erdes30[ierdes] = 0
                else:
                    if pred_b == 1:  # False positive
                        false_pos += 1
                        erdes5[ierdes] = float(total_pos) / float(len(self.gold_b))
                        erdes30[ierdes] = float(total_pos) / float(len(self.gold_b))
                    else:  # False negative
                        erdes5[ierdes] = 1
                        erdes30[ierdes] = 1
            else:
                pred_b = 0 
                y_pred_b.append(pred_b)
                
                #print(f"User: {nick} | Prediction: {pred_b} (default) | Gold: {gold_value}")
                
                if gold_value == 1:  # False negative
                    erdes5[ierdes] = 1
                    erdes30[ierdes] = 1
                else:  # True negative
                    erdes5[ierdes] = 0
                    erdes30[ierdes] = 0
                    
            ierdes += 1

        print(f"Tamaños - Gold: {len(y_true)}, Predicciones: {len(y_pred_b)}")
        assert len(y_true) == len(y_pred_b), "Gold y prediciones no cuadran"

        _speed = 1-np.median(np.array(penalty_tps)) if penalty_tps else 0
        if true_pos != 0:
            precision = float(true_pos) / float(true_pos+false_pos)    
            recall = float(true_pos) / float(total_pos)
            f1_erde = 2 * (precision * recall) / (precision + recall)
            _latencyweightedF1 = f1_erde*_speed
        else:
            _latencyweightedF1 = 0
            _speed = 0

        accuracy = metrics.accuracy_score(y_true, y_pred_b)
        macro_precision = metrics.precision_score(y_true, y_pred_b, average='macro')
        macro_recall = metrics.recall_score(y_true, y_pred_b, average='macro')
        macro_f1 = metrics.f1_score(y_true, y_pred_b, average='macro')
        micro_precision = metrics.precision_score(y_true, y_pred_b, average='micro')
        micro_recall = metrics.recall_score(y_true, y_pred_b, average='micro')
        micro_f1 = metrics.f1_score(y_true, y_pred_b, average='micro')

        print("BINARY METRICS: =============================")
        print("Accuracy:"+str(accuracy))
        print("Macro precision:"+str(macro_precision))
        print("Macro recall:"+str(macro_recall))
        print("Macro f1:"+str(macro_f1))
        print("Micro precision:"+str(micro_precision))
        print("Micro recall:"+str(micro_recall))
        print("Micro f1:"+str(micro_f1))

        print("LATENCY-BASED METRICS: =============================")
        print("ERDE_5:"+str(np.mean(erdes5)))
        print("ERDE_30:"+str(np.mean(erdes30)))
        print("Median latency:"+str(np.median(np.array(latency_tps))) if latency_tps else "N/A") 
        print("Speed:"+str(_speed)) 
        print("latency-weightedF1:"+str(_latencyweightedF1)) 
        return {'Accuracy': accuracy, 'Macro_P': macro_precision, 'Macro_R': macro_recall,'Macro_F1': macro_f1,'Micro_P': micro_precision, 'Micro_R': micro_recall,
        'Micro_F1': micro_f1, 'ERDE5':np.mean(erdes5),'ERDE30': np.mean(erdes30), 'latencyTP': np.median(np.array(latency_tps)) if latency_tps else 0, 
        'speed': _speed, 'latency-weightedF1': _latencyweightedF1}

class Emissions():
    """ Clase para calcular emisiones de carbono, realmente no se ha usado en la experimentación
    """
    def __init__(self, emissions_run) -> None:
        self.emissions_run = emissions_run
        self.aux = {}
        for key, value in emissions_run.items():
            self.aux[key] = 0
        pass

    def update_emissions(self, emissions_round):
        if len(emissions_round.items()) != 0:
            if emissions_round['duration'] - self.aux['duration'] < 0 :
                print("RESETEO: ", self.emissions_run)
                for key, value in self.aux.items():
                    self.aux[key] = 0
            for key, value in self.emissions_run.items():
                if key not in ["cpu_count","gpu_count","cpu_model","gpu_model", "ram_total_size","country_iso_code"]:
                    round_ = emissions_round[key] - self.aux[key]
                    self.emissions_run[key].append(round_)
                    self.aux[key] = emissions_round[key]
                else:
                    self.emissions_run[key] = emissions_round[key]
        else:
            print("Empty: ", self.emissions_run)
            for key, value in self.aux.items():
                self.aux[key] = 0

    def calculate_emissions(self):
        dict_ = {}
        for key, value in self.emissions_run.items():
            if key in ["cpu_count","gpu_count","cpu_model","gpu_model", "ram_total_size","country_iso_code"]:
                dict_[key] = self.emissions_run[key]
            else: 
                dict_[key+"_min"] = min(self.emissions_run[key])
                dict_[key+"_max"] = max(self.emissions_run[key])
                dict_[key+"_mean"] = sum(self.emissions_run[key])/len(self.emissions_run[key])
                dict_[key+"_desv"] = statistics.pstdev(self.emissions_run[key])
        return dict_