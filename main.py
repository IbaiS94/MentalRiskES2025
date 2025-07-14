import os
from Evaluacion import binaryClassification, multiclassClassification
import ServidorExp
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import math
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import time
from ServidorExp import ServidorExp

import random
from sklearn.metrics import f1_score, accuracy_score, classification_report
import sys
import threading
import hashlib
SEMILLA = 69
torch.manual_seed(SEMILLA)
torch.cuda.manual_seed_all(SEMILLA)
np.random.seed(SEMILLA)
random.seed(SEMILLA)





class Config:
    """Parámetros de configuración para la aplicación."""
    NOMBRE_MODELO = "ignacio-ave/beto-sentiment-analysis-Spanish" #"hiiamsid/sentence_similarity_spanish_es"   #"nomic-ai/nomic-embed-text-v2-moe"     #"ignacio-ave/beto-sentiment-analysis-Spanish"  #"tabularisai/multilingual-sentiment-analysis"     #"jinaai/jina-embeddings-v3"           # "intfloat/multilingual-e5-large"       # Primero: "allenai/longformer-base-4096" Segundo: "pysentimiento/robertuito-base-uncased"
    MODELO_SENTIMIENTO = "pysentimiento/robertuito-sentiment-analysis"
    LONGITUD_MAXIMA =  512
    EPOCAS = 200
    DIMENSION_OCULTA = 128
    DIMENSION_SALIDA = 2 
    TASA_DROPOUT = 0.4
    DIRECTORIO_CACHE = "./transformer_cache"
    DIRECTORIO_SALIDA = "./resultadoscarpeta"
    ARCHIVO_SENTIMIENTO = "mensajes_ludopatia_sinonimos.txt"
    DIRECTORIO_MODELO_AJUSTADO = "./modelo_fine_tuned"
    DIRECTORIO_MODELOS = "./modelos_entrenados"

class DataProcessor:
    """Maneja la carga y el procesamiento de datos."""
    
    @staticmethod
    def cargar_etiquetas(ruta):
        """Carga los IDs de usuario y las etiquetas desde un archivo CSV."""
        id_usuarios = []
        etiquetas = []
        with open(ruta, 'r', encoding='utf-8') as archivo:
            lector = csv.reader(archivo)
            primfila = False
            for fila in lector:
               if(primfila):
                if len(fila) >= 2:
                    id_usuario, etiqueta = fila[0], fila[1]
                    if etiqueta in ['0', '1']:
                        id_usuarios.append(id_usuario)
                        etiquetas.append(int(etiqueta))
                    else:
                        id_usuarios.append(id_usuario)
                        etiquetas.append(etiqueta)
               else:
                primfila = True
                    
        return id_usuarios, etiquetas
    
    @staticmethod
    def cargar_datos_usuarios(carpeta, id_usuarios):
        """Carga los datos de usuario desde archivos JSON."""
        datos_usuarios = []
        for id_usuario in id_usuarios:
            ruta_archivo = os.path.join(carpeta, f"{id_usuario}.json")
            if os.path.exists(ruta_archivo):
                with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
                    datos = json.load(archivo)
                datos_usuarios.append(datos)
        return datos_usuarios

class ModelHandler:
    """Maneja la carga del modelo y la generación de embeddings."""
    
    @staticmethod
    def cargar_modelo(nombre_modelo=None):
        """Carga un modelo pre-entrenado y su tokenizador con opción de respaldo."""
        nombre_modelo = nombre_modelo or Config.NOMBRE_MODELO
        try:
            modelo = AutoModel.from_pretrained(
                nombre_modelo, 
                trust_remote_code=True, 
                local_files_only=False, 
                force_download=False, 
                cache_dir=Config.DIRECTORIO_CACHE
            )
            tokenizador = AutoTokenizer.from_pretrained(
                nombre_modelo, 
                trust_remote_code=True, 
                local_files_only=False, 
                force_download=False, 
                cache_dir=Config.DIRECTORIO_CACHE
            )
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")

        
        return modelo, tokenizador
    @staticmethod
    def obtener_embeddings(texto, modelo, tokenizador, max_length=512, usar_gemini=True):
        """
        Genera embeddings para el texto de entrada con caché en disco y manejo de textos largos.
        Soporta modelos de HuggingFace y Gemini (Google AI).
        
        Args:
            texto (str): Texto para generar embeddings
            modelo: Modelo pre-entrenado
            tokenizador: Tokenizador asociado al modelo (no usado para Gemini)
            max_length (int): Longitud máxima de tokens por bloque
            usar_gemini (bool): Si es True, usa el modelo gemini-embedding-exp en lugar del modelo HuggingFace
            
        Returns:
            numpy.ndarray: Vector de embeddings
        """
        # caché
        if not hasattr(ModelHandler, '_cache_hits'):
            ModelHandler._cache_hits = 0
            ModelHandler._cache_misses = 0
            print("Inicializado sistema de caché de embeddings en disco")
        
        cache_dir = os.path.join(Config.DIRECTORIO_CACHE, "embeddings_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        if not texto or not isinstance(texto, str):
            texto = " "
        
        try:
            
            # Reemplazar explícitamente caracteres raros y otros problemáticos con espacios
            texto_limpio = ''.join(c if ord(c) < 0x10000 and c.isprintable() else ' ' for c in texto)
            texto_limpio = texto_limpio.encode('utf-8', 'ignore').decode('utf-8')
            texto_hash = hashlib.md5(texto_limpio.encode()).hexdigest()
            nombre_modelo = "gemini" if usar_gemini else Config.NOMBRE_MODELO[:3]
            archivo_cache = os.path.join(cache_dir, f"{nombre_modelo}_{texto_hash}.npy")
            # verificar caché 
            if os.path.exists(archivo_cache):
                try:
                    # Cargar desde disco
                    embedding = np.load(archivo_cache)
                    ModelHandler._cache_hits += 1
                    if ModelHandler._cache_hits % 100 == 0:
                        aciertos_cache = ModelHandler._cache_hits / (ModelHandler._cache_hits + ModelHandler._cache_misses)
                        print(f"Estadisticas cache - Hits: {ModelHandler._cache_hits},  Fallos: {ModelHandler._cache_misses}, Tasa: {aciertos_cache:.2f}")
                    return embedding
                except Exception as e:
                    print(f"Error al cargar caché: {str(e)}. Regenerando embedding.")
                    # Si error al cargar, regeneramos
                    ModelHandler._cache_misses += 1
            else:
                ModelHandler._cache_misses += 1
            
            # Gemini
            if usar_gemini:
                try:
                    import google.generativeai as genai
                    
                    # API
                    if not hasattr(genai, '_configured') or not genai._configured:
                        api_key = os.environ.get('GOOGLE_API_KEY')
                        if not api_key:
                            api_key = ""  
                        genai.configure(api_key=api_key)
                    
                    if len(texto) > 60000: # 8192 serian aproximademente los 2048 tokens de Gemini, poner para evitar truncamiento
                        bloques = [texto[i:i+60000] for i in range(0, len(texto), 60000)]
                        embeddings_bloques = []
                        
                        for bloque in bloques:
                            resultado = genai.embed_content(
                                model="models/text-embedding-004", 
                                content=bloque        
                            )
                            embeddings_bloques.append(np.array(resultado["embedding"]))
                        
                        embedding = np.mean(embeddings_bloques, axis=0)
                    else:
                        resultado = genai.embed_content(
                            model="models/text-embedding-004",  
                            content=texto
                        )
                        embedding = np.array(resultado["embedding"])
                
                except ImportError:
                    print("Error al usar embeddings de Gemini")
                    # Fallback a modelo HuggingFace
                    usar_gemini = False
                    nombre_modelo = "gemini" if usar_gemini else Config.NOMBRE_MODELO[:3]
                    archivo_cache = os.path.join(cache_dir, f"{nombre_modelo}_{texto_hash}.npy")
                except Exception as e:
                    # Si falla por emojis o chorradas, intentar limpiar el texto
                    try:
                        print(f"Reintentando con texto limpiado...")
                        texto = texto_limpio
                        
                        if len(texto) > 60000:   # 8192 serian aproximademente los 2048 tokens de Gemini, poner para evitar truncamiento
                            print(f"Texto muy largo para Gemini ({len(texto)} caracteres), procesando en bloques...")
                            bloques = [texto[i:i+60000] for i in range(0, len(texto), 60000)]
                            embeddings_bloques = []
                            
                            for bloque in bloques:
                                resultado = genai.embed_content(
                                    model="models/text-embedding-004",
                                    content=bloque        
                                )
                                embeddings_bloques.append(np.array(resultado["embedding"]))
                            
                            embedding = np.mean(embeddings_bloques, axis=0)
                        else:
                            resultado = genai.embed_content(
                                model="models/text-embedding-004",  
                                content=texto
                            )
                            embedding = np.array(resultado["embedding"])
                    except Exception as retry_e:
                        print(f"Error en segundo intento con Gemini: {str(retry_e)}. Fallback al modelo HuggingFace.")
                        usar_gemini = False
                        nombre_modelo = "gemini" if usar_gemini else Config.NOMBRE_MODELO[:3]
                        archivo_cache = os.path.join(cache_dir, f"{nombre_modelo}_{texto_hash}.npy")
            
            if not usar_gemini:
                tokens_ids = tokenizador.encode(texto, add_special_tokens=True)
                numero_tokens = len(tokens_ids)
                
                if numero_tokens <= max_length:
                    entradas = tokenizador(
                        texto,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=max_length
                    )
                    
                    with torch.no_grad():
                        salida = modelo(**entradas)
                    
                    embedding = salida.last_hidden_state.mean(dim=1).numpy().flatten()
                else:
                    print(f"Texto largo: {numero_tokens} tokens -> procesando en bloques de {max_length}")
                    
                    oraciones = texto.split('. ')
                    if len(oraciones) == 1:  
                        oraciones = texto.split(', ')
                    if len(oraciones) == 1: 
                        oraciones = [texto[i:i+max_length*4] for i in range(0, len(texto), max_length*4)]
                    
                    bloques = []
                    bloque_actual = ""
                    tokens_bloque = 0
                    
                    for oracion in oraciones:
                        tokens_oracion = len(tokenizador.encode(oracion, add_special_tokens=(bloque_actual=="")))
                        
                        if tokens_bloque + tokens_oracion <= max_length:
                            if bloque_actual:
                                bloque_actual += ". " + oracion
                            else:
                                bloque_actual = oracion
                            tokens_bloque += tokens_oracion
                        else:
                            if bloque_actual:
                                bloques.append(bloque_actual)
                            bloque_actual = oracion
                            tokens_bloque = tokens_oracion
                    
                    if bloque_actual:
                        bloques.append(bloque_actual)
                    
                    embeddings_bloques = []
                    pesos_bloques = []
                    
                    for i, bloque in enumerate(bloques):
                        entradas = tokenizador(
                            bloque,
                            return_tensors="pt",
                            padding="max_length",
                            truncation=True,
                            max_length=max_length
                        )
                        
                        num_tokens = (entradas.attention_mascara[0] == 1).sum().item()
                        pesos_bloques.append(num_tokens)
                        
                        with torch.no_grad():
                            salida = modelo(**entradas)
                        
                        emb_bloque = salida.last_hidden_state.mean(dim=1).numpy().flatten()
                        embeddings_bloques.append(emb_bloque)
                    
                    total_tokens = sum(pesos_bloques)
                    embedding = np.zeros_like(embeddings_bloques[0])
                    
                    for i, emb in enumerate(embeddings_bloques):
                        embedding += emb * (pesos_bloques[i] / total_tokens)
            
            try:
                np.save(archivo_cache, embedding)
                if not os.path.exists(archivo_cache) or os.path.getsize(archivo_cache) == 0:
                    print(f"No se pudo guardar correctamente el embedding en {archivo_cache}")
            except Exception as e:
                print(f"Error al guardar en caché: {str(e)}")
            
            if ModelHandler._cache_misses % 100 == 0:
                archivo_caches = os.listdir(cache_dir)
                tamano_cache_MB = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in archivo_caches) / (1024 * 1024)
                print(f"Tamaño de caché en disco: {len(archivo_caches)} entradas, {tamano_cache_MB:.2f} MB")
            
            return embedding
            
        except Exception as e:
            print(f"Error al generar embeddings: {str(e)}")
            if usar_gemini:
                return np.zeros(768)
            else:
                return np.zeros(modelo.config.hidden_size)
            
            #Prueba con solo ultimos tokens
    """       if not isinstance(texto, str):
            texto = str(texto)
        
        # Truncar texto a aproximadamente 512 tokens (estimando 4 chars/token)
        limite_aprox_char =     
        if len(texto) > limite_aprox_char:
            texto = texto[-limite_aprox_char:]
        
        try:
            import google.generativeai as genai
            
            # Configurar API
            api_key = os.environ.get('GOOGLE_API_KEY', "")
            genai.configure(api_key=api_key)
            
            # Generar embeddings
            resultado = genai.embed_content(
                model="models/text-embedding-004",
                content=texto
            )
            embedding = np.array(resultado["embedding"])
            return embedding
            
        except Exception as e:
            print(f"Error con Gemini: {str(e)}")
            return np.zeros(768)"""
class LockedDropout(nn.Module):
        def __init__(self, tasa_dropout=0.2):
            super().__init__()
            self.tasa_dropout = tasa_dropout
            
        def forward(self, x):
            if not self.training or self.tasa_dropout == 0:
                return x
            
            if len(x.shape) == 2:
                # si es 2d reshape
                tamano_batch, input_size = x.size()
                mascara = x.new_ones(tamano_batch, 1)
                mascara = nn.functional.dropout(mascara, p=self.tasa_dropout, training=True)
                return x * mascara
            else:
                tamano_batch, longitud_secuencia, input_size = x.size()
                mascara = x.new_ones(tamano_batch, 1, input_size)
                mascara = nn.functional.dropout(mascara, p=self.tasa_dropout, training=True)
                return x * mascara
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduccion='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduccion = reduccion
        self.ce = nn.CrossEntropyLoss(reduccion='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduccion == 'mean':
            return focal_loss.mean()
        elif self.reduccion == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LSTMMultiClassifier(nn.Module):
    def __init__(self, dim_entrada, dim_oculta, tasa_dropout=Config.TASA_DROPOUT):
        super().__init__()
        self.dropout_entrada = LockedDropout(0.2)
        self.lstm = nn.LSTM(dim_entrada, dim_oculta // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(tasa_dropout)
        self.lstm2 = nn.LSTM(dim_oculta, dim_oculta // 2, batch_first=True, bidirectional=True)
        self.fc_dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(dim_oculta, 4)  # 4 dimensiones para multiclass
        self._inicializar_pesos()
    
    def _inicializar_pesos(self):
        """usa inicialización de Xavier"""
        for nombre, parametro in self.named_parameters():
            if 'weight' in nombre:
                nn.init.xavier_normal_(parametro, 0.2)
            elif 'bias' in nombre:
                nn.init.constant_(parametro, 0.0)
        
    def forward(self, x):
        x = self.dropout_entrada(x)

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        salida_lstm1, _ = self.lstm(x)
        salida_lstm1 = self.dropout(salida_lstm1)
        salida_lstm2, _ = self.lstm2(salida_lstm1)
        x = salida_lstm2[:, -1, :]
        x = self.fc_dropout(x)

        salida_final = self.fc(x)
        
        return salida_final
class LSTMRegressor(nn.Module):
    def __init__(self, embedding_dim, dim_oculta, tasa_dropout=0.5): 
        super().__init__()
        
        modelo, tokenizador = ModelHandler.cargar_modelo(Config.NOMBRE_MODELO)
        self.terminos_riesgo = self._inicializar_embedding_riesgo_lexico(modelo, tokenizador)
    
        self.entrada_dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=dim_oculta,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=tasa_dropout
        )
        
        self.atencion = RiskAttention(dim_oculta*2, self.terminos_riesgo)
        
        self.salida = nn.Sequential(
            nn.Linear(dim_oculta*4, dim_oculta),
            nn.LayerNorm(dim_oculta),  
            nn.GELU(),
            nn.Dropout(tasa_dropout),
            nn.Linear(dim_oculta, 1),
            nn.Sigmoid()
        )
        
        self.inicializar_pesos()
    
    def inicializar_pesos(self):
        """usa inicialización de Xavier"""
        for n, p in self.named_parameters():
            if 'lstm' not in n and p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.02)
                
    def _inicializar_embedding_riesgo_lexico(self, modelo, tokenizador):
        terminos_riesgo = [
            # --- Términos Clínicos y Diagnósticos (Indicadores de Problema Reconocido o Grave) ---
            "ludopatía", "ludopatia", 
            "juego patológico", "juego patologico", 
            "trastorno de juego", "trastorno del juego",
            "adicción al juego", "adiccion al juego", 
            "dependencia al juego", 
            "juego compulsivo", 
            "juego problemático", "juego problematico", 
            "trastorno del control de impulsos",
            "DSM-5 gambling disorder", 
            "juego patológico crónico", "trastorno de juego crónico",
            "comorbilidad adictiva", 
            "tolerancia al juego", 
            "síndrome de abstinencia de juego", "sintomatología de abstinencia", "sintomas de abstinencia",
            "craving de juego", 
            "impulso irresistible a jugar", 
            "ansia por jugar",
            "recaída", 
            "diagnóstico dual", "patología dual", 

            # --- Comportamientos Específicos de Juego Problemático (Indicadores Fuertes de Alto Riesgo) ---
            "perseguir pérdidas", "chasing losses", 
            "martingala", 
            "doblar apuesta", 
            "recuperar lo perdido", 
            "aumentar la apuesta", 
            "todo o nada", 
            "última oportunidad", "ultima oportunidad", 
            "jugar hasta el último céntimo", 
            "apostarlo todo",
            "doblar o nada",
            "retirada de fondos cancelada", 
            "depósito urgente", 
            "recarga de cuenta", 
            "volver a depositar",
            "jugar más tiempo del planeado", 
            "perder noción del tiempo", "perder la nocion del tiempo", 
            "sesiones maratónicas", "sesiones maratonicas", 
            "jugar toda la noche",
            "esconder el historial de apuestas", 
            "borrar historial", 
            "cuenta secreta",
            "perder el control de tiempo y dinero",

            # --- Terminología Financiera Asociada a Problemas Graves ---
            "deuda por juego", 
            "préstamo rápido", "prestamo rapido", 
            "crédito exprés", "credito expres", 
            "minicrédito", 
            "préstamo sin nómina", "prestamo sin nomina", 
            "préstamo con ASNEF", "prestamo con ASNEF", 
            "dinero urgente", 
            "pedir dinero prestado", 
            "línea de crédito", "linea de credito",
            "hipoteca", "segunda hipoteca",
            "anticipo de nómina", "anticipo de nomina", 
            "empeño", "casa de empeños", 
            "vender posesiones", 
            "descubierto bancario", 
            "embargo", 
            "quiebra personal", 
            "concurso de acreedores", 
            "morosidad", 
            "impago", 
            "deuda acumulada", 
            "prestamista", 
            "usura", 
            "interés abusivo", "interes abusivo",
            "deuda impagable", 
            "asfixia financiera", 
            "ruina económica",
            "insolvencia", 
            "sobreendeudamiento", 
            "incumplimiento de pagos",
            "préstamos encadenados", "prestamos encadenados",
            "malversación",
            "fraude para financiar juego",
            "estafa", 
            "pérdida de todos los ahorros", "pérdida total de ahorros",
            "vender casa para jugar",

            # --- Indicadores Psicológicos y Emocionales (A menudo Intensos en Alto Riesgo) ---
            "desesperación financiera", 
            "ansiedad por jugar", 
            "euforia de juego", 
            "pensamiento mágico", 
            "ilusión de control", "ilusion de control", 
            "falacia del jugador",
            "negación de problema", "negacion de problema", 
            "minimización del problema", "minimizacion del problema",
            "racionalización", 
            "culpabilidad", 
            "vergüenza", 
            "remordimiento", 
            "arrepentimiento",
            "irritabilidad",
            "agresividad",
            "idea suicida", 
            "pensamientos autodestructivos", 
            "desesperanza", 
            "impotencia",
            "depresión", "depresion", 
            "insomnio por juego", 
            "obsesión por recuperar", "obsesion por recuperar",
            "alteraciones del humor", 
            "crisis de ansiedad", 
            "ataques de pánico", 
            "ideación suicida", 
            "plan suicida",
            "intentos autolíticos", "intentos autoliticos",
            "sensación de vacío", "sensacion de vacio",

            # --- Comportamientos Sociales y Familiares Asociados (Indicadores de Impacto Grave) ---
            "mentir sobre el juego", 
            "ocultar pérdidas", 
            "esconder deudas", 
            "engañar a la familia",
            "inventar excusas", 
            "aislamiento social", 
            "descuido familiar", 
            "problemas laborales", 
            "absentismo laboral", 
            "pérdida de empleo", "perdida de empleo",
            "robo para jugar", 
            "apropiación indebida",
            "vender objetos personales", 
            "pedir prestado constantemente", 
            "romper relaciones", 
            "divorcio",
            "abandono de responsabilidades", 
            "negligencia parental", 
            "maltrato económico", 
            "violencia doméstica", "violencia domestica", 
            "abandono del hogar", 
            "pérdida de custodia", "pérdida de custodia por juego", 
            "desatención a hijos",
            "ausencia en eventos familiares", 
            "problemas con la justicia",
            "juicio por impago",
            "demanda por deuda",
            "prisión por delitos relacionados",
            "engañar a seres queridos",
            "hurto para apostar",
            "falsificar documentos", 
            "arruinar a la familia",

            # --- Términos de Tratamiento y Autoayuda (Implican Reconocimiento del Problema) ---
            "autoexclusión", "exclusión voluntaria", 
            "prohibición de entrada", "registro RGIAJ", 
            "terapia para ludopatía", "terapia para ludopatia", 
            "grupo de apoyo", 
            "jugadores anónimos", "jugadores anonimos",
            # "psicoterapia", "terapia cognitivo-conductual", 
            # "naltrexona para ludopatía", "ludopatia naltrexona", 

            # --- Términos Específicos de Modalidades (Betting, Trading, Crypto, Lootboxes) en Contexto de Riesgo ---
            # Betting:
            "apuestas deportivas", 
            "casa de apuestas", "bookie", 
            "apuesta en vivo", "live betting",
            "cashout", 
            "tipster",
            "pronóstico garantizado", "pick segura", "método infalible", 
            "apostador profesional", 
            "múltiples casas de apuestas", 
            "jugarse el sueldo", 
            "jugarse hasta la camisa",
            # Trading / Crypto:
            "trading", "day trading", 
            "trading compulsivo", 
            "inversión de alto riesgo", "inversion de alto riesgo",
            "opciones binarias", 
            "forex",
            "criptomonedas", 
            "apalancamiento", 
            "margin call", 
            "liquidación forzosa",
            "inversión compulsiva", "inversion compulsiva",
            "FOMO inversión", 
            "trading adictivo",
            "cripto gambling", 
            "juego especulativo",
            # Videojuegos / Lootboxes:
            "lootbox", "caja de botín", "cajita de botin", 
            "gacha", "gachapon", 
            "microtransacción", "micro transaccion",
            "compra in-app", 
            "pay-to-win",
            "skin betting", "apuestas de skins",
            "casino social",
            # --- Términos de Racionalización y Negación (Comunes en Adicción) ---
            "solo una vez más", "una sola vez más", 
            "última apuesta", "ultima apuesta", 
            "corazonada", "presentimiento", 
            "recuperar lo invertido", 
            "casi gano", "casi gané", "a punto de ganar", 
            "esta vez será diferente",
            "tengo el control", 
            "puedo parar cuando quiera", 
            "no tengo problema", 
            "juego responsable", 
            "solo juego por diversión", "solamente juego por diversion",
            "es solo entretenimiento", "entretenimiento solamente", 
            "no afecta mi vida",
            "solo juego con lo que me puedo permitir", 
            "distorsión cognitiva", 
            "pensamiento supersticioso",
            "minimización de consecuencias", 
            "justificación constante",
            "negación persistente", "negacion de problema",

            # --- Expresiones Coloquiales Indicativas de Alto Riesgo / Compulsión ---
            "meterle a la maquinita", 
            "jugar al parley", "hacer una combinada", 
            "apostar fuerte",
            "estar en racha", 
            "jugarse el sueldo",
            "quemarse", 
            "estar en la mala", 
            "jugarse hasta la camisa",
            "vicioso del juego",

            # --- Consecuencias Extremas (Indicadores Inequívocos de Gravedad) ---
            "suicidio por deudas", 
            "intento de suicidio", 
            "desahucio por juego", 
            "orden de alejamiento", 
            "problemas legales por deudas", 
            "robo a familiares", 

            # --- Signos de Frecuencia y Compulsión Elevada ---
            "juego diario", 
            "apuesta constante", 
            "jugar a todas horas", 
            "a cualquier hora",
            "primera cosa en la mañana", 
            "última cosa antes de dormir", 
            "revisar resultados obsesivamente", 
            "calcular apuestas constantemente", 
            "hablar solo de juego",
            "apostar en cualquier deporte", 
            "apostar sin conocimiento", 
            "apostar a cualquier cosa",
            "no poder ver deporte sin apostar",
        ]

        # Procesar cada término individualmente y combinar en un tensor 2D

        
        # Cachear embeddings
        archivo_cache = os.path.join(Config.DIRECTORIO_CACHE, "embeddings_riesgo.pt")
        if os.path.exists(archivo_cache):
            try:
                return torch.load(archivo_cache)
            except:
                pass
                
        # Procesar términos
        lista_embeddings = []
        for term in terminos_riesgo:
            embedding = ModelHandler.obtener_embeddings(term, modelo, tokenizador)
            lista_embeddings.append(embedding)
        
        stack_embeddings = np.stack(lista_embeddings, axis=0)
        embeddings_riesgo = torch.FloatTensor(stack_embeddings)
        
        os.makedirs(os.path.dirname(archivo_cache), exist_ok=True)
        torch.save(embeddings_riesgo, archivo_cache)
        
        return embeddings_riesgo

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [tamano_batch, 1, embedding_dim]
        
        x = self.entrada_dropout(x)
        x = self.layer_norm(x)
        
        lstm_salida, _ = self.lstm(x)
        
        contexto = self.atencion(lstm_salida)
        
        media_pool = lstm_salida.mean(dim=1)
        max_pool = lstm_salida.max(dim=1)[0]
        
        combinacion = torch.cat([media_pool, contexto], dim=1)
        
        return self.salida(combinacion)

class RiskAttention(nn.Module):
    def __init__(self, dim_oculta, embeddings_riesgo):
        super().__init__()
        self.embeddings_riesgo = nn.Parameter(embeddings_riesgo, requires_grad=True)
        self.proy_riesgo = nn.Linear(embeddings_riesgo.size(1), dim_oculta)
        
        self.query = nn.Linear(dim_oculta, dim_oculta)
        self.key = nn.Linear(dim_oculta, dim_oculta)
        self.value = nn.Linear(dim_oculta, dim_oculta)
        
        self.escala = nn.Parameter(torch.tensor(1.0 / (dim_oculta ** 0.5)))
        self.layer_norm = nn.LayerNorm(dim_oculta)
        
    def forward(self, lstm_output):
        q = self.query(lstm_output)  # [tamano_batch, longitud_secuencia, dim_oculta*2]
        k = self.key(lstm_output)    # [tamano_batch, longitud_secuencia, dim_oculta*2]
        v = self.value(lstm_output)  # [tamano_batch, longitud_secuencia, dim_oculta*2]

        riesgo_proy = self.proy_riesgo(self.embeddings_riesgo) # [num_terminos_riesgo, dim_oculta*2]

        # Calcular relevancia de riesgo léxico
        puntuacion_riesgo_lexico = torch.matmul(q, riesgo_proy.t())
        relevancia_riesgo_lexico = torch.sigmoid(puntuacion_riesgo_lexico.max(dim=-1, keepdim=True)[0]) # [tamano_batch, longitud_secuencia, 1]

        # Calcular self-attention estándar
        puntuacion_atencion = torch.bmm(q, k.transpose(1, 2)) * self.escala
        pesos_atencion = F.softmax(puntuacion_atencion, dim=-1)
        contexto = torch.bmm(pesos_atencion, v) # [tamano_batch, longitud_secuencia, dim_oculta*2]

        # Aplicar el boost por cada peso
        contexto_boosteado = contexto * (1.0 + relevancia_riesgo_lexico)
        contexto_boosteado = self.layer_norm(contexto_boosteado) # Normalizar después del boost

        return contexto_boosteado.mean(dim=1) 
class LSTMWrapper:
    """Contenedor para hacer los modelos LSTM compatibles con la interfaz de scikit-learn."""
    
    def __init__(self, modelo_regresion, modelo_multiclase):
        self.modelo_regresion = modelo_regresion
        self.modelo_multiclase = modelo_multiclase
        self.modelo_regresion.eval()
        self.modelo_multiclase.eval()
    
    def _prepare_input(self, X):
        """Prepara los datos de entrada"""
        if not isinstance(X, torch.Tensor):
            X = np.array(X)
            X = torch.from_numpy(X).float()
        return X
    
    def predecir(self, X):
        X_tensor = self._prepare_input(X)
        with torch.no_grad():
            salidas = self.modelo_regresion(X_tensor)
        return salidas.squeeze().cpu().numpy()
    
    def predecir_proba(self, X):
        X_tensor = self._prepare_input(X)
        with torch.no_grad():
            salidas = self.modelo_multiclase(X_tensor)
            probabilidades = torch.nn.functional.softmax(salidas, dim=1)
        return probabilidades.cpu().numpy()
    
    def predecir_combinado(self, X):
        X_tensor = self._prepare_input(X)
        
        with torch.no_grad():
            valores_regresion = self.modelo_regresion(X_tensor).squeeze().cpu().numpy()
            logits = self.modelo_multiclase(X_tensor)
            probabiliades_multiclase = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            
        
        return valores_regresion , probabiliades_multiclase

class GamblingPredictor:
    """Predice niveles de riesgo de ludopatía basados en datos de texto."""
    
    def __init__(self, modelo, tokenizador, clasificador_combinado):
        self.modelo = modelo
        self.tokenizador = tokenizador
        self.clasificador_combinado = clasificador_combinado
    
    def predecir(self, textos):
        resultados = []
        for texto in textos:
            if not isinstance(texto, str): # hay algun gracioso que mete un numero
                texto = str(texto)
            embedding = ModelHandler.obtener_embeddings(texto, self.modelo, self.tokenizador)
            valor_regresion, probabilidades_clase = self.clasificador_combinado.predecir_combinado([embedding])
            

            valor_regresion_simple = float(valor_regresion) if isinstance(valor_regresion, np.ndarray) else valor_regresion
            probabilidades_lista = [float(p) for p in probabilidades_clase[0]]
            
            resultado = {
                'texto': texto[:100] + "..." if len(texto) > 100 else texto,
                'valor_prediccion': round(valor_regresion_simple, 3),
                'probabilidades_clase': [round(p, 3) for p in probabilidades_lista],
            }
            resultados.append(resultado)
        return resultados
    
    def predecir_regresion(self, textos):
        resultados = []
        for texto in textos:
            if not isinstance(texto, str):
                texto = str(texto)
            embedding = ModelHandler.obtener_embeddings(texto, self.modelo, self.tokenizador)
            valor_regresion = self.clasificador_combinado.predecir([embedding])
            valor_regresion_simple = float(valor_regresion) if isinstance(valor_regresion, np.ndarray) else valor_regresion
            resultado = {
                'texto': texto[:100] + "..." if len(texto) > 100 else texto,
                'valor_prediccion': round(valor_regresion_simple, 3)
            }
            resultados.append(resultado)
        return resultados

    def predecir_multiclase(self, textos):
        resultados = []
        for texto in textos:
            if not isinstance(texto, str):  
                texto = str(texto)
            embedding = ModelHandler.obtener_embeddings(texto, self.modelo, self.tokenizador)
            probabilidades = self.clasificador_combinado.predecir_proba([embedding])
            probabilidades_lista = [float(p) for p in probabilidades[0]]
            resultado = {
                'texto': texto[:100] + "..." if len(texto) > 100 else texto,
                'probabilidades_clase': [round(p, 3) for p in probabilidades_lista]
            }
            resultados.append(resultado)
        return resultados 
    @staticmethod
    def entrenar_modelos_combinados(X_entren, y_entren, y_multiclase, X_dev=None, y_dev=None, y_dev_multi=None):
        # metricas_kfold = GamblingPredictor.evaluar_kfold(X_entren, y_entren, y_multiclase)
        #print("Completada evaluación k-fold. Entrenando modelo final con todos los datos...")
        
        #training tensores
        X_entren_tensor = torch.from_numpy(X_entren).float()
        y_entren = np.array(y_entren)
        y_entren_tensor = torch.from_numpy(y_entren).float()
        
        y_multiclase = np.array(y_multiclase)
        labels_unicas = np.unique(y_multiclase)
        label_a_idx = {label: idx for idx, label in enumerate(labels_unicas)}
        print(label_a_idx)
        y_multiclase_indices = np.array([label_a_idx[label] for label in y_multiclase])
        y_multiclase_tensor = torch.from_numpy(y_multiclase_indices).long()
        
        #dev tensores
        hay_dev = X_dev is not None and y_dev is not None and y_dev_multi is not None
        if hay_dev:
            X_dev_tensor = torch.from_numpy(np.array(X_dev)).float()
            y_dev_tensor = torch.from_numpy(np.array(y_dev)).float()
            y_dev_multi_indices = np.array([label_a_idx[label] for label in y_dev_multi])
            y_dev_multi_tensor = torch.from_numpy(y_dev_multi_indices).long()
            print(f"Dev con {len(X_dev)} usuarios")
            
            #dev dataloaders
            conjunto_dev_regresion = TensorDataset(X_dev_tensor, y_dev_tensor)
            cargador_dev_regresion = TorchDataLoader(conjunto_dev_regresion, batch_size=32, shuffle=False)
            
            conjunto_dev_multiclase = TensorDataset(X_dev_tensor, y_dev_multi_tensor)
            cargador_dev_multiclase = TorchDataLoader(conjunto_dev_multiclase, batch_size=32, shuffle=False)
        
        #training dataloaders
        conjunto_entren_regresion = TensorDataset(X_entren_tensor, y_entren_tensor)
        cargador_entren_regresion = TorchDataLoader(conjunto_entren_regresion, batch_size=32, shuffle=True)
        
        conjunto_entren_multiclase = TensorDataset(X_entren_tensor, y_multiclase_tensor)
        cargador_entren_multiclase = TorchDataLoader(conjunto_entren_multiclase, batch_size=32, shuffle=True)
        dim_entrada = X_entren.shape[1]
        
        
        # ----------------------- GroupDRO -----------------------
        class GroupDROLoss(nn.Module):
            """
            Función de perdida GroupDRO
            """
            def __init__(self, num_grupos=3, eta=0.005, alpha=0.7, tipo_modelo='regression'):
                super().__init__()
                self.num_grupos = num_grupos
                self.eta = eta  
                self.alpha = alpha 
                self.tipo_modelo = tipo_modelo  # 'regression' o 'multiclass'
                
                self.register_buffer('pesos_grupos', torch.ones(num_grupos) / num_grupos)
                self.register_buffer('loss_grupos_EMA', torch.zeros(num_grupos))
                self.recuento_grupos = torch.zeros(num_grupos)
                self.smoothing = 0.01  
                self.register_buffer('loss_grupos_historico', torch.zeros(num_grupos))
                
                if tipo_modelo == 'regression':
                    self.loss_base = nn.MSELoss(reduction='none')
                else:
                    self.loss_base = nn.CrossEntropyLoss(reduction='none')
    
            def forward(self, salida, targets):
                if isinstance(salida, tuple):
                    salida = salida[0]
                elif not isinstance(salida, torch.Tensor):
                    salida = torch.tensor(salida)
                
                if self.tipo_modelo == 'regression':
                    loss_individuales = self.loss_base(salida, targets)
                    errores = torch.abs(salida - targets)
                else:
                    loss_individuales = self.loss_base(salida, targets)
                    obj_logits = salida[torch.arange(salida.size(0)), targets]
                    salida_ajustada = salida.clone()
                    salida_ajustada[torch.arange(salida.size(0)), targets] = float('-inf')
                    max_error_logits, _ = torch.max(salida_ajustada, dim=1)
                    errores = max_error_logits - obj_logits
    
                cuantiles = torch.tensor([i / self.num_grupos for i in range(1, self.num_grupos)], device=salida.device)
                umbrales = torch.quantile(errores, cuantiles)
                
                grupos = torch.zeros_like(errores, dtype=torch.long)
                for g in range(1, self.num_grupos):
                    grupos = torch.where(errores >= umbrales[g-1], torch.tensor(g, device=salida.device), grupos)
                
                loss_grupos = torch.zeros(self.num_grupos, device=salida.device)
                for g in range(self.num_grupos):
                    mascara = (grupos == g)
                    if mascara.sum() > 0:
                        loss_grupos[g] = loss_individuales[mascara].mean()
                    else:
                        loss_grupos[g] = torch.tensor(0.0, device=salida.device)
                
                # Actualización de los pesos de grupo solo en modo entrenamiento
                if self.training:
                    with torch.no_grad():
                        escala_grad = torch.clamp(loss_grupos, min=-10.0, max=10.0)
                        self.loss_grupos_historico = self.alpha * self.loss_grupos_historico + (1 - self.alpha) * escala_grad
                        logits = torch.log(self.pesos_grupos + 1e-8) + self.eta * self.loss_grupos_historico
                        self.pesos_grupos = torch.softmax(logits, dim=0)
                
                loss_ponderada = (loss_grupos * self.pesos_grupos).sum()
                reg_term = 0.01 * (-torch.log(self.pesos_grupos + 1e-8)).mean()
                
                return loss_ponderada + reg_term


        modelo_regresion = LSTMRegressor(dim_entrada, Config.DIMENSION_OCULTA)
        # Para el modelo de regresión (devuelve prob1)
        criterio_regresion = GroupDROLoss(num_grupos=5, tipo_modelo='regression')
        optimizador_regresion = optim.AdamW(modelo_regresion.parameters(), lr=0.005)
        scheduler_regresion = optim.lr_scheduler.ReduceLROnPlateau(
            optimizador_regresion,
            mode='min',
            factor=0.5,
            patience=15,
            verbose=True,
            min_lr=1e-5
        )
    
        mejor_perdida_dev_reg = float('inf')
        mejor_modelo_reg_state = None
        
        for epoca in range(Config.EPOCAS):
            modelo_regresion.train()
            perdida_total = 0.0
            for lote_X, lote_y in cargador_entren_regresion:
                optimizador_regresion.zero_grad()
                salidas = modelo_regresion(lote_X)
                salidas_reshaped = salidas.view(-1)
                lote_y_reshaped = lote_y.view(-1)
                perdida = criterio_regresion(salidas_reshaped, lote_y_reshaped)
                perdida.backward()
                torch.nn.utils.clip_grad_norm_(modelo_regresion.parameters(), max_norm=1.0)
                optimizador_regresion.step()
                perdida_total += perdida.item()
            
            if hay_dev:
                modelo_regresion.eval()
                with torch.no_grad():
                    perdida_dev_total = 0.0
                    for lote_X_dev, lote_y_dev in cargador_dev_regresion:
                        salidas_dev = modelo_regresion(lote_X_dev)
                        perdida_dev = criterio_regresion(salidas_dev.view(-1), lote_y_dev.view(-1))
                        perdida_dev_total += perdida_dev.item()
                    perdida_dev_promedio = perdida_dev_total / len(cargador_dev_regresion)
                
                scheduler_regresion.step(perdida_dev_promedio)
                if perdida_dev_promedio < mejor_perdida_dev_reg:
                    mejor_perdida_dev_reg = perdida_dev_promedio
                    mejor_modelo_reg_state = modelo_regresion.state_dict().copy()
                
                if (epoca + 1) % 10 == 0:
                    lr_actual = optimizador_regresion.param_groups[0]['lr']
                    print(f"Época {epoca+1}/{Config.EPOCAS}, Pérdida Regresión: {perdida_total/len(cargador_entren_regresion):.4f}, Pérdida Dev: {perdida_dev_promedio:.4f}, LR: {lr_actual:.6f}")
            else:
                scheduler_regresion.step(perdida_total/len(cargador_entren_regresion))
                if (epoca + 1) % 10 == 0:
                    lr_actual = optimizador_regresion.param_groups[0]['lr']
                    print(f"Época {epoca+1}/{Config.EPOCAS}, Pérdida Regresión: {perdida_total/len(cargador_entren_regresion):.4f}, LR: {lr_actual:.6f}")
        
        if hay_dev and mejor_modelo_reg_state is not None:
            modelo_regresion.load_state_dict(mejor_modelo_reg_state)
            print(f"Cargado mejor modelo de regresión con pérdida dev: {mejor_perdida_dev_reg:.4f}")
        
        # ---------------------- Entrenamiento para el modelo multiclase ----------------------
        modelo_multiclase = LSTMMultiClassifier(dim_entrada, Config.DIMENSION_OCULTA)
        # Se puede usar GroupDROLoss
        criterio_multiclase = nn.CrossEntropyLoss()
        optimizador_multiclase = optim.AdamW(modelo_multiclase.parameters(), lr=0.005)
        scheduler_multiclase = optim.lr_scheduler.ReduceLROnPlateau(
            optimizador_multiclase,
            mode='min',
            factor=0.5,
            patience=15,
            verbose=True,
            min_lr=1e-5
        )
        
        mejor_perdida_dev_multi = float('inf')
        mejor_modelo_multi_state = None
        print("\nEntrenando modelo multiclase...")
        
        for epoca in range(Config.EPOCAS):
            modelo_multiclase.train()
            perdida_total = 0.0
            for lote_X, lote_y in cargador_entren_multiclase:
                optimizador_multiclase.zero_grad()
                salidas = modelo_multiclase(lote_X)
                perdida = criterio_multiclase(salidas, lote_y)
                perdida.backward()
                torch.nn.utils.clip_grad_norm_(modelo_multiclase.parameters(), max_norm=1.0)
                optimizador_multiclase.step()
                perdida_total += perdida.item()
            
            if hay_dev:
                modelo_multiclase.eval()
                with torch.no_grad():
                    perdida_dev_total = 0.0
                    for lote_X_dev, lote_y_dev in cargador_dev_multiclase:
                        salidas_dev = modelo_multiclase(lote_X_dev)
                        perdida_dev = criterio_multiclase(salidas_dev, lote_y_dev)
                        perdida_dev_total += perdida_dev.item()
                    perdida_dev_promedio = perdida_dev_total / len(cargador_dev_multiclase)
                
                scheduler_multiclase.step(perdida_dev_promedio)
                if perdida_dev_promedio < mejor_perdida_dev_multi:
                    mejor_perdida_dev_multi = perdida_dev_promedio
                    mejor_modelo_multi_state = modelo_multiclase.state_dict().copy()
                
                if (epoca + 1) % 10 == 0:
                    lr_actual = optimizador_multiclase.param_groups[0]['lr']
                    print(f"Época {epoca+1}/{Config.EPOCAS}, Pérdida Multiclase: {perdida_total/len(cargador_entren_multiclase):.4f}, Pérdida Dev: {perdida_dev_promedio:.4f}, LR: {lr_actual:.6f}")
            else:
                scheduler_multiclase.step(perdida_total/len(cargador_entren_multiclase))
                if (epoca + 1) % 10 == 0:
                    lr_actual = optimizador_multiclase.param_groups[0]['lr']
                    print(f"Época {epoca+1}/{Config.EPOCAS}, Pérdida Multiclase: {perdida_total/len(cargador_entren_multiclase):.4f}, LR: {lr_actual:.6f}")
        
        if hay_dev and mejor_modelo_multi_state is not None:
            modelo_multiclase.load_state_dict(mejor_modelo_multi_state)
            print(f"Cargado mejor modelo multiclase con pérdida dev: {mejor_perdida_dev_multi:.4f}")
        
        # Cambiar a modo evaluación
        modelo_regresion.eval()
        modelo_multiclase.eval()
        
        envoltorio_combinado = LSTMWrapper(modelo_regresion, modelo_multiclase)
    
        modelo_regresion_path = os.path.join(Config.DIRECTORIO_MODELOS, 'lstm_regresion.pt')
        modelo_multiclase_path = os.path.join(Config.DIRECTORIO_MODELOS, 'lstm_multiclase.pt')
    
        os.makedirs(Config.DIRECTORIO_MODELOS, exist_ok=True)
    
        torch.save(modelo_regresion.state_dict(), modelo_regresion_path)
        torch.save(modelo_multiclase.state_dict(), modelo_multiclase_path)
        print(f"Modelos guardados en {Config.DIRECTORIO_MODELOS}")
        
        return envoltorio_combinado

    @staticmethod
    def cargar_modelos_combinados(X_entren, y_entren, y_multiclase):
        X_entren = np.array(X_entren)   
        dim_entrada = X_entren.shape[1]
        try:
            modelo_regresion_path = os.path.join(Config.DIRECTORIO_MODELOS, 'lstm_regresion.pt')
            modelo_multiclase_path = os.path.join(Config.DIRECTORIO_MODELOS, 'lstm_multiclase.pt')
            
            # revisar si los modelos ya existen
            if os.path.exists(modelo_regresion_path) and os.path.exists(modelo_multiclase_path):
                print("Cargando modelos entrenados previamente...")
                
                modelo_regresion = LSTMRegressor(dim_entrada, Config.DIMENSION_OCULTA)
                modelo_multiclase = LSTMMultiClassifier(dim_entrada, Config.DIMENSION_OCULTA)
                
                # cargar pesos
                modelo_regresion.load_state_dict(torch.load(modelo_regresion_path))
                modelo_multiclase.load_state_dict(torch.load(modelo_multiclase_path))
                
                # MODO EVALUACIÓN
                modelo_regresion.eval()
                modelo_multiclase.eval()
                
                return LSTMWrapper(modelo_regresion, modelo_multiclase)
            else:
                print("No se encontraron modelos pre-entrenados. Entrenando nuevos modelos...")
                return GamblingPredictor.entrenar_modelos_combinados(X_entren, y_entren, y_multiclase)
        except Exception as e:
            print(f"Error al cargar los modelos pre-entrenados: {e}")
            print("Intentando entrenar nuevos modelos...")
            return GamblingPredictor.entrenar_modelos_combinados(X_entren, y_entren, y_multiclase)


    
def main():
    """Función principal que gestiona el análisis de riesgo de ludopatía."""
    modelo, tokenizador = ModelHandler.cargar_modelo()
    
    print("Cargando datos de entrenamiento...")
    id_usuarios, etiquetas = DataProcessor.cargar_etiquetas("./MentalRisk2025/task1/train/gold_task1.txt")
    datos_usuarios = DataProcessor.cargar_datos_usuarios("./MentalRisk2025/task1/train/subjects/", id_usuarios)
    
    multi_id_usuarios, multi_etiquetas = DataProcessor.cargar_etiquetas("./MentalRisk2025/task2/train/gold_task2.txt")
    labels_unicas = np.unique(multi_etiquetas)
    label_a_idx = {label: idx for idx, label in enumerate(labels_unicas)}
    print(label_a_idx)
    
    print(f"Total de usuarios: {len(datos_usuarios)}")

    print("Procesando textos...")
    textos = []
    for mensajes_usuario in datos_usuarios:
        lista_mensajes = []
        for mensaje in mensajes_usuario:
            if isinstance(mensaje, dict) and 'message' in mensaje:
                mensaje_con_fecha = mensaje.copy()
                if 'date' not in mensaje_con_fecha:
                    mensaje_con_fecha['date'] = ''  
                lista_mensajes.append(mensaje_con_fecha)
        textos.append(lista_mensajes) 
    print("Generando embeddings...")

    indices = np.arange(len(textos))

    def input_with_timeout(prompt, timeout=3):
        resultado = [None]
        
        def get_input():
            resultado[0] = input(prompt)
        
        thread = threading.Thread(target=get_input)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            print("\nTiempo agotado. Continuando con la ejecución por defecto.")
            return "si"  
        return resultado[0]
    
    confirmacion = input_with_timeout("¿Desea realizar 3 ejecuciones con diferentes train-test splits? (escriba 'si' o 'y' para continuar, tiempo de espera: 3 segundos): ")
    if confirmacion.lower() in ['si', 'y', 's']:
        resultados = []
        for i in range(3):
            print(f"\n=== Ejecución {i+1}/3 ===")
            
            indices_originales = []
            indices_aumentados = {}  

            for idx, id_usuario in enumerate(id_usuarios):
                if '+' in id_usuario:
                    original_id = id_usuario.split('+')[1]
                    if original_id not in indices_aumentados:
                        indices_aumentados[original_id] = []
                    indices_aumentados[original_id].append(idx)
                else:
                    indices_originales.append(idx)
            print(indices_aumentados)
            print(f"Datos originales: {len(indices_originales)}, Usuarios con datos aumentados: {len(indices_aumentados)}")

            etiquetas_originales = [etiquetas[idx] for idx in indices_originales]
            multi_etiquetas_originales = [multi_etiquetas[idx] for idx in indices_originales]

            # Combinación para estratificación conjunta
            estratificacion = np.array([f"{a}_{b}" for a, b in zip(etiquetas_originales, multi_etiquetas_originales)])

            # Dividir SOLO los datos originales en train+dev y test
            indices_orig_train_dev, indices_orig_test, _, _, _, _ = train_test_split(
                indices_originales, etiquetas_originales, multi_etiquetas_originales, 
                test_size=0.1, random_state=SEMILLA+i,
                stratify=estratificacion
            )

            # Luego dividir train+dev en train y dev
            y_orig_train_dev = [etiquetas[idx] for idx in indices_orig_train_dev]
            multi_orig_train_dev = [multi_etiquetas[idx] for idx in indices_orig_train_dev]
            estratificacion_train_dev = np.array([f"{a}_{b}" for a, b in zip(y_orig_train_dev, multi_orig_train_dev)])

            indices_orig_train, indices_orig_dev, _, _, _, _ = train_test_split(
                indices_orig_train_dev, y_orig_train_dev, multi_orig_train_dev,
                test_size=0.1, random_state=SEMILLA+i+100,  # ~10% del dataset original
                stratify=estratificacion_train_dev
            )

            ids_usuarios_train = {id_usuarios[idx] for idx in indices_orig_train}
            ids_usuarios_dev = {id_usuarios[idx] for idx in indices_orig_dev}
            ids_usuarios_test = {id_usuarios[idx] for idx in indices_orig_test}

            indices_train_aug = []
            for orig_id, aug_indices in indices_aumentados.items():
                if orig_id in ids_usuarios_train:
                    indices_train_aug.extend(aug_indices)

            indices_train = indices_orig_train + indices_train_aug
            indices_dev = indices_orig_dev
            indices_test = indices_orig_test

            # Crear los conjuntos de datos finales
            y_train = [etiquetas[idx] for idx in indices_train]
            y_dev = [etiquetas[idx] for idx in indices_dev]
            y_test = [etiquetas[idx] for idx in indices_test]

            multi_train = [multi_etiquetas[idx] for idx in indices_train]
            multi_dev = [multi_etiquetas[idx] for idx in indices_dev]
            multi_test = [multi_etiquetas[idx] for idx in indices_test]

            print(f"División final - Train: {len(indices_train)} ({len(indices_orig_train)} orig + {len(indices_train_aug)} aug), Dev: {len(indices_dev)}, Test: {len(indices_test)}")
            print(f"Datos aumentados eliminados para {len(indices_aumentados) - sum(1 for id in indices_aumentados if id in ids_usuarios_train)} usuarios que estaban en test/dev")

            print("Generando embeddings para conjunto de entrenamiento...")
            X_train = np.array([
                ModelHandler.obtener_embeddings(" ".join([
                    f"{msg.get('date', '')} {msg['message']}" for msg in textos[idx]
                ]), modelo, tokenizador)
                for idx in indices_train
            ])

            print("Generando embeddings para conjunto de validación...")
            X_dev = np.array([
                ModelHandler.obtener_embeddings(" ".join([
                    f"{msg.get('date', '')} {msg['message']}" for msg in textos[idx]
                ]), modelo, tokenizador)
                for idx in indices_dev
            ])

            print("Generando embeddings para conjunto de prueba...")
            X_test = np.array([
                ModelHandler.obtener_embeddings(" ".join([
                    f"{msg.get('date', '')} {msg['message']}" for msg in textos[idx]
                ]), modelo, tokenizador)
                for idx in indices_test
            ])

            print("Distribución de los Datos")
            y_train = np.array(y_train)
            y_dev = np.array(y_dev)
            y_test = np.array(y_test)
            etiquetas = np.array(etiquetas)
            porcentaje_tarea_binaria = {
                'Dataset': ['Train', 'Dev', 'Test', 'Total'],
                'No riesgo (0)': [np.sum(y_train == 0), np.sum(y_dev == 0), np.sum(y_test == 0), np.sum(etiquetas == 0)],
                'riesgo (1)': [np.sum(y_train == 1), np.sum(y_dev == 1), np.sum(y_test == 1), np.sum(etiquetas == 1)],
                'Total': [len(y_train), len(y_dev), len(y_test), len(etiquetas)]
            }
            print(pd.DataFrame(porcentaje_tarea_binaria).to_string(index=False))
            
            multi_labels = sorted(np.unique(multi_etiquetas))
            multi_recuento = {'Dataset': ['Train', 'Dev', 'Test', 'Total']}
            
            for label in multi_labels:
                multi_recuento[f'Clase {label}'] = [
                    np.sum(np.array(multi_train) == label), 
                    np.sum(np.array(multi_dev) == label), 
                    np.sum(np.array(multi_test) == label),
                    np.sum(np.array(multi_etiquetas) == label)
                ]
            
            multi_recuento['Total'] = [len(multi_train), len(multi_dev), len(multi_test), len(multi_etiquetas)]
            
            print(pd.DataFrame(multi_recuento).to_string(index=False))
            print(f"Train conjunto: {X_train.shape[0]} muestras, Dev conjunto: {X_dev.shape[0]} muestras, Test conjunto: {X_test.shape[0]} muestras")
            
            test_usuarios = [id_usuarios[idx] for idx in indices_test]
            
            # Create gold test dictionaries using the exact IDs from the original data
            dict_gold_test = {}
            dict_gold_test_multi = {}
            usuario_id_lista = []
            for a, usuario_idx in enumerate(indices_test):
                usuario_id = id_usuarios[usuario_idx]
                dict_gold_test[usuario_id] = y_test[a]
                dict_gold_test_multi[usuario_id] = multi_test[a]
                usuario_id_lista.append(usuario_id)
            if len(dict_gold_test) != len(test_usuarios) or len(dict_gold_test_multi) != len(test_usuarios):
                print(f"Advertencia: Gold tamaño diccionario ({len(dict_gold_test)}, {len(dict_gold_test_multi)}) " 
                      f"no cuadra con el test({len(test_usuarios)})")

            if not set(dict_gold_test.keys()).issubset(set(id_usuarios)):
                raise Exception("Error: Algunas IDs en el gold test no se encuentran en la lista original de usuarios.")
            
            if set(dict_gold_test.keys()) != set(dict_gold_test_multi.keys()):
                raise Exception("Error: Las IDs entre el gold binario y multiclase no coinciden.")
            
            
            print("Entrenando modelos combinados...")
            clasificador_combinado = GamblingPredictor.entrenar_modelos_combinados(
                X_train, y_train, multi_train, X_dev, y_dev, multi_dev
            )
            predictor = GamblingPredictor(modelo, tokenizador, clasificador_combinado)
            
            print("\n=== EVALUACIÓN EN TEST conjunto COMPLETO ANTES DE DIVIDIR POR RONDAS ===")

            with torch.no_grad():
                predicciones_binarias_raw = clasificador_combinado.predecir(X_test)
                predicciones_binarias = np.array([1 if p >= 0.65 else 0 for p in predicciones_binarias_raw])

                
                multi_probs = clasificador_combinado.predecir_proba(X_test)
                multi_preds = np.argmax(multi_probs, axis=1)


            binario_f1_macro = f1_score(y_test, predicciones_binarias, average='macro')
            multi_f1_macro = f1_score([label_a_idx[label] for label in multi_test], multi_preds, average='macro')

            print(f"Rendimiento en test conjunto completo:")
            print(f"- Binario F1 Macro: {binario_f1_macro:.4f}")
            print(f"- Multiclase F1 Macro: {multi_f1_macro:.4f}")
            print("\nMatriz de confusión y reporte completo:")
            print("\nBinario Classification Reporte:")
            print(classification_report(y_test, predicciones_binarias))
            print("\nMulticlase Classification Reporte:")
            print(classification_report([label_a_idx[label] for label in multi_test], multi_preds))
            

            print("Esperando 1 segundo...")
            time.sleep(1)
            
            test_textos = [textos[idx] for idx in indices_test]

            servidor_fake = ServidorExp(
                textos_test=test_textos, 
                id=usuario_id_lista
            )
            
            
            usuarios_riesgo = {}
            mensajes_acumulados = {}  
            usuarios_clases = {}
            
            assert id_usuarios == multi_id_usuarios, "Los IDs de usuario de task1 y task2 no coinciden."
            ronda_actual = 0
            while servidor_fake.hay_siguiente_ronda():
                ronda_actual += 1
                print(f"  Procesando ronda {ronda_actual}...")
                mensajes_ronda, num_ronda = servidor_fake.siguiente_ronda()
                predicciones_ronda = []
                
                for usuario_id, mensaje_info in mensajes_ronda:
                    texto = f"{mensaje_info.get('date', '')} {mensaje_info.get('message', '')}"
                    if usuario_id not in usuarios_riesgo:
                        usuarios_riesgo[usuario_id] = 0
                        usuarios_clases[usuario_id] = [0, 0, 0, 0]
                        mensajes_acumulados[usuario_id] = []
                    
                    
                    
                    mensajes_acumulados[usuario_id].append(texto)
                
                    texto_acumulado_hasta_ahora = " ".join([str(mensaje) for mensaje in mensajes_acumulados[usuario_id]])
                    
                    if usuario_id not in test_usuarios:
                        raise Exception(f"ID de usuario {usuario_id} no encontrado en la lista de test_usuarios")
                    
                    resultado = predictor.predecir_regresion([texto_acumulado_hasta_ahora])[0]
                    prob_binaria = resultado['valor_prediccion']
                    
                    usuarios_riesgo[usuario_id] += prob_binaria
                    """if texto in mensajes_acumulados[usuario_id]:
                        raise Exception(f"Mensaje duplicado encontrado para el usuario {usuario_id}: '{texto[:500]}...'")"""
                    
                    #solo se hace para guardar el resumen
                    mensajes_saneados = []
                    for msg in mensajes_acumulados[usuario_id]:
                        if isinstance(msg, str):
                            # Reemplazar caracteres problemáticos como emojis
                            msg_saneado = ''.join(c if ord(c) < 0x10000 and c.isprintable() else ' ' for c in msg)
                            mensajes_saneados.append(msg_saneado)
                        else:
                            mensajes_saneados.append(str(msg))
                    
                    predicciones_ronda.append({
                        'usuario_id': usuario_id,
                        'mensaje': mensajes_saneados,
                        'riesgo_ludopatia': prob_binaria
                    })
                
                decisiones_usuarios = []
                dict_riesgo = {}
                
                for usuario_id, suma_riesgo in usuarios_riesgo.items():
                    es_riesgo = suma_riesgo >= 3 + (ronda_actual / 6)
                    
                    dict_riesgo[usuario_id] = 1 if es_riesgo else 0
                
                resultado_eval = {
                    "predictions": dict_riesgo,
                    # No incluimos "type" aquí para multiclass
                }
                
                os.makedirs("resultadosPredictor", exist_ok=True)
                pd.DataFrame(predicciones_ronda).to_json(
                    f"resultadosPredictor/ronda_{num_ronda}_predicciones_ejecucion_{i+1}.json", 
                    index=False,
                    orient='records'
                )
                
                Evaluacion = binaryClassification(gold=dict_gold_test, json_data=resultado_eval, round=ronda_actual)
                Evaluacion.eval_performance()
            
            print(f"\n=== RESUMEN FINAL EJECUCIÓN {i+1} ===")
            print("Calculando predicciones multiclase finales con todos los mensajes acumulados...")
            dict_multi_final = {}
            for usuario_id, mensajes in mensajes_acumulados.items():
                texto_completo = " ".join([str(mensaje) for mensaje in mensajes])
                with torch.no_grad():
                    resultado = predictor.predecir_multiclase([texto_completo])
                probabilidades_clase = resultado[0]['probabilidades_clase']
                clase_predicha = np.argmax(probabilidades_clase)
                dict_multi_final[usuario_id] = clase_predicha
            resumen_final = []
            dict_riesgo_final = {}
            
            for usuario_id, suma_riesgo in usuarios_riesgo.items():
                es_riesgo = suma_riesgo >= 3 + (ronda_actual / 6)
                clase_predicha = dict_multi_final[usuario_id]
                
                dict_riesgo_final[usuario_id] = 1 if es_riesgo else 0
                
                resumen_final.append({
                    'usuario_id': usuario_id,
                    'decision_final': 1 if es_riesgo else 0,
                    'clase_final': clase_predicha,
                    'total_mensajes_riesgo': suma_riesgo
                })
            
            resultadofinal = {
                "predictions": dict_riesgo_final,
                "type": dict_multi_final
            }
            
            Evaluacion_multi = multiclassClassification(gold=dict_gold_test_multi, json_data=resultadofinal, labid=label_a_idx)
            eval_metricas_multi = Evaluacion_multi.eval_performance()
            
            pd.DataFrame(resumen_final).to_csv(f"resultadosPredictor/resumen_final_ejecucion_{i+1}.csv", index=False)
            print(f"\n Ejecución {i+1} completada - resultados guardados")
            
            eval_metricas_binarias = Evaluacion.eval_performance()
            
            resultados.append({
                'ejecucion': i+1,
                'ronda_final': ronda_actual,
                'dict_riesgo': dict_riesgo_final,
                'dict_multi': dict_multi_final,
                'metricas_binarias': eval_metricas_binarias,
                'metricas_multi': eval_metricas_multi
            })
        
            print("\n=== ANÁLISIS FINAL DE LAS 3 EJECUCIONES ===")
            binarias_metricas = {}
            multi_metricas = {}
            
            for resultado in resultados:
                bin_metricas = resultado.get('metricas_binarias', {})
                mult_metricas = resultado.get('metricas_multi', {})
                
                if bin_metricas:
                    for key, value in bin_metricas.items():
                        if key not in binarias_metricas:
                            binarias_metricas[key] = []
                        if isinstance(value, (int, float)):
                            binarias_metricas[key].append(value)
                
                if mult_metricas:
                    for key, value in mult_metricas.items():
                        if key not in multi_metricas:
                            multi_metricas[key] = []
                        if isinstance(value, (int, float)):
                            multi_metricas[key].append(value)
            
            print("\nESTADÍSTICAS DE MÉTRICAS BINARIAS:")
            for metric_name, valores in binarias_metricas.items():
                if valores:
                    media_val = np.mean(valores)
                    desv_val = np.std(valores)
                    print(f"{metric_name:<15}: Media = {media_val:.4f}, Desviación Estándar = {desv_val:.4f}")
            
            print("\nESTADÍSTICAS DE MÉTRICAS MULTICLASE:")
            for metric_name, valores in multi_metricas.items():
                if valores:
                    media_val = np.mean(valores)
                    desv_val = np.std(valores)
                    print(f"{metric_name:<15}: Media = {media_val:.4f}, Desviación Estándar = {desv_val:.4f}")
            
            estadisticas = {
                'binarias': {metricas: {'media': np.mean(valores), 'std': np.std(valores)} 
                             for metricas, valores in binarias_metricas.items() if valores},
                'multiclase': {metricas: {'media': np.mean(valores), 'std': np.std(valores)} 
                               for metricas, valores in multi_metricas.items() if valores}
            }
            pd.DataFrame(estadisticas).to_json('resultadosPredictor/estadisticas_ejecuciones.json', orient='index')
            print("\nEstadísticas guardadas en 'resultadosPredictor/estadisticas_ejecuciones.json'")
    else:
        print("Proceso finalizado. No se realizará la evaluación con splits múltiples.")



######################### Función para entrenar y guardar el modelo completo para la entrega final ###########################



    
def entrenar_y_guardar_modelo(ruta_guardado="./modelos_entrenados_finales"):
    
    modelo, tokenizador = ModelHandler.cargar_modelo()
    
    print("Cargando datos de entrenamiento...")
    id_usuarios, etiquetas = DataProcessor.cargar_etiquetas("./MentalRisk2025/task1/train/gold_task1.txt")
    datos_usuarios = DataProcessor.cargar_datos_usuarios("./MentalRisk2025/task1/train/subjects/", id_usuarios)
    
    multi_id_usuarios, multi_etiquetas = DataProcessor.cargar_etiquetas("./MentalRisk2025/task2/train/gold_task2.txt")
    labels_unicas = np.unique(multi_etiquetas)
    label_a_idx = {label: idx for idx, label in enumerate(labels_unicas)}
    print(f"Etiquetas multiclase: {label_a_idx}")
    
    print(f"Total de usuarios: {len(datos_usuarios)}")
    
    print("Procesando textos...")
    textos = []
    for mensajes_usuario in datos_usuarios:
        lista_mensajes = []
        for mensaje in mensajes_usuario:
            if isinstance(mensaje, dict) and 'message' in mensaje:
                mensaje_con_fecha = mensaje.copy()
                if 'date' not in mensaje_con_fecha:
                    mensaje_con_fecha['date'] = '' 
                lista_mensajes.append(mensaje_con_fecha)
        textos.append(lista_mensajes)
    
    print("Generando embeddings...")
    indices = np.arange(len(textos))

    indices_originales = []
    indices_aumentados = {}

    for idx, id_usuario in enumerate(id_usuarios):
        if '+' in id_usuario:
            original_id = id_usuario.split('+')[1]
            if original_id not in indices_aumentados:
                indices_aumentados[original_id] = []
            indices_aumentados[original_id].append(idx)
        else:
            indices_originales.append(idx)
    
    print(f"Datos originales: {len(indices_originales)}, Usuarios con datos aumentados: {len(indices_aumentados)}")

    etiquetas_originales = [etiquetas[idx] for idx in indices_originales]
    multi_etiquetas_originales = [multi_etiquetas[idx] for idx in indices_originales]

    estratificacion = np.array([f"{a}_{b}" for a, b in zip(etiquetas_originales, multi_etiquetas_originales)])

    # Dividir SOLO los datos originales en train y dev (no test)
    indices_orig_train, indices_orig_dev, _, _, _, _ = train_test_split(
        indices_originales, etiquetas_originales, multi_etiquetas_originales, 
        test_size=0.1, random_state=SEMILLA, 
        stratify=estratificacion
    )

    ids_usuarios_train = {id_usuarios[idx] for idx in indices_orig_train}
    ids_usuarios_dev = {id_usuarios[idx] for idx in indices_orig_dev}

    indices_train_aug = []
    for orig_id, aug_indices in indices_aumentados.items():
        if orig_id in ids_usuarios_train:
            indices_train_aug.extend(aug_indices)

    indices_train = indices_orig_train + indices_train_aug
    indices_dev = indices_orig_dev

    y_train = [etiquetas[idx] for idx in indices_train]
    y_dev = [etiquetas[idx] for idx in indices_dev]

    multi_train = [multi_etiquetas[idx] for idx in indices_train]
    multi_dev = [multi_etiquetas[idx] for idx in indices_dev]

    print(f"División final - Train: {len(indices_train)} ({len(indices_orig_train)} orig + {len(indices_train_aug)} aug), Dev: {len(indices_dev)}")

    print("Generando embeddings para conjunto de entrenamiento...")
    X_train = np.array([
        ModelHandler.obtener_embeddings(" ".join([
            f"{msg.get('date', '')} {msg['message']}" for msg in textos[idx]
        ]), modelo, tokenizador)
        for idx in indices_train
    ])

    print("Generando embeddings para conjunto de validación...")
    X_dev = np.array([
        ModelHandler.obtener_embeddings(" ".join([
            f"{msg.get('date', '')} {msg['message']}" for msg in textos[idx]
        ]), modelo, tokenizador)
        for idx in indices_dev
    ])

    print("\n=== Distribución de los Datos ===")
    y_train = np.array(y_train)
    y_dev = np.array(y_dev)
    etiquetas = np.array(etiquetas)
    porcentaje_tarea_binaria = {
        'Dataset': ['Train', 'Dev', 'Total'],
        'No riesgo (0)': [np.sum(y_train == 0), np.sum(y_dev == 0), np.sum(etiquetas == 0)],
        'riesgo (1)': [np.sum(y_train == 1), np.sum(y_dev == 1), np.sum(etiquetas == 1)],
        'Total': [len(y_train), len(y_dev), len(etiquetas)]
    }
    print("\nDistribución binaria en Clasificación:")
    print(pd.DataFrame(porcentaje_tarea_binaria).to_string(index=False))
    
    multi_labels = sorted(np.unique(multi_etiquetas))
    multi_recuento = {'Dataset': ['Train', 'Dev', 'Total']}
    
    for label in multi_labels:
        multi_recuento[f'Clase {label}'] = [
            np.sum(np.array(multi_train) == label), 
            np.sum(np.array(multi_dev) == label),
            np.sum(np.array(multi_etiquetas) == label)
        ]
    
    multi_recuento['Total'] = [len(multi_train), len(multi_dev), len(multi_etiquetas)]
    
    print("\nDistribución multiclase en Clasificación:")
    print(pd.DataFrame(multi_recuento).to_string(index=False))
    print(f"Train conjunto: {X_train.shape[0]} muestras, Dev conjunto: {X_dev.shape[0]} muestras")
    
    print("Entrenando modelos combinados...")
    clasificador_combinado = GamblingPredictor.entrenar_modelos_combinados(
        X_train, y_train, multi_train, X_dev, y_dev, multi_dev
    )
    
    print("\n=== Evaluación en conjunto de validación ===")
    
    with torch.no_grad():
        # Umbral de 0.65 para las predicciones binarias (para ver rendimiento ficticio preliminar estara sobrevalorado)
        predicciones_binarias_raw = clasificador_combinado.predecir(X_dev)
        predicciones_binarias = np.array([1 if p >= 0.65 else 0 for p in predicciones_binarias_raw])
        binarias_f1_macro = f1_score(y_dev, predicciones_binarias, average='macro')
        
        multi_probs = clasificador_combinado.predecir_proba(X_dev)
        multi_preds = np.argmax(multi_probs, axis=1)
        
        multi_dev_idx = np.array([label_a_idx[label] for label in multi_dev])
        multi_f1_macro = f1_score(multi_dev_idx, multi_preds, average='macro')
    
    print(f"F1 Macro Score (Binario): {binarias_f1_macro:.4f}")
    print(f"F1 Macro Score (Multiclase): {multi_f1_macro:.4f}")
    
    print("\nInforme detallado de clasificación binaria:")
    print(classification_report(y_dev, predicciones_binarias))
    
    print("\nInforme detallado de clasificación multiclase:")
    print(classification_report(multi_dev_idx, multi_preds))
    
    os.makedirs(ruta_guardado, exist_ok=True)
    
    modelo_regresion = clasificador_combinado.modelo_regresion
    modelo_multiclase = clasificador_combinado.modelo_multiclase
    
    torch.save(modelo_regresion.state_dict(), f"{ruta_guardado}/modelo_regresion.pt")
    torch.save(modelo_multiclase.state_dict(), f"{ruta_guardado}/modelo_multiclase.pt")
    
    metadatos = {
        "dimension_embeddings": X_train.shape[1],
        "dimension_oculta": Config.DIMENSION_OCULTA,
        "f1_macro_binarias": float(binarias_f1_macro),
        "f1_macro_multiclass": float(multi_f1_macro),  #SESGO CLARAMENTE OPTIMISTA POR SER EL DEV, SIRVE PARA DARSE LA IDEA RESTAR -0,15
        "fecha_entrenamiento": time.strftime("%Y-%m-%d %H:%M:%S"),
        "umbral_prediccion": 0.65
    }
    
    with open(f"{ruta_guardado}/metadatos_modelo.json", "w") as f:
        json.dump(metadatos, f, indent=2)
    
    print(f"Modelos guardados en {ruta_guardado}:")
    
    np.save(f"{ruta_guardado}/embeddings_train.npy", X_train)
    np.save(f"{ruta_guardado}/etiquetas_train.npy", y_train)
    np.save(f"{ruta_guardado}/etiquetas_multi_train.npy", multi_train)
    
    print("Embeddings y etiquetas de entrenamiento guardados.")
    
    predictor = GamblingPredictor(modelo, tokenizador, clasificador_combinado)
    
    return predictor

def cargar_modelo_guardado(ruta_guardado="./modelos_entrenados_finales"):

    ruta_modelo_regresion = f"{ruta_guardado}/modelo_regresion.pt"
    ruta_modelo_multiclase = f"{ruta_guardado}/modelo_multiclase.pt"
    ruta_metadatos = f"{ruta_guardado}/metadatos_modelo.json"
    
    if not (os.path.exists(ruta_modelo_regresion) and os.path.exists(ruta_modelo_multiclase)):
        raise FileNotFoundError(f"No se encontraron los modelos en {ruta_guardado}")
    
    with open(ruta_metadatos, "r") as f:
        metadatos = json.load(f)
    
    dimension_entrada = metadatos.get("dimension_embeddings", 768)  
    dimension_oculta = metadatos.get("dimension_oculta", Config.DIMENSION_OCULTA)
    
    modelo, tokenizador = ModelHandler.cargar_modelo()
    
    modelo_regresion = LSTMRegressor(dimension_entrada, dimension_oculta)
    modelo_multiclase = LSTMMultiClassifier(dimension_entrada, dimension_oculta)
    
    modelo_regresion.load_state_dict(torch.load(ruta_modelo_regresion))
    modelo_multiclase.load_state_dict(torch.load(ruta_modelo_multiclase))
    
    modelo_regresion.eval()
    modelo_multiclase.eval()
    
    clasificador_combinado = LSTMWrapper(modelo_regresion, modelo_multiclase)
    
    predictor = GamblingPredictor(modelo, tokenizador, clasificador_combinado)
    print(f"Modelos cargados exitosamente desde {ruta_guardado}")
    
    return predictor
























###########################
###########################
###########################
###########################
###########################
###########################
###########################    
###########################
###########################
###########################
###########################
###########################
###########################
###########################    
###########################
###########################
###########################
###########################
###########################
###########################
###########################    
###########################
###########################
###########################
###########################
###########################
###########################
###########################    
###########################
###########################
###########################
###########################
###########################
###########################
###########################    
###########################
###########################
###########################
###########################
###########################
###########################
###########################    

"""  
    confirmacion_completa = input("\n¿Desea continuar con el entrenamiento completo y evaluación utilizando todos los datos? (escriba 'si' para continuar): ")
    
    if confirmacion_completa.lower() == 'si':
        print("\n=== ENTRENANDO MODELO FINAL CON TODOS LOS DATOS ===")
    else:
        print("Proceso finalizado. No se realizará el entrenamiento completo.")
        sys.exit(0)
    textos = []
    for mensajes_usuario in datos_usuarios:
        texto_usuario = []
        for mensaje in mensajes_usuario:
            if isinstance(mensaje, dict) and 'message' in mensaje:
                texto_usuario.append(mensaje['message'])
            elif isinstance(mensaje, str):
                texto_usuario.append(mensaje)
        textos.append(" ".join(str(texto_usuario)))
    
    print("Generando embeddings...")
    X = np.array([ModelHandler.obtener_embeddings(texto, modelo, tokenizador) for texto in textos])
    y = np.array(etiquetas)
    
    print("Entrenando modelos combinados...")
    clasificador_combinado = GamblingPredictor.cargar_modelos_combinados(
        X, y, multi_etiquetas
    )
    
    predictor = GamblingPredictor(modelo, tokenizador, clasificador_combinado)
    
    
    textos_ejemplo = textos_control + textos_ludopatia + textos_ludopatia_compleja
    resultados_ejemplo = predictor.predecir(textos_ejemplo)
    
    print("\n=== resultadoS: TEXTOS DE CONTROL (SIN LUDOPATÍA) ===")
    for i, resultado in enumerate(resultados_ejemplo[:20]):
        print(f"\nTexto {i+1}: {resultado['texto']}")
        print(f"Riesgo de ludopatía: {resultado['valor_prediccion']:.3f}")
        print(f"Probabilidades por clase: {[round(p, 3) for p in resultado['probabilidades_clase']]}")
        clase_predicha = np.argmax(resultado['probabilidades_clase'])
        print(f"Clase predicha: {clase_predicha}")
        print("-" * 80)
    
    print("\n=== resultadoS: TEXTOS DE LUDOPATÍA ===")
    for i, resultado in enumerate(resultados_ejemplo[20:40]):
        print(f"\nTexto {i+1}: {resultado['texto']}")
        print(f"Riesgo de ludopatía: {resultado['valor_prediccion']:.3f}")
        print(f"Probabilidades por clase: {[round(p, 3) for p in resultado['probabilidades_clase']]}")
        clase_predicha = np.argmax(resultado['probabilidades_clase'])
        print(f"Clase predicha: {clase_predicha}")
        print("-" * 80)
    
    print("\n=== resultadoS: TEXTOS DE LUDOPATÍA COMPLEJA ===")
    for i, resultado in enumerate(resultados_ejemplo[40:]):
        print(f"\nTexto {i+1}: {resultado['texto']}")
        print(f"Riesgo de ludopatía: {resultado['valor_prediccion']:.3f}")
        print(f"Probabilidades por clase: {[round(p, 3) for p in resultado['probabilidades_clase']]}")
        clase_predicha = np.argmax(resultado['probabilidades_clase'])
        print(f"Clase predicha: {clase_predicha}")
        print("-" * 80)
    
    print("\nContinuando con el análisis de datos de prueba...")
        # Preparamos el servidor de prueba
    servidor_fake = ServidorFake(
        textos_test=test_datos_usuarios, 
        y_test=test_etiquetas,
        id=test_id_usuarios
    )
    
    
    
    
    os.makedirs("resultadosPredictor", exist_ok=True)

    usuarios_riesgo = {}  # Conteo de mensajes de alto riesgo
    usuarios_clases = {}  # Distribución de clases multiclase
    usuarios_prob_acum = {}
    ronda_actual = 0
    while servidor_fake.hay_siguiente_ronda():
    #for i in range(1):
        ronda_actual += 1
        print(f"\n--- PROCESANDO RONDA {ronda_actual} ---")
        
        mensajes_ronda, num_ronda = servidor_fake.siguiente_ronda()
        predicciones_ronda = []
        
        # Procesamos cada mensaje de la ronda
        for usuario_id, mensaje_info in mensajes_ronda:
            texto = mensaje_info.get("message", "")
            
            # Obtenemos predicción para este mensaje
            resultado = predictor.predecir([texto])[0]
            prob_binaria = resultado['valor_prediccion']
            prob_clases = resultado['probabilidades_clase']
            clase_predicha = np.argmax(prob_clases)
            # Inicializamos datos de usuario si es la primera vez
            if usuario_id not in usuarios_riesgo:
                usuarios_riesgo[usuario_id] = 0
                usuarios_clases[usuario_id] = [0, 0, 0, 0]
                usuarios_prob_acum[usuario_id] = [0.0, 0.0, 0.0, 0.0]
            
            # Actualizamos contadores - sumamos directamente la probabilidad
            usuarios_riesgo[usuario_id] += prob_binaria
            
            temp_array = np.array(usuarios_prob_acum[usuario_id])

            temp_array += np.array(prob_clases)

            usuarios_prob_acum[usuario_id] = temp_array

            usuarios_clases[usuario_id][clase_predicha] += 1
            
            # Guardamos predicción
            predicciones_ronda.append({
                'usuario_id': usuario_id,
                'texto': texto[:50] + "..." if len(texto) > 50 else texto,  # Versión corta del texto
                'riesgo_ludopatia': prob_binaria,
                'clase_0': prob_clases[0],
                'clase_1': prob_clases[1],
                'clase_2': prob_clases[2],
                'clase_3': prob_clases[3],
                'clase_predicha': clase_predicha
            })
        print(usuarios_prob_acum)
        print(f"Analizados {len(predicciones_ronda)} mensajes en ronda {num_ronda}")
        
        # Preparamos decisiones finales para esta ronda
        decisiones_usuarios = []
        resultado_eval = []
        dict_riesgo = {}
        dict_multi = {}
        for usuario_id, mensajes_riesgo in usuarios_riesgo.items():
            # Determinamos etiqueta binaria (riesgo/no riesgo)
            es_riesgo = mensajes_riesgo >= 3 + (ronda_actual // 10)
            # Determinamos clase multiclase más frecuente
            dist_clases = usuarios_prob_acum[usuario_id]  #cambiado de de entero a probabilidad acumulada
            clase_frecuente = np.argmax(dist_clases)
            
            # Guardamos decisión
            decisiones_usuarios.append({
                'usuario_id': usuario_id,
                'mensajes_alto_riesgo': mensajes_riesgo,
                'en_riesgo': 1 if es_riesgo else 0,
                'clase_predicha': clase_frecuente,
                'conteo_clase_0': dist_clases[0],
                'conteo_clase_1': dist_clases[1],
                'conteo_clase_2': dist_clases[2],
                'conteo_clase_3': dist_clases[3],
            })
            []
            dict_riesgo.update({usuario_id: 1 if es_riesgo else 0})
            dict_multi.update({usuario_id: clase_frecuente})
            
            print(f"Usuario {usuario_id}: {'EN RIESGO' if es_riesgo else '✓ Sin riesgo'} - "
                    f"Clase más probable: {clase_frecuente} - "
                    f"Mensajes de alto riesgo: {mensajes_riesgo}")
        # Guardamos resultados en CSV//json ahora 
        resultado_eval.append({
                "predictions": dict_riesgo,
                "type": dict_multi
            })
                    
        pd.DataFrame(decisiones_usuarios).to_json(
            f"resultadosPredictor/ronda_{num_ronda}_decisiones.json", 
            index=False,
            orient='records'
        )
        
        pd.DataFrame(resultado_eval).to_json(
            f"resultadosPredictor/ronda_{num_ronda}_formato.json", 
            index=False)
        pd.DataFrame(predicciones_ronda).to_json(
            f"resultadosPredictor/ronda_{num_ronda}_predicciones.json", 
            index=False,
            orient='records'
        )
        Evaluacion = binariasClassification(gold=dict_gold_test, json_data=resultado_eval, round=ronda_actual)
        Evaluacion.eval_performance()
    # Fuera del while
    print("\n=== RESUMEN FINAL ===")
    resumen_final = []
    dict_multi_final = {}
    dict_riesgo_final = {}
    for usuario_id, mensajes_riesgo in usuarios_riesgo.items():
        es_riesgo = mensajes_riesgo >= 3 + (ronda_actual // 10)
        clase_frecuente = np.argmax(usuarios_prob_acum[usuario_id]) #lo mismo que antes
        
        dict_riesgo_final.update({usuario_id: 1 if es_riesgo else 0})
        dict_multi_final.update({usuario_id: clase_frecuente})

        resumen_final.append({
            'usuario_id': usuario_id,
            'decision_final': 1 if es_riesgo else 0,
            'clase_final': clase_frecuente,
            'total_mensajes_riesgo': mensajes_riesgo
        })
        
        print(f"Usuario {usuario_id}: {' EN RIESGO' if es_riesgo else '✓ Sin riesgo'} - "
                f"Clase final: {clase_frecuente}")
    resultadofinal = {
        "predictions": dict_riesgo_final,
        "type": dict_multi_final
    }
    Evaluacion_multi = multiclassClassification(gold=dict_gold_test_multi, json_data=resultadofinal, labid = label_a_idx)
    Evaluacion_multi.eval_performance()
    pd.DataFrame(resumen_final).to_csv("resultadosPredictor/resumen_final.csv", index=False)
    print("\n Análisis completo - resultados guardados en 'resultadosPredictor/'")
"""
if __name__ == "__main__":
    main()
    #entrenar_y_guardar_modelo()

    """
    y_pred = clasificador_ludopatia.predecir(X_test)
    print("\nresultados de Evaluación en el Conjunto de Prueba:")
    print(classification_report(y_test, y_pred))
    
    
    print("\n===== PREDICCIONES EN CONJUNTO TEST =====")
    resultados_test = predictor.predecir(textos_test)
    for i, resultado in enumerate(resultados_test[:5]):
        print(f"\nEjemplo {i+1}:")
        print(f"Texto: {resultado['texto']}")
        print(f"Predicción: {resultado['predicción']}")
        print(f"Probabilidad de alto riesgo: {resultado['probabilidad_alto']}")
        print(f"Probabilidad de bajo riesgo: {resultado['probabilidad_bajo']}")
        print(f"Etiqueta verdadera: {y_test[i]}")
    
    print("\n===== ENTRENANDO MODELO DE ANÁLISIS DE SENTIMIENTOS =====")
    SentimentAnalyzer.entrenar_modelo_sentimientos()
    
    print("\n===== ANALIZANDO SENTIMIENTO PARA 30 USUARIOS =====")
    resultados_sentimientos = SentimentAnalyzer.analizar_usuarios(
        id_usuarios[:3], 
        datos_usuarios[:3]
    )
    

    verdad_simulada = {id_usuario: etiqueta for id_usuario, etiqueta in zip(id_usuarios[:5], etiquetas[:5])}
    
    
    return {
        "clasificador_ludopatia": clasificador_ludopatia,
        "resultados_sentimientos": resultados_sentimientos,
    }"""
"""
if __name__ == "__main__":
    main()
    #ComparadorModelos.main_comparacion()"""