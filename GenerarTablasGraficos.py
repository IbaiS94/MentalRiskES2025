import csv
from collections import Counter
import json
import main
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from google import generativeai as genai
from collections import defaultdict
from datetime import datetime


def load_etiquetas(path):
    etiquetas = []
    user_ids = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for fila in reader:
            if len(fila) >= 2:
                id_usuario, etiqueta = fila[0], fila[1]
                if etiqueta in ['0', '1']:
                    user_ids.append(id_usuario)
                    etiquetas.append(int(etiqueta))
    return user_ids, etiquetas
def ruta(path, path2):
    train_user_ids, train_etiquetas = load_etiquetas(path)
    trial_user_ids, trial_etiquetas = load_etiquetas(path2)

    train_recuentos = Counter(train_etiquetas)
    trial_recuentos = Counter(trial_etiquetas)

    etiqueta_nombres = {0: 'Bajo riesgo', 1: 'Alto riesgo'}

    tabla_filas = []
    for etiqueta in [0, 1]:
        train = train_recuentos.get(etiqueta, 0)
        trial = trial_recuentos.get(etiqueta, 0)
        total = train + trial
        tabla_filas.append([etiqueta_nombres[etiqueta], train, trial, total])

    total_train = sum(train_recuentos.values())
    total_trial = sum(trial_recuentos.values())
    total_total = total_train + total_trial
    tabla_filas.append(['Total', total_train, total_trial, total_total])

    cabeceras = ['riesgo', 'Train', 'Trial', 'Total']
    print(f"| {' | '.join(cabeceras)} |")
    print(f"| {' | '.join(['---']*len(cabeceras))} |")
    for fila in tabla_filas:
        print(f"| {' | '.join(map(str, fila))} |")
        
    print("\nTabla 1: Distribución de etiquetas en los conjuntos de entrenamiento y prueba.")
def generar_distribucion_de_categorias(task2_ruta):
    task1_ruta = task2_ruta.replace("task2", "task1")
    usuario = {}
    
    with open(task1_ruta, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for fila in reader:
            if len(fila) >= 2:
                id_usuario, etiqueta = fila[0], fila[1]
                if etiqueta in ['0', '1']:
                    usuario[id_usuario] = etiqueta
    
    categorias = ["betting", "onlinegaming", "trading", "lootboxes"]
    riesgo_etiquetas = ['0', '1']
    distribution = {mh: {cat: 0 for cat in categorias} for mh in riesgo_etiquetas}
    
    with open(task2_ruta, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for fila in reader:
            if len(fila) >= 2: 
                id_usuario, category = fila[0], fila[1]
                if id_usuario in usuario and category in categorias:
                    mh_status = usuario[id_usuario]
                    distribution[mh_status][category] += 1
    
    etiqueta_nombres = {'0': 'Bajo riesgo', '1': 'Alto riesgo'}
    cabeceras = ["riesgo"] + categorias + ["Total"]
    print(f"| {' | '.join(cabeceras)} |")
    print(f"| {' | '.join(['---'] * len(cabeceras))} |")
    
    total_total = {cat: 0 for cat in categorias}
    for mh in riesgo_etiquetas:
        filas_tot = sum(distribution[mh].values())
        datos_fila = [etiqueta_nombres[mh]]
        for cat in categorias:
            recuento = distribution[mh][cat]
            datos_fila.append(str(recuento))
            total_total[cat] += recuento
        datos_fila.append(str(filas_tot))
        print(f"| {' | '.join(datos_fila)} |")
    
    filas_totales = ["Total"]
    for cat in categorias:
        filas_totales.append(str(total_total[cat]))
    filas_totales.append(str(sum(total_total.values())))

    print(f"| {' | '.join(filas_totales)} |")
    print("\nTabla 2: Distribución de etiquetas en la tarea 2.")


def generar_clases_por_riesgo(task2_ruta):
    task1_ruta = task2_ruta.replace("task2", "task1")
    usr_riesgo = {}
    
    with open(task1_ruta, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for fila in reader:
            if len(fila) >= 2:
                id_usuario, etiqueta = fila[0], fila[1]
                if etiqueta in ['0', '1']:
                    usr_riesgo[id_usuario] = '0' if etiqueta == '0' else '1'
    
    usr_categorias = {}
    with open(task2_ruta, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for fila in reader:
            if len(fila) >= 2:
                id_usuario, category = fila[0], fila[1]
                if id_usuario not in usr_categorias:
                    usr_categorias[id_usuario] = set()
                usr_categorias[id_usuario].add(category)
    
    cantidad_clases = {
        '0': {1: 0, 2: 0, 3: 0, 4: 0},  
        '1': {1: 0, 2: 0, 3: 0, 4: 0}   
    }
    
    for id_usuario, categorias in usr_categorias.items():
        if id_usuario in usr_riesgo:
            mh = usr_riesgo[id_usuario]
            num_clases = len(categorias)
            cantidad_clases[mh][num_clases] += 1
    
    cabeceras = ["riesgo"] + [f"{i} class{'es' if i > 1 else ''}" for i in range(1, 5)] + ["Total"]
    print(f"| {' | '.join(cabeceras)} |")
    print(f"| {' | '.join(['---'] * len(cabeceras))} |")
    
    etiqueta_nombres = {'0': 'Bajo riesgo', '1': 'Alto riesgo'}
    total_total = {i: 0 for i in range(1, 5)}
    
    for mh in ['0', '1']:
        fila = [etiqueta_nombres[mh]]
        filas_tot = 0
        for num_clases in range(1, 5):
            recuento = cantidad_clases[mh][num_clases]
            fila.append(str(recuento))
            total_total[num_clases] += recuento
            filas_tot += recuento
        fila.append(str(filas_tot))
        print(f"| {' | '.join(fila)} |")
    
    filas_totales = ["Total"]
    for num_clases in range(1, 5):
        filas_totales.append(str(total_total[num_clases]))
    filas_totales.append(str(sum(total_total.values())))
    print(f"| {' | '.join(filas_totales)} |")
    print("\nTabla 3: Numero de clases por usuario en la tarea 2:")    
    
def generar_tabla_distribucion_plataformas(ruta, carpeta_json):
    id_usuarios, etiquetas = main.DataProcessor.cargar_etiquetas(ruta)
    
    datos_usuarios = main.DataProcessor.cargar_datos_usuarios(carpeta_json, id_usuarios)
    
    plataformas = ["twitch", "telegram"]
    etiquetas_ludopatia = [0, 1] 
    nombres_etiquetas = {0: 'Bajo riesgo', 1: 'Alto riesgo'}
    
    distribucion = {etiqueta: {plataforma: 0 for plataforma in plataformas} 
                   for etiqueta in etiquetas_ludopatia}
    
 
    for i, datos in enumerate(datos_usuarios):
        id_usuario = id_usuarios[i] 
        etiqueta = etiquetas[i]   
        plataforma = ""
        for item in datos:
            if isinstance(item, dict) and "platform" in item:
                plataforma = item["platform"].lower()
                break
        else:
            plataforma = ""
            
        
        if plataforma in plataformas:
            distribucion[etiqueta][plataforma] += 1
    
    print("\nDistribución de usuarios por plataforma y nivel de riesgo:")
    cabeceras = ["riesgo"] + plataformas + ["Total"]
    print(f"| {' | '.join(cabeceras)} |")
    print(f"| {' | '.join(['---'] * len(cabeceras))} |")
    
    total_por_plataforma = {plataforma: 0 for plataforma in plataformas}
    
    for etiqueta in etiquetas_ludopatia:
        filas_tot = sum(distribucion[etiqueta].values())
        datos_fila = [nombres_etiquetas[etiqueta]]
        
        for plataforma in plataformas:
            recuento = distribucion[etiqueta][plataforma]
            datos_fila.append(str(recuento))
            total_por_plataforma[plataforma] += recuento
        
        datos_fila.append(str(filas_tot))
        print(f"| {' | '.join(datos_fila)} |")
    
    filas_totales = ["Total"]
    for plataforma in plataformas:
        filas_totales.append(str(total_por_plataforma[plataforma]))
    filas_totales.append(str(sum(total_por_plataforma.values())))
    print(f"| {' | '.join(filas_totales)} |")
    
    print("\nTabla 4: Distribución de usuarios por plataforma y nivel de riesgo.")
    
def generar_diagramas_bigotes_tokens(ruta_etiquetas, carpeta_json):
    """
    Genera diagramas de bigotes (box plots) para la longitud en tokens
    de mensajes individuales y mensajes totales por usuario, usando
    la tokenización del modelo Gemini text-embedding-004.
    """
    
    def contar_tokens_gemini(texto, api_key=None):
        if not texto:
            return 0
        
        if not isinstance(texto, str):
            texto = str(texto)
        
        palabras = texto.split()
        caracteres = len(texto)
        return max(1, int(caracteres / 4))

    id_usuarios, etiquetas = main.DataProcessor.cargar_etiquetas(ruta_etiquetas)
    datos_usuarios = main.DataProcessor.cargar_datos_usuarios(carpeta_json, id_usuarios)
    
    longitud_mensajes_individuales = []
    longitud_mensajes_por_usuario = []
    
    print("Procesando textos...")
    for datos_usuario in datos_usuarios:
        mensajes_tokens_usuario = []
        
        for mensaje in datos_usuario:
            if isinstance(mensaje, dict) and 'message' in mensaje:
                texto = mensaje['message']
                num_tokens = contar_tokens_gemini(texto)
                
                longitud_mensajes_individuales.append(num_tokens)
                mensajes_tokens_usuario.append(num_tokens)
        
        if mensajes_tokens_usuario:
            longitud_mensajes_por_usuario.append(sum(mensajes_tokens_usuario))
    
    sns.set_style("whitegrid")
    
    flierprops = dict(
        marker='o', 
        markerfacecolor='#1E88E5', 
        markersize=5,              
        alpha=0.7,                 
        markeredgecolor='white',    
        markeredgewidth=0.5          
    )
    
    medianprops = dict(
        color='red', 
        linewidth=1.5
    )
    
    directorio_salida = "./graficos"
    os.makedirs(directorio_salida, exist_ok=True)
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    bp1 = ax1.boxplot(
        longitud_mensajes_individuales, 
        patch_artist=True, 
        flierprops=flierprops, 
        showfliers=True, 
        vert=False,
        medianprops=medianprops
    )
    
    for box in bp1['boxes']:
        box.set(facecolor='#4682B4', alpha=0.7)  
    
    ax1.set_xlabel('Number of tokens', fontsize=12)
    ax1.set_yticks([])  
    
    plt.tight_layout()
    plt.savefig(os.path.join(directorio_salida, 'mensaje_individual_tokens_horizontal.png'), 
                dpi=300, transparent=True)
    plt.close(fig1)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    bp2 = ax2.boxplot(
        longitud_mensajes_por_usuario, 
        patch_artist=True, 
        vert=False,
        flierprops=flierprops,  
        medianprops=medianprops
    )
    
    for box in bp2['boxes']:
        box.set(facecolor='#6AB187', alpha=0.7)

    ax2.set_yticks([]) 
    
    plt.tight_layout()
    plt.savefig(os.path.join(directorio_salida, 'usuario_total_tokens_horizontal.png'), 
                dpi=300, transparent=True)
    plt.close(fig2)
    
    print("\nEstadísticas de longitud de mensajes (en tokens para Gemini):")
    print(f"Mensajes individuales - Media: {np.mean(longitud_mensajes_individuales):.2f}, "
          f"Mediana: {np.median(longitud_mensajes_individuales):.2f}, "
          f"Desviación estándar: {np.std(longitud_mensajes_individuales):.2f}")
    
    print("\nEstadísticas de longitud por usuario (en tokens para Gemini):")
    print(f"Por usuario - Media: {np.mean(longitud_mensajes_por_usuario):.2f}, "
          f"Mediana: {np.median(longitud_mensajes_por_usuario):.2f}, "
          f"Desviación estándar: {np.std(longitud_mensajes_por_usuario):.2f}")
    
def generar_diagramas_bigotes_mensajes(ruta_etiquetas, carpeta_json):
    """
    Genera un diagrama de bigotes horizontal comparando la distribución de mensajes 
    por usuario entre grupos de bajo y alto riesgo
    """
    def contar_mensajes(archivo):
        """cuenta el número de mensajes en un archivo JSON"""
        try:
            if not os.path.exists(archivo):
                return 0
                
            with open(archivo, 'r', encoding='utf-8') as f:
                contenido = f.read().strip()
                
            if not contenido or contenido == "[]":
                return 0
                
            data = json.loads(contenido)
            return len(data) if isinstance(data, list) else 0
            
        except Exception as e:
            print(f"Error procesando {archivo}: {str(e)}")
            return 0

    etiquetas_usuarios = {}
    if os.path.exists(ruta_etiquetas):
        with open(ruta_etiquetas, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for fila in reader:
                if len(fila) >= 2 and fila[1] in ['0', '1']:
                    etiquetas_usuarios[fila[0]] = 'Bajo riesgo' if fila[1] == '0' else 'Alto riesgo'

    archivos = [os.path.join(carpeta_json, f) for f in os.listdir(carpeta_json) 
                if f.endswith('.json')]
    
    datos_bajo_riesgo = []
    datos_alto_riesgo = []
    
    for archivo in archivos:
        usuario = os.path.splitext(os.path.basename(archivo))[0]
        mensajes = contar_mensajes(archivo)
        
        if usuario in etiquetas_usuarios:
            if etiquetas_usuarios[usuario] == 'Bajo riesgo':
                datos_bajo_riesgo.append(mensajes)
            else:
                datos_alto_riesgo.append(mensajes)


    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    fig, ax = plt.subplots(figsize=(12, 8))

    datos_para_plot = []
    etiquetas_para_plot = []
    
    if datos_bajo_riesgo:
        datos_para_plot.append(datos_bajo_riesgo)
        etiquetas_para_plot.append(f'Bajo riesgo (n={len(datos_bajo_riesgo)})')
    
    if datos_alto_riesgo:
        datos_para_plot.append(datos_alto_riesgo)
        etiquetas_para_plot.append(f'Alto riesgo (n={len(datos_alto_riesgo)})')

    bp = ax.boxplot(
        datos_para_plot,
        vert=False,  
        patch_artist=True,
        showfliers=True,
        labels=etiquetas_para_plot
    )

    colors = ['#6AB187', '#CC5850'] 
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('#2c3e50')
        patch.set_linewidth(0.8)

    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(1.5)

    for whisker in bp['whiskers']:
        whisker.set_color('#2c3e50')
        whisker.set_linewidth(0.8)
    
    for cap in bp['caps']:
        cap.set_color('#2c3e50')
        cap.set_linewidth(0.8)

    for flier in bp['fliers']:
        flier.set_marker('o')
        flier.set_markerfacecolor('#1E88E5')
        flier.set_markersize(4)
        flier.set_alpha(0.6)

    ax.set_xlabel('Número de mensajes por usuario', fontsize=14, fontweight='bold') 
    ax.set_ylabel('Grupo de riesgo', fontsize=14, fontweight='bold') 
    ax.tick_params(axis='both', labelsize=12)
    
    ax.grid(True, alpha=0.7)
    
    plt.tight_layout()
    
    directorio_salida = "./graficos"
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)
    
    ruta_salida = os.path.join(directorio_salida, "comparacion_mensajes_usuarios_horizontal.png")
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"\nDiagrama guardado en: {ruta_salida}")
    print("\n=== ESTADÍSTICAS DE MENSAJES POR USUARIO ===")
    
    if datos_bajo_riesgo:
        print(f"\nBajo riesgo ({len(datos_bajo_riesgo)} usuarios):")
        print(f"  Media: {np.mean(datos_bajo_riesgo):.2f}")
        print(f"  Mediana: {np.median(datos_bajo_riesgo):.2f}")
        print(f"  Desviación estándar: {np.std(datos_bajo_riesgo):.2f}")
        print(f"  Mínimo: {np.min(datos_bajo_riesgo)}")
        print(f"  Máximo: {np.max(datos_bajo_riesgo)}")
    
    if datos_alto_riesgo:
        print(f"\nAlto riesgo ({len(datos_alto_riesgo)} usuarios):")
        print(f"  Media: {np.mean(datos_alto_riesgo):.2f}")
        print(f"  Mediana: {np.median(datos_alto_riesgo):.2f}")
        print(f"  Desviación estándar: {np.std(datos_alto_riesgo):.2f}")
        print(f"  Mínimo: {np.min(datos_alto_riesgo)}")
        print(f"  Máximo: {np.max(datos_alto_riesgo)}")

def generar_diagrama_actividad_3h_balanceado(carpeta_json, ruta_etiquetas=None):
    """
    Genera un diagrama de actividad mostrando la distribución de mensajes
    por tramos de 3 horas, corrigiendo el desbalanceo de clases.
    """
    
    def extraer_hora_mensaje(fecha_str):
        """Extrae la hora de un string de fecha"""
        try:
            fecha = datetime.fromisoformat(fecha_str.replace('+01:00', ''))
            return fecha.hour
        except:
            return None
    
    def obtener_tramo_3h(hora):
        """Convierte una hora en un tramo de 3 horas"""
        if hora is None:
            return None
        inicio = hora // 3 * 3
        fin = inicio + 3
        return f"{inicio:02d}:00-{fin:02d}:00"
    
    if not os.path.exists(carpeta_json):
        raise FileNotFoundError(f"La carpeta no existe: {carpeta_json}")
    
    etiquetas_usuarios = {}
    if ruta_etiquetas and os.path.exists(ruta_etiquetas):
        with open(ruta_etiquetas, 'r', encoding='utf-8') as f:
            import csv
            reader = csv.reader(f)
            for fila in reader:
                if len(fila) >= 2 and fila[1] in ['0', '1']:
                    etiquetas_usuarios[fila[0]] = 'Bajo riesgo' if fila[1] == '0' else 'Alto riesgo'
    
    archivos = [f for f in os.listdir(carpeta_json) if f.endswith('.json')]
    print(f"Procesando {len(archivos)} archivos...")
    
    actividad_total = defaultdict(int)
    actividad_por_grupo = defaultdict(lambda: defaultdict(int))
    mensajes_por_usuario_grupo = defaultdict(lambda: defaultdict(int))
    usuarios_por_grupo = defaultdict(int) 
    usuarios_procesados = 0
    mensajes_procesados = 0
    
    for archivo in archivos:
        usuario_id = os.path.splitext(archivo)[0]
        ruta_archivo = os.path.join(carpeta_json, archivo)
        
        try:
            with open(ruta_archivo, 'r', encoding='utf-8') as f:
                contenido = f.read().strip()
                
            if not contenido or contenido == "[]":
                continue
                
            data = json.loads(contenido)
            if not isinstance(data, list):
                continue
            
            usuarios_procesados += 1
            grupo = etiquetas_usuarios.get(usuario_id, 'Sin clasificar')
            usuarios_por_grupo[grupo] += 1
            
            mensajes_usuario_por_tramo = defaultdict(int)
            
            for mensaje in data:
                if isinstance(mensaje, dict) and 'date' in mensaje:
                    hora = extraer_hora_mensaje(mensaje['date'])
                    if hora is not None:
                        tramo = obtener_tramo_3h(hora)
                        if tramo:
                            actividad_total[tramo] += 1
                            actividad_por_grupo[grupo][tramo] += 1
                            mensajes_usuario_por_tramo[tramo] += 1
                            mensajes_procesados += 1
            
            for tramo, recuento in mensajes_usuario_por_tramo.items():
                mensajes_por_usuario_grupo[grupo][tramo] += recuento
                            
        except Exception as e:
            print(f"Error procesando {archivo}: {str(e)}")
            continue
    
    print(f"Usuarios procesados: {usuarios_procesados}")
    print(f"Mensajes procesados: {mensajes_procesados}")
    
    for grupo, recuento in usuarios_por_grupo.items():
        total_mensajes = sum(actividad_por_grupo[grupo].values())
        promedio_mensajes = total_mensajes / recuento if recuento > 0 else 0
        print(f"{grupo}: {recuento} usuarios, {total_mensajes:,} mensajes totales")
        print(f"  Promedio: {promedio_mensajes:.1f} mensajes por usuario")
    
    if not actividad_total:
        print("No se encontraron datos de actividad válidos")
        return
    
    tramos_ordenados = [f"{i:02d}:00-{(i+3):02d}:00" for i in range(0, 24, 3)]
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    directorio_salida = "./graficos"
    os.makedirs(directorio_salida, exist_ok=True)
    
    valores_actividad = [actividad_total.get(tramo, 0) for tramo in tramos_ordenados]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(range(len(tramos_ordenados)), valores_actividad, 
                  color='#4682B4', alpha=0.7, edgecolor='#1e3f66', linewidth=0.8)
    
    ax.set_xlabel('Tramos horarios (3 horas)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Número de mensajes', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(tramos_ordenados)))
    ax.set_xticklabels(tramos_ordenados, rotation=45, ha='right')
    
    for bar, valor in zip(bars, valores_actividad):
        if valor > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(valores_actividad)*0.01,
                   f'{valor:,}', ha='center', va='bottom', fontsize=9)
    
    promedio = np.mean(valores_actividad)
    ax.axhline(y=promedio, color='red', linestyle='--', alpha=0.7, 
               label=f'Promedio: {promedio:.0f} mensajes')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(directorio_salida, 'actividad_por_tramos_3h_total.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    if etiquetas_usuarios and len(actividad_por_grupo) > 1:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        grupos = [g for g in actividad_por_grupo.keys() if g in ['Bajo riesgo', 'Alto riesgo']]
        colors = {'Bajo riesgo': '#6AB187', 'Alto riesgo': '#CC5850'}
        
        x = np.arange(len(tramos_ordenados))
        width = 0.35
        
        for i, grupo in enumerate(grupos):
            valores_grupo = [actividad_por_grupo[grupo].get(tramo, 0) for tramo in tramos_ordenados]
            ost = (i - len(grupos)/2 + 0.5) * width
            ax1.bar(x + ost, valores_grupo, width, 
                   label=f'{grupo} (n={usuarios_por_grupo[grupo]})', 
                   color=colors[grupo], alpha=0.7)
        
        ax1.set_xlabel('Tramos horarios (3 horas)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Número de mensajes (total)', fontsize=12, fontweight='bold')
        ax1.set_title('Comparación Absoluta por Grupos', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(tramos_ordenados, rotation=45, ha='right')
        ax1.legend()
        
        for i, grupo in enumerate(grupos):
            valores_normalizados = []
            for tramo in tramos_ordenados:
                total_tramo = actividad_por_grupo[grupo].get(tramo, 0)
                usuarios_grupo = usuarios_por_grupo[grupo]
                promedio_por_usuario = total_tramo / usuarios_grupo if usuarios_grupo > 0 else 0
                valores_normalizados.append(promedio_por_usuario)
            
            ost = (i - len(grupos)/2 + 0.5) * width
            ax2.bar(x + ost, valores_normalizados, width, 
                   label=f'{grupo} (promedio por usuario)', 
                   color=colors[grupo], alpha=0.7)
        
        ax2.set_xlabel('Tramos horarios (3 horas)', fontsize=18, fontweight='bold')
        ax2.set_ylabel('Mensajes promedio por usuario', fontsize=18, fontweight='bold')
        ax2.set_title('Comparación Normalizada por Usuario', fontsize=18, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(tramos_ordenados, rotation=45, ha='right')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(directorio_salida, 'actividad_por_tramos_3h_comparacion.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, grupo in enumerate(grupos):
            valores_grupo = [actividad_por_grupo[grupo].get(tramo, 0) for tramo in tramos_ordenados]
            total_grupo = sum(valores_grupo)
            
            if total_grupo > 0:
                porcentajes = [(v / total_grupo) * 100 for v in valores_grupo]
                ost = (i - len(grupos)/2 + 0.5) * width
                ax.bar(x + ost, porcentajes, width, 
                      label=f'{grupo} (% del total del grupo)', 
                      color=colors[grupo], alpha=0.7)
        
        ax.set_xlabel('Tramos horarios (3 horas)', fontsize=18, fontweight='bold')
        ax.set_ylabel('Porcentaje de mensajes del grupo (%)', fontsize=18, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tramos_ordenados, rotation=45, ha='right')
        ax.legend(prop={'size': 14})
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        
        plt.tight_layout()
        plt.savefig(os.path.join(directorio_salida, 'actividad_por_tramos_3h_porcentual.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\n ESTADÍSTICAS")
    
    if etiquetas_usuarios:
        print("\n--- ANÁLISIS POR GRUPOS ---")
        for grupo in ['Bajo riesgo', 'Alto riesgo']:
            if grupo in actividad_por_grupo:
                print(f"\n{grupo}:")
                usuarios_grupo = usuarios_por_grupo[grupo]
                
                for tramo in tramos_ordenados:
                    total_tramo = actividad_por_grupo[grupo].get(tramo, 0)
                    promedio_usuario = total_tramo / usuarios_grupo if usuarios_grupo > 0 else 0
                    total_grupo = sum(actividad_por_grupo[grupo].values())
                    porcentaje = (total_tramo / total_grupo) * 100 if total_grupo > 0 else 0
                    
                    print(f"  {tramo}: {total_tramo:,} msgs ({promedio_usuario:.1f} por usuario, {porcentaje:.1f}%)")
                
                promedios_por_tramo = {}
                for tramo in tramos_ordenados:
                    total_tramo = actividad_por_grupo[grupo].get(tramo, 0)
                    promedios_por_tramo[tramo] = total_tramo / usuarios_grupo if usuarios_grupo > 0 else 0
                
                if promedios_por_tramo:
                    tramo_max = max(promedios_por_tramo, key=promedios_por_tramo.get)
                    max_promedio = promedios_por_tramo[tramo_max]
                    print(f"  Pico normalizado: {tramo_max} con {max_promedio:.1f} mensajes por usuario")
    
    if len(actividad_por_grupo) >= 2:
        print(f"\n--- ANÁLISIS DE DIFERENCIAS ---")
        grupos = ['Bajo riesgo', 'Alto riesgo']
        
        for tramo in tramos_ordenados:
            print(f"\n{tramo}:")
            for grupo in grupos:
                if grupo in actividad_por_grupo:
                    total = actividad_por_grupo[grupo].get(tramo, 0)
                    usuarios = usuarios_por_grupo[grupo]
                    promedio = total / usuarios if usuarios > 0 else 0
                    print(f"  {grupo}: {promedio:.2f} mensajes/usuario")
            
            if all(grupo in actividad_por_grupo for grupo in grupos):
                prom_alta = actividad_por_grupo['Alto riesgo'].get(tramo, 0) / usuarios_por_grupo.get('Alto riesgo', 1)
                prom_baja = actividad_por_grupo['Bajo riesgo'].get(tramo, 0) / usuarios_por_grupo.get('Bajo riesgo', 1)
                
                if prom_baja > 0:
                    ratio = prom_alta / prom_baja
                    print(f"  Ratio (Alto/Bajo): {ratio:.2f}x")


"""generar_diagrama_actividad_3h_balanceado(
    "./MentalRisk2025/task1/train/subjects",
    "./MentalRisk2025/task1/train/gold_task1.txt"
)"""


generar_diagramas_bigotes_mensajes("./MentalRisk2025 - original/task1/train/gold_task1.txt", 
                                   "./MentalRisk2025 - original/task1/train/subjects")


"""
    
generar_diagramas_bigotes_tokens("./MentalRisk2025 - original/task1/train/gold_task1.txt", 
                                   "./MentalRisk2025 - original/task1/train/subjects")

ruta("./Datos originales sin augmentation/task1/train/gold_task1.txt", "./Datos originales sin augmentation/task1/trial/gold_task1.txt")
generar_distribución("./MentalRisk2025 - original/task2/train/gold_task2.txt")
generar_clases_por_riesgo("./MentalRisk2025 - original/task2/train/gold_task2.txt")
generar_tabla_distribucion_plataformas("./MentalRisk2025 - original/task1/train/gold_task1.txt","./MentalRisk2025 - original/task1/train/subjects")
generar_diagramas_bigotes_mensajes("./MentalRisk2025 - 0/task1/train/subjects")"""
""" Para test 
generar_diagramas_bigotes_tokens("./Datos originales sin augmentation/task2/test/test/gold_task2.txt", 
                                   "./Datos originales sin augmentation/task2/test/test/subject")
ruta("./Datos originales sin augmentation/task1/test/test/gold_task1.txt", "./Datos originales sin augmentation/task2/test/test/gold_task2.txt")
generar_distribución("./Datos originales sin augmentation/task2/test/test/gold_task2.txt")
generar_clases_por_riesgo("./Datos originales sin augmentation/task2/test/test/gold_task2.txt")
#generar_tabla_distribucion_plataformas("./MentalRisk2025 - original/task1/train/gold_task1.txt","./MentalRisk2025 - original/task1/train/subjects")"""



"""ruta("./MentalRisk2024/Train/Gold/gold_a.txt")
generar_distribución("./MentalRisk2024/Train/Gold/gold_a.txt")
generar_clases_por_riesgo("./MentalRisk2024/Train/Gold/gold_a.txt")
generar_tabla_distribucion_plataformas("./MentalRisk2024/Train/Gold/gold_a.txt","./MentalRisk2024/Train/")
"""