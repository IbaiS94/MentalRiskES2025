import sys
import os
import logging
import google.generativeai as genai
import argparse
import json
import shutil
import csv

logging.basicConfig(level=logging.INFO)

def get_api():
    api = ""
    return api

def configurar_genai(api_key):
    genai.configure(api_key=api_key)
    modelo = genai.oGenerativeModel('gemini-2.0-flash-thinking-exp')
    return modelo

def modelos_disponibles():
    """Lista los modelos disponibles"""
    try:
        modelos = genai.list_models()
        print("Modelos disponibles:")
        for modelo in modelos:
            print(f"- {modelo.name}: {modelo.display_name}")
        return modelos
    except Exception as e:
        logging.error(f"Error: {e}")
        return []
    
def augment_data(model, inputd, clasificacion):
    
    # Instrucciones
    prompt = f"""
    Act as a data augmentation expert. Create a different variations of the following text while:
    
    1. Preserving the original meaning and key information
    2. Using different sentence structures and synonyms
    3. You can change the fomality of the text and the tone while you maintain the same meaning
    4. Keeping the same emotional content and sentiment
    5. Ensuring the text remains natural and fluent
    6. Messages are on spanish, take this into account
    7. You can use emojis to make the text more friendly and natural
    8. Avoid using the same words or phrases as the original text
    9. Avoid using the same sentence structure as the original text
    10. Avoid using the same punctuation as the original text
    11. You can make ortograficial mistakes as this would be written by a human
    12. You can use slang to make the text more friendly and natural
    13. You can make grammatical mistakes as this would be written by a human
    14. Remember the text is written by a human, so it can have mistakes
    15. Do not always commit the same mistakes, try to make it different each time
    16. Your main goal is doing data augmentation REMEMBER THIS
        
    This text is classified as {clasificacion} (where 1 indicates HIGH risk content and 0 indicates low risk).
    For mental health related text, ensure that clinical significance and risk indicators are preserved.
    
    Format your response as a JSON array the same structure as your input.
    
    Original text that you shall modify:
    {inputd}
    """
    
    try:
        respuesta = model.generate_content(prompt)
        try:
            respuesta_text = respuesta.text
            inicio = respuesta_text.find('[')
            final = respuesta_text.rfind(']') + 1
            if inicio >= 0 and final > inicio:
                json_str = respuesta_text[inicio:final]
                json_data = json.loads(json_str)
            else:
                json_data = json.loads(respuesta_text)
            return json_data
        except json.JSONDecodeError:
            logging.warning("Error de formato en respuesta -> JSON no encontrado, devolviendo texto RAW")
            return [respuesta.text]
    except Exception as e:
        logging.error(f"Error: {e}")
        return []

def cargar_clasificacion(gold_file):
    """Carga las clasificaciones desde el archivo CSV"""
    clasificaciones = {}
    with open(gold_file, "r", encoding="utf-8") as f:
        next(f)
        for fila in csv.reader(f):
            if len(fila) >= 2:
                usr = fila[0]  
                etiq = fila[1]  
                clasificaciones[usr] = etiq
    print(clasificaciones)
    return clasificaciones

def main():
    parser = argparse.ArgumentParser(description="Aumento de datos")
    parser.add_argument("--input_dir", "-i", help="Directorio con los archivos JSON", required=True)
    parser.add_argument("--gold", "-g", help="Gold con las etiquetas", required=True)
    parser.add_argument("--output_dir_0", "-o0", help="Directorio salida para clase 0", default="clase_0")
    parser.add_argument("--output_dir_1", "-o1", help="Directorio salida para clase 1", default="clase_1")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir_0, exist_ok=True)
    os.makedirs(args.output_dir_1, exist_ok=True)
    
    clasi = cargar_clasificacion(args.gold)
    logging.info(f"Cargadas {len(clasi)} etiquetas de {args.gold}")
    
    api = get_api()
    modelo = configurar_genai(api)
    
    procesados = 0
    for archivo in os.listdir(args.input_dir):
        if archivo.endswith('.json'):
            ruta = os.path.join(args.input_dir, archivo)
            
            base_archivo = os.path.splitext(archivo)[0]
            clasificacion = clasi.get(base_archivo, "error")
            
            if clasificacion not in ["0", "1"]:
                logging.warning(f"No hay etiqueta para {archivo}, saltando")
                continue
            
            dir_salida = args.output_dir_1 if clasificacion == "1" else args.output_dir_0
            
            ruta_salida = os.path.join(dir_salida, archivo)
            if os.path.exists(ruta_salida):
                logging.info(f"El archivo {archivo} ya existe en {dir_salida}, saltando")
                continue
            
            with open(ruta, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    input_data = json.dumps(data)
                except json.JSONDecodeError:
                    logging.error(f"Error")
                    continue
            
            aumentado = augment_data(modelo, input_data, clasificacion)
            
            dir_salida = args.output_dir_1 if clasificacion == "1" else args.output_dir_0
            
            salida = os.path.join(dir_salida, archivo)
            with open(salida, "w", encoding="utf-8") as f:
                json.dump(aumentado, 
                          f, indent=2, ensure_ascii=False)
            
            procesados += 1
            logging.info(f"Procesado {archivo} (Clase {clasificacion}) -> {dir_salida}")
    
    logging.info(f"Procesados {procesados} archivos en total")
    print(f"Correctamente procesados {procesados} ficheros:")
    print(f"- Clase 0 guardados en: {os.path.abspath(args.output_dir_0)}")
    print(f"- Clase 1 guardados en: {os.path.abspath(args.output_dir_1)}")


def limpiar_archivos_vacios():
    """Elimina archivos JSON que contienen solo un array vacío debido al rate limit"""
    
    parser = argparse.ArgumentParser(description="Aumento de datos")
    parser.add_argument("--input_dir", "-i", help="Directorio con los archivos JSON", required=True)
    parser.add_argument("--gold", "-g", help="Gold con las etiquetas", required=True)
    parser.add_argument("--output_dir_0", "-o0", help="Directorio salida para clase 0", default="clase_0")
    parser.add_argument("--output_dir_1", "-o1", help="Directorio salida para clase 1", default="clase_1")
    
    args = parser.parse_args()
    
    total_eliminados = 0
    
    for directorio in args.output_dir_0, args.output_dir_1:
        if not os.path.exists(directorio):
            continue
            
        eliminados = 0
        for archivo in os.listdir(directorio):
            if archivo.endswith('.json'):
                ruta = os.path.join(directorio, archivo)
                
                try:
                    with open(ruta, "r", encoding="utf-8") as f:
                        contenido = f.read().strip()
                    
                    if contenido == "[]":
                        os.remove(ruta)
                        eliminados += 1
                        logging.info(f"Eliminado archivo vacío: {ruta}")
                except Exception as e:
                    logging.error(f"Error al verificar archivo vacío {ruta}: {e}")
        
        logging.info(f"Eliminados {eliminados} archivos vacíos en {directorio}")
        total_eliminados += eliminados
    
    return total_eliminados


def modificado_auxiliar():
    """
    Crea un archivo gold modificado añadiendo '+' al inicio de cada nombre de usuario
    junto con su etiqueta de clase correspondiente. También renombra los archivos
    para que coincidan con los nombres de usuario modificados.
    """
    parser = argparse.ArgumentParser(description="Aumento de datos")
    parser.add_argument("--gold", "-g", help="Gold con las etiquetas", required=True)
    parser.add_argument("--output_dir_0", "-o0", help="Directorio salida para clase 0", default="clase_0")
    parser.add_argument("--output_dir_1", "-o1", help="Directorio salida para clase 1", default="clase_1")
    
    args = parser.parse_args()
    
    dir_clase_0 = args.output_dir_0
    dir_clase_1 = args.output_dir_1
    output_file = "gold_modificado.txt"
    
    resultados = []
    archivos_cambiados = 0
    
    if os.path.exists(dir_clase_0):
        for archivo in os.listdir(dir_clase_0):
         if archivo.endswith('.json'):
            usuario = os.path.splitext(archivo)[0]
            resultados.append(f"+{usuario},0")
            
            ruta_vieja = os.path.join(dir_clase_0, archivo)
            ruta_nueva = os.path.join(dir_clase_0, f"+{usuario}.json")
            os.rename(ruta_vieja, ruta_nueva)
            archivos_cambiados += 1
    
    if os.path.exists(dir_clase_1):
        for archivo in os.listdir(dir_clase_1):
         if archivo.endswith('.json'):
            usuario = os.path.splitext(archivo)[0]
            resultados.append(f"+{usuario},1")
            
            ruta_vieja = os.path.join(dir_clase_1, archivo)
            ruta_nueva = os.path.join(dir_clase_1, f"+{usuario}.json")
            os.rename(ruta_vieja, ruta_nueva)
            archivos_cambiados += 1
    
    with open(output_file, "w", encoding="utf-8") as f:
        for linea in resultados:
            f.write(linea + "\n")
            
    return len(resultados)
if __name__ == "__main__":
    """    limpiar_archivos_vacios() """
    api = get_api()
    genai.configure(api_key=api)
    modelos_disponibles()
    #main()
    #modificado_auxiliar()
