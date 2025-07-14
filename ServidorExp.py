from sklearn.metrics import accuracy_score, f1_score

class ServidorExp:
    def __init__(self, textos_test, id):
            self.textos_usuarios = textos_test
            if isinstance(textos_test, list):
                self.mensajes_por_usuario = {usr_id: textos for usr_id, textos in zip(id, textos_test)}
            else:
                self.mensajes_por_usuario = textos_test
            self.rondas_procesadas = 0
            self.mensaje_actual_por_usuario = {usr_id: 0 for usr_id in id}
            self.id = id

    def siguiente_ronda(self):
        self.rondas_procesadas += 1
        mensajes_ronda = []
        for usuario_id in self.id:
            if self.mensaje_actual_por_usuario[usuario_id] < len(self.mensajes_por_usuario[usuario_id]):
                mensaje = self.mensajes_por_usuario[usuario_id][self.mensaje_actual_por_usuario[usuario_id]]
                mensajes_ronda.append((usuario_id, mensaje))
                self.mensaje_actual_por_usuario[usuario_id] += 1
        return mensajes_ronda, int(self.rondas_procesadas)
    
    def hay_siguiente_ronda(self):
        """
        Verifica si hay al menos un mensaje más para procesar en la siguiente ronda.
        
        Returns:
            bool: True si al menos un usuario tiene mensajes pendientes, False en caso contrario.
        """
        for usuario_id in self.id:
            if self.mensaje_actual_por_usuario[usuario_id] < len(self.mensajes_por_usuario[usuario_id]):
                return True
        return False

    """    def _procesar_ronda(self, mensajes_ronda):
        for usuario_id, mensaje in mensajes_ronda:
            # Acumular contexto histórico
            contexto = ". ".join(self.mensajes_por_usuario[usuario_id][:self.mensaje_actual_por_usuario[usuario_id]])
            
            # Realizar predicción
            resultado = self.predictor.predecir([contexto])[0]
            decision = 1 if resultado['predicción'] == 'Alto riesgo' else 0
            
            # Registrar decisión en el histórico
            self.historico_predicciones[usuario_id].append(
                (self.mensaje_actual_por_usuario[usuario_id], decision))
    """
    """    def obtener_mensajes_por_ronda(self):
        
        Devuelve un mensaje por cada usuario en la ronda actual y avanza a la siguiente ronda.
        
        Returns:
            list: Lista de tuplas (usuario_id, mensaje) con un mensaje de cada usuario.
        
        mensajes_ronda = self._siguiente_ronda()
        self._procesar_ronda(mensajes_ronda)
        print(mensajes_ronda)
        return mensajes_ronda"""