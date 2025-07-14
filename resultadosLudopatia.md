
| Ludopatía | Train | Trial | Total |
| --- | --- | --- | --- |
| Baja ludopatia | 182 | 182 | 364 |
| Alta ludopatía | 175 | 175 | 350 |
| Total | 357 | 357 | 714 |

Tabla 1: Distribución de etiquetas en los conjuntos de entrenamiento y prueba.

Distribution of samples across categories:
| Ludopatía | betting | onlinegaming | trading | lootboxes | Total |
| --- | --- | --- | --- | --- | --- |
| Baja ludopatia | 41 | 51 | 76 | 14 | 182 |
| Alta ludopatía | 46 | 55 | 61 | 13 | 175 |
| Total | 87 | 106 | 137 | 27 | 357 |

Tabla 2: Distribución de etiquetas en la tarea 2.
| Ludopatía | 1 class | 2 classes | 3 classes | 4 classes | Total |
| --- | --- | --- | --- | --- | --- |
| Baja ludopatia | 182 | 0 | 0 | 0 | 182 |
| Alta ludopatía | 175 | 0 | 0 | 0 | 175 |
| Total | 357 | 0 | 0 | 0 | 357 |

Tabla 3: Numero de clases por usuario en la tarea 2:

Distribución de usuarios por plataforma y nivel de ludopatía:
| Ludopatía | twitch | telegram | Total |
| --- | --- | --- | --- |
| Baja ludopatía | 65 | 117 | 182 |
| Alta ludopatía | 68 | 107 | 175 |
| Total | 133 | 224 | 357 |

Tabla 4: Distribución de usuarios por plataforma y nivel de ludopatía.



25/03/2025 commit 589cffdd258da4a0f1327f7292d0e64e015e6677
ustilizando esos modelos para esta tarea, añadiendo scheduler y early stopping

Resultados ejecución 3:
Binario - Accuracy: 0.5000, F1: 0.3333
Multiclase - Accuracy: 0.8750, F1: 0.8027

=== RESULTADOS PROMEDIO (3 EJECUCIONES) ===
binary_accuracy: 0.5796 ± 0.0585
binary_f1: 0.3661 ± 0.0239
multi_accuracy: 0.8889 ± 0.0196
multi_f1: 0.8450 ± 0.0331



=== RESULTADOS PROMEDIO (3 EJECUCIONES) === Con bistm inicial se ve que peor
binary_accuracy: 0.5476 ± 0.0533
binary_f1: 0.3601 ± 0.0315
multi_accuracy: 0.9187 ± 0.0144
multi_f1: 0.8656 ± 0.0271


=== RESULTADOS PROMEDIO (3 EJECUCIONES) ===     focal, ligeramente mejores tarda mas?
binary_accuracy: 0.5476 ± 0.0533
binary_f1: 0.3601 ± 0.0315
multi_accuracy: 0.9075 ± 0.0029
multi_f1: 0.8648 ± 0.0159

=== RESULTADOS PROMEDIO (3 EJECUCIONES) ===
binary_accuracy: 0.5476 ± 0.0533
binary_f1: 0.3601 ± 0.0315
multi_accuracy: 0.9075 ± 0.0029
multi_f1: 0.8648 ± 0.0159



AHORA CON SERVIDOR

=== ANÁLISIS FINAL DE LAS 3 EJECUCIONES ===

ESTADÍSTICAS DE MÉTRICAS BINARIAS: (sin groupdro)
Accuracy       : Media = 0.5278, Desviación Estándar = 0.0227
Macro_P        : Media = 0.5636, Desviación Estándar = 0.0452
Macro_R        : Media = 0.5278, Desviación Estándar = 0.0227
Macro_F1       : Media = 0.4352, Desviación Estándar = 0.0505
Micro_P        : Media = 0.5278, Desviación Estándar = 0.0227
Micro_R        : Media = 0.5278, Desviación Estándar = 0.0227
Micro_F1       : Media = 0.5278, Desviación Estándar = 0.0227
ERDE5          : Media = 0.7162, Desviación Estándar = 0.0179
ERDE30         : Media = 0.3046, Desviación Estándar = 0.0675
latencyTP      : Media = 18.1667, Desviación Estándar = 3.6591
speed          : Media = 0.8311, Desviación Estándar = 0.0355
latency-weightedF1: Media = 0.5506, Desviación Estándar = 0.0266

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9537, Desviación Estándar = 0.0262
Macro_P        : Media = 0.9321, Desviación Estándar = 0.0725
Macro_R        : Media = 0.9247, Desviación Estándar = 0.0673
Macro_F1       : Media = 0.9261, Desviación Estándar = 0.0715
Micro_P        : Media = 0.9537, Desviación Estándar = 0.0262
Micro_R        : Media = 0.9537, Desviación Estándar = 0.0262
Micro_F1       : Media = 0.9537, Desviación Estándar = 0.0262



 (con groupdro, ligeramente mejores resultados), sigue siendo true en los siguientes ya que usan groupdro 5 bastante bien 10 grupos saca peor resultados
ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.5278, Desviación Estándar = 0.0227
Macro_P        : Media = 0.5592, Desviación Estándar = 0.0595
Macro_R        : Media = 0.5278, Desviación Estándar = 0.0227
Macro_F1       : Media = 0.4499, Desviación Estándar = 0.0522
Micro_P        : Media = 0.5278, Desviación Estándar = 0.0227
Micro_R        : Media = 0.5278, Desviación Estándar = 0.0227
Micro_F1       : Media = 0.5278, Desviación Estándar = 0.0227
ERDE5          : Media = 0.7074, Desviación Estándar = 0.0229
ERDE30         : Media = 0.3116, Desviación Estándar = 0.0878
latencyTP      : Media = 17.3333, Desviación Estándar = 3.0912
speed          : Media = 0.8390, Desviación Estándar = 0.0300
latency-weightedF1: Media = 0.5477, Desviación Estándar = 0.0387


CON ULTIMO MODELO 29/03 MODULO PALABRAS VICIOSAS

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.5648, Desviación Estándar = 0.0524
Macro_P        : Media = 0.5720, Desviación Estándar = 0.0556
Macro_R        : Media = 0.5648, Desviación Estándar = 0.0524
Macro_F1       : Media = 0.5509, Desviación Estándar = 0.0584
Micro_P        : Media = 0.5648, Desviación Estándar = 0.0524
Micro_R        : Media = 0.5648, Desviación Estándar = 0.0524
Micro_F1       : Media = 0.5648, Desviación Estándar = 0.0524
ERDE5          : Media = 0.6450, Desviación Estándar = 0.0246
ERDE30         : Media = 0.3829, Desviación Estándar = 0.0742
latencyTP      : Media = 21.1667, Desviación Estándar = 4.3653
speed          : Media = 0.8023, Desviación Estándar = 0.0420
latency-weightedF1: Media = 0.5017, Desviación Estándar = 0.0637




Con atencion mejorada, probar a cambiar textos???, mejor 29/03/2025
ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.5741, Desviación Estándar = 0.0944
Macro_P        : Media = 0.5815, Desviación Estándar = 0.1016
Macro_R        : Media = 0.5741, Desviación Estándar = 0.0944
Macro_F1       : Media = 0.5705, Desviación Estándar = 0.0912
Micro_P        : Media = 0.5741, Desviación Estándar = 0.0944
Micro_R        : Media = 0.5741, Desviación Estándar = 0.0944
Micro_F1       : Media = 0.5741, Desviación Estándar = 0.0944
ERDE5          : Media = 0.5867, Desviación Estándar = 0.0396
ERDE30         : Media = 0.4035, Desviación Estándar = 0.0582
latencyTP      : Media = 20.5000, Desviación Estándar = 1.0801
speed          : Media = 0.8087, Desviación Estándar = 0.0102
latency-weightedF1: Media = 0.4404, Desviación Estándar = 0.0594




MULTICLASE AUMENTANDO SIMPLEMENTA EL LR A 0.005 DE 0.0005
ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9722, Desviación Estándar = 0.0227
Macro_P        : Media = 0.9444, Desviación Estándar = 0.0671
Macro_R        : Media = 0.9415, Desviación Estándar = 0.0641
Macro_F1       : Media = 0.9422, Desviación Estándar = 0.0659
Micro_P        : Media = 0.9722, Desviación Estándar = 0.0227
Micro_R        : Media = 0.9722, Desviación Estándar = 0.0227
Micro_F1       : Media = 0.9722, Desviación Estándar = 0.0227
===================================================



Convoluciones es malo, no ayudan mucho


===================================================

SI EL REGRESOR ES IGUAL AL MULTICLASE (TENIENDO EN CUENTA LA SALIDA)

BINARY METRICS: =============================
Accuracy:0.5277777777777778
Macro precision:0.537037037037037
Macro recall:0.5277777777777778
Macro f1:0.4962962962962963
Micro precision:0.5277777777777778
Micro recall:0.5277777777777778
Micro f1:0.5277777777777778
LATENCY-BASED METRICS: =============================
ERDE_5:0.6767482991816792
ERDE_30:0.32296686666534397
Median latency:12.5
Speed:0.8860745734223818
latency-weightedF1:0.551335290129482

===================================================

ACUMULANDO MENSAJES DURANTE LAS RONDAS, ANTES DABA 0,5705 Y AHORA 0.6208 MEJORA SUSTANCIAL
ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.6296, Desviación Estándar = 0.0346
Macro_P        : Media = 0.6431, Desviación Estándar = 0.0376
Macro_R        : Media = 0.6296, Desviación Estándar = 0.0346
Macro_F1       : Media = 0.6208, Desviación Estándar = 0.0358
Micro_P        : Media = 0.6296, Desviación Estándar = 0.0346
Micro_R        : Media = 0.6296, Desviación Estándar = 0.0346
Micro_F1       : Media = 0.6296, Desviación Estándar = 0.0346
ERDE5          : Media = 0.5925, Desviación Estándar = 0.0096
ERDE30         : Media = 0.2774, Desviación Estándar = 0.0221
latencyTP      : Media = 8.8333, Desviación Estándar = 1.9293
speed          : Media = 0.9222, Desviación Estándar = 0.0191
latency-weightedF1: Media = 0.6244, Desviación Estándar = 0.0270


>Para dataaugmentation se ha pensado en hacer varias cosas, primero era usar un modelo de hugginface pero ejecutarlos con el hardware actual resulto dificil, 
los unicos modelos que se podian procesar eran los de 1B de parametros y rendian con mucha demencia, luego se penso en usar una api que scrapease chatgpt pero se descarto rapidamete,
lo que mejor ha funciona es usar google studio el cual tiene cuotas bastante generosas y ademas como ya tenia cuenta de factuaracion gracias a los creditos gratuitos de la uni para otras asignaturas
lo aumentaban, por lo que el data augmentation se ha realizado con google aistudio.

##################################
PRUEBA DE EMBEDDINGS, LO ANTERIOR ERA CON HIIAMSID

>ignacio/bert-sentiment-analysys o como sea, el de ignacio

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.6481, Desviación Estándar = 0.0131
Macro_P        : Media = 0.7707, Desviación Estándar = 0.0349
Macro_R        : Media = 0.6481, Desviación Estándar = 0.0131
Macro_F1       : Media = 0.6038, Desviación Estándar = 0.0165
Micro_P        : Media = 0.6481, Desviación Estándar = 0.0131
Micro_R        : Media = 0.6481, Desviación Estándar = 0.0131
Micro_F1       : Media = 0.6481, Desviación Estándar = 0.0131
ERDE5          : Media = 0.6060, Desviación Estándar = 0.0076
ERDE30         : Media = 0.2029, Desviación Estándar = 0.0358
latencyTP      : Media = 7.3333, Desviación Estándar = 0.2357
speed          : Media = 0.9371, Desviación Estándar = 0.0023
latency-weightedF1: Media = 0.6897, Desviación Estándar = 0.0089

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9722, Desviación Estándar = 0.0227
Macro_P        : Media = 0.9438, Desviación Estándar = 0.0653
Macro_R        : Media = 0.9091, Desviación Estándar = 0.0649
Macro_F1       : Media = 0.9194, Desviación Estándar = 0.0611
Micro_P        : Media = 0.9722, Desviación Estándar = 0.0227
Micro_R        : Media = 0.9722, Desviación Estándar = 0.0227
Micro_F1       : Media = 0.9722, Desviación Estándar = 0.0227



>models/embedding-001 google, muy prometedor para el multiclase

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.5556, Desviación Estándar = 0.0393
Macro_P        : Media = 0.5587, Desviación Estándar = 0.0373
Macro_R        : Media = 0.5556, Desviación Estándar = 0.0393
Macro_F1       : Media = 0.5449, Desviación Estándar = 0.0485
Micro_P        : Media = 0.5556, Desviación Estándar = 0.0393
Micro_R        : Media = 0.5556, Desviación Estándar = 0.0393
Micro_F1       : Media = 0.5556, Desviación Estándar = 0.0393
ERDE5          : Media = 0.5925, Desviación Estándar = 0.0333
ERDE30         : Media = 0.3472, Desviación Estándar = 0.0393
latencyTP      : Media = 7.3333, Desviación Estándar = 0.4714
speed          : Media = 0.9371, Desviación Estándar = 0.0047
latency-weightedF1: Media = 0.5540, Desviación Estándar = 0.0323

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9907, Desviación Estándar = 0.0131
Macro_P        : Media = 0.9722, Desviación Estándar = 0.0393
Macro_R        : Media = 0.9924, Desviación Estándar = 0.0107
Macro_F1       : Media = 0.9794, Desviación Estándar = 0.0292
Micro_P        : Media = 0.9907, Desviación Estándar = 0.0131
Micro_R        : Media = 0.9907, Desviación Estándar = 0.0131
Micro_F1       : Media = 0.9907, Desviación Estándar = 0.0131


>models/text-embedding-004 google, muy prometedor en general mejor que los de huggingface y rinde en conjunto mejor, me quedare con este de momento 30/03


ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.6944, Desviación Estándar = 0.0227
Macro_P        : Media = 0.6972, Desviación Estándar = 0.0239
Macro_R        : Media = 0.6944, Desviación Estándar = 0.0227
Macro_F1       : Media = 0.6934, Desviación Estándar = 0.0223
Micro_P        : Media = 0.6944, Desviación Estándar = 0.0227
Micro_R        : Media = 0.6944, Desviación Estándar = 0.0227
Micro_F1       : Media = 0.6944, Desviación Estándar = 0.0227
ERDE5          : Media = 0.5644, Desviación Estándar = 0.0103
ERDE30         : Media = 0.2907, Desviación Estándar = 0.0324
latencyTP      : Media = 11.6667, Desviación Estándar = 3.2998
speed          : Media = 0.8944, Desviación Estándar = 0.0325
latency-weightedF1: Media = 0.6327, Desviación Estándar = 0.0390

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9815, Desviación Estándar = 0.0262
Macro_P        : Media = 0.9508, Desviación Estándar = 0.0696
Macro_R        : Media = 0.9508, Desviación Estándar = 0.0696
Macro_F1       : Media = 0.9508, Desviación Estándar = 0.0696
Micro_P        : Media = 0.9815, Desviación Estándar = 0.0262
Micro_R        : Media = 0.9815, Desviación Estándar = 0.0262
Micro_F1       : Media = 0.9815, Desviación Estándar = 0.0262


###################################

Probando con data augmentation, solo los de clase 0 aumentados, la evaluacion no es muy honesta ya que el original puede estar en 
el train y la variacion en el test, aqui se puede ver que al añadir solo los de la clase 0 el modelo sufre un cierto bias hacia esas misma clase

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.7346, Desviación Estándar = 0.0531
Macro_P        : Media = 0.6047, Desviación Estándar = 0.1936
Macro_R        : Media = 0.6481, Desviación Estándar = 0.1101
Macro_F1       : Media = 0.6189, Desviación Estándar = 0.1580
Micro_P        : Media = 0.7346, Desviación Estándar = 0.0531 	FALSO NO CONTAR
Micro_R        : Media = 0.7346, Desviación Estándar = 0.0531
Micro_F1       : Media = 0.7346, Desviación Estándar = 0.0531
ERDE5          : Media = 0.3461, Desviación Estándar = 0.0112
ERDE30         : Media = 0.2482, Desviación Estándar = 0.0640
latencyTP      : Media = 8.5000, Desviación Estándar = 6.5701
speed          : Media = 0.5892, Desviación Estándar = 0.4174
latency-weightedF1: Media = 0.3690, Desviación Estándar = 0.2679

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9938, Desviación Estándar = 0.0087
Macro_P        : Media = 0.9951, Desviación Estándar = 0.0069
Macro_R        : Media = 0.9792, Desviación Estándar = 0.0295
Macro_F1       : Media = 0.9856, Desviación Estándar = 0.0204 	FALSO NO CONTAR
Micro_P        : Media = 0.9938, Desviación Estándar = 0.0087
Micro_R        : Media = 0.9938, Desviación Estándar = 0.0087
Micro_F1       : Media = 0.9938, Desviación Estándar = 0.0087


CON 0 BIEN HECHA, Se ve una gran mejora en el regresor que es lo que se buscaba y una pequeña mejor en el multiclase,
alguno creeria que realmente empeora teniendo en cuenta el anterior pero hay que tener en cuenta que en el anterior no era
muy honesto ya que datos del train estaban con su variacion en el test, por lo que no es un resultado realista, aqui
esto se ha solucionado haciendo que vayan por grupos

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.7716, Desviación Estándar = 0.0175
Macro_P        : Media = 0.7477, Desviación Estándar = 0.0214
Macro_R        : Media = 0.7407, Desviación Estándar = 0.0472
Macro_F1       : Media = 0.7383, Desviación Estándar = 0.0331
Micro_P        : Media = 0.7716, Desviación Estándar = 0.0175
Micro_R        : Media = 0.7716, Desviación Estándar = 0.0175
Micro_F1       : Media = 0.7716, Desviación Estándar = 0.0175
ERDE5          : Media = 0.3612, Desviación Estándar = 0.0046
ERDE30         : Media = 0.2041, Desviación Estándar = 0.0319
latencyTP      : Media = 18.3333, Desviación Estándar = 6.0185
speed          : Media = 0.8298, Desviación Estándar = 0.0584
latency-weightedF1: Media = 0.5405, Desviación Estándar = 0.0918

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9753, Desviación Estándar = 0.0231
Macro_P        : Media = 0.9468, Desviación Estándar = 0.0630
Macro_R        : Media = 0.9580, Desviación Estándar = 0.0515
Macro_F1       : Media = 0.9512, Desviación Estándar = 0.0587
Micro_P        : Media = 0.9753, Desviación Estándar = 0.0231
Micro_R        : Media = 0.9753, Desviación Estándar = 0.0231
Micro_F1       : Media = 0.9753, Desviación Estándar = 0.0231

CON 0 Y CON 1, UN RESULTADO MUY CURIOSO, EL MULTI MEJORA Y EL REGRESOR EMPEORA???

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.6806, Desviación Estándar = 0.0196
Macro_P        : Media = 0.6833, Desviación Estándar = 0.0235
Macro_R        : Media = 0.6806, Desviación Estándar = 0.0196
Macro_F1       : Media = 0.6796, Desviación Estándar = 0.0183
Micro_P        : Media = 0.6806, Desviación Estándar = 0.0196
Micro_R        : Media = 0.6806, Desviación Estándar = 0.0196
Micro_F1       : Media = 0.6806, Desviación Estándar = 0.0196
ERDE5          : Media = 0.5669, Desviación Estándar = 0.0071
ERDE30         : Media = 0.3070, Desviación Estándar = 0.0308
latencyTP      : Media = 13.3333, Desviación Estándar = 3.1710
speed          : Media = 0.8780, Desviación Estándar = 0.0311
latency-weightedF1: Media = 0.6061, Desviación Estándar = 0.0496

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9861, Desviación Estándar = 0.0113
Macro_P        : Media = 0.9678, Desviación Estándar = 0.0366
Macro_R        : Media = 0.9894, Desviación Estándar = 0.0093
Macro_F1       : Media = 0.9756, Desviación Estándar = 0.0269
Micro_P        : Media = 0.9861, Desviación Estándar = 0.0113
Micro_R        : Media = 0.9861, Desviación Estándar = 0.0113
Micro_F1       : Media = 0.9861, Desviación Estándar = 0.0113


SOLO CON 1

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.7407, Desviación Estándar = 0.0262
Macro_P        : Media = 0.7111, Desviación Estándar = 0.0326
Macro_R        : Media = 0.6713, Desviación Estándar = 0.0365
Macro_F1       : Media = 0.6802, Desviación Estándar = 0.0382
Micro_P        : Media = 0.7407, Desviación Estándar = 0.0262
Micro_R        : Media = 0.7407, Desviación Estándar = 0.0262
Micro_F1       : Media = 0.7407, Desviación Estándar = 0.0262
ERDE5          : Media = 0.7112, Desviación Estándar = 0.0257
ERDE30         : Media = 0.2338, Desviación Estándar = 0.0377
latencyTP      : Media = 7.6667, Desviación Estándar = 1.3123
speed          : Media = 0.9338, Desviación Estándar = 0.0130
latency-weightedF1: Media = 0.7650, Desviación Estándar = 0.0228

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9815, Desviación Estándar = 0.0151
Macro_P        : Media = 0.9627, Desviación Estándar = 0.0433
Macro_R        : Media = 0.9395, Desviación Estándar = 0.0432
Macro_F1       : Media = 0.9483, Desviación Estándar = 0.0402
Micro_P        : Media = 0.9815, Desviación Estándar = 0.0151
Micro_R        : Media = 0.9815, Desviación Estándar = 0.0151
Micro_F1       : Media = 0.9815, Desviación Estándar = 0.0151


###
SE HA HECHO QUE EN EL REGRESOR Y ATENCION SE INICIEN LOS PESOS COMO EN EN EL DE 2023 CON XAVIER

CON LAS CLASES 0 Y 1 EL REGRESOR MEJORA!
ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.6898, Desviación Estándar = 0.0458
Macro_P        : Media = 0.7012, Desviación Estándar = 0.0590
Macro_R        : Media = 0.6898, Desviación Estándar = 0.0458
Macro_F1       : Media = 0.6869, Desviación Estándar = 0.0432
Micro_P        : Media = 0.6898, Desviación Estándar = 0.0458
Micro_R        : Media = 0.6898, Desviación Estándar = 0.0458
Micro_F1       : Media = 0.6898, Desviación Estándar = 0.0458
ERDE5          : Media = 0.5716, Desviación Estándar = 0.0164
ERDE30         : Media = 0.2885, Desviación Estándar = 0.0611
latencyTP      : Media = 13.6667, Desviación Estándar = 1.6997
speed          : Media = 0.8747, Desviación Estándar = 0.0166
latency-weightedF1: Media = 0.6159, Desviación Estándar = 0.0597

CON SOLO LA CLASE 0
RESULTADOS NO TAN BUENOS COMO LO ANTERIOR???


# Se ha sacado la data augmentation del dev y del test para que la evaluacion sea mas realista

### Solo con la clase 0
ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.6759, Desviación Estándar = 0.0346
Macro_P        : Media = 0.6950, Desviación Estándar = 0.0592
Macro_R        : Media = 0.6759, Desviación Estándar = 0.0346
Macro_F1       : Media = 0.6707, Desviación Estándar = 0.0285
Micro_P        : Media = 0.6759, Desviación Estándar = 0.0346
Micro_R        : Media = 0.6759, Desviación Estándar = 0.0346
Micro_F1       : Media = 0.6759, Desviación Estándar = 0.0346
ERDE5          : Media = 0.5743, Desviación Estándar = 0.0131
ERDE30         : Media = 0.2916, Desviación Estándar = 0.0629
latencyTP      : Media = 14.8333, Desviación Estándar = 5.5428
speed          : Media = 0.8636, Desviación Estándar = 0.0543
latency-weightedF1: Media = 0.5986, Desviación Estándar = 0.0905

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9815, Desviación Estándar = 0.0262
Macro_P        : Media = 0.9508, Desviación Estándar = 0.0696
Macro_R        : Media = 0.9508, Desviación Estándar = 0.0696
Macro_F1       : Media = 0.9508, Desviación Estándar = 0.0696
Micro_P        : Media = 0.9815, Desviación Estándar = 0.0262
Micro_R        : Media = 0.9815, Desviación Estándar = 0.0262
Micro_F1       : Media = 0.9815, Desviación Estándar = 0.0262

### Con la clase 0 y 1 conjuntas

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.6759, Desviación Estándar = 0.0131
Macro_P        : Media = 0.6797, Desviación Estándar = 0.0112
Macro_R        : Media = 0.6759, Desviación Estándar = 0.0131
Macro_F1       : Media = 0.6741, Desviación Estándar = 0.0143
Micro_P        : Media = 0.6759, Desviación Estándar = 0.0131
Micro_R        : Media = 0.6759, Desviación Estándar = 0.0131
Micro_F1       : Media = 0.6759, Desviación Estándar = 0.0131
ERDE5          : Media = 0.5696, Desviación Estándar = 0.0180
ERDE30         : Media = 0.2836, Desviación Estándar = 0.0254
latencyTP      : Media = 12.1667, Desviación Estándar = 2.3214
speed          : Media = 0.8895, Desviación Estándar = 0.0228
latency-weightedF1: Media = 0.6136, Desviación Estándar = 0.0116

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9907, Desviación Estándar = 0.0131
Macro_P        : Media = 0.9931, Desviación Estándar = 0.0098
Macro_R        : Media = 0.9583, Desviación Estándar = 0.0589
Macro_F1       : Media = 0.9686, Desviación Estándar = 0.0444
Micro_P        : Media = 0.9907, Desviación Estándar = 0.0131
Micro_R        : Media = 0.9907, Desviación Estándar = 0.0131
Micro_F1       : Media = 0.9907, Desviación Estándar = 0.0131




### Sin ninguna
ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.6944, Desviación Estándar = 0.0227
Macro_P        : Media = 0.7109, Desviación Estándar = 0.0105
Macro_R        : Media = 0.6944, Desviación Estándar = 0.0227
Macro_F1       : Media = 0.6877, Desviación Estándar = 0.0296
Micro_P        : Media = 0.6944, Desviación Estándar = 0.0227
Micro_R        : Media = 0.6944, Desviación Estándar = 0.0227
Micro_F1       : Media = 0.6944, Desviación Estándar = 0.0227
ERDE5          : Media = 0.5700, Desviación Estándar = 0.0316
ERDE30         : Media = 0.2921, Desviación Estándar = 0.0118
latencyTP      : Media = 12.0000, Desviación Estándar = 3.5355
speed          : Media = 0.8911, Desviación Estándar = 0.0348
latency-weightedF1: Media = 0.6479, Desviación Estándar = 0.0195

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9815, Desviación Estándar = 0.0262
Macro_P        : Media = 0.9508, Desviación Estándar = 0.0696
Macro_R        : Media = 0.9508, Desviación Estándar = 0.0696
Macro_F1       : Media = 0.9508, Desviación Estándar = 0.0696
Micro_P        : Media = 0.9815, Desviación Estándar = 0.0262
Micro_R        : Media = 0.9815, Desviación Estándar = 0.0262
Micro_F1       : Media = 0.9815, Desviación Estándar = 0.0262



## Mejora al clasificador para reducir overfitting



# CON 0 Y CON NEURONAS NORMALES 128
ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.7130, Desviación Estándar = 0.0472
Macro_P        : Media = 0.7232, Desviación Estándar = 0.0520
Macro_R        : Media = 0.7130, Desviación Estándar = 0.0472
Macro_F1       : Media = 0.7100, Desviación Estándar = 0.0468
Micro_P        : Media = 0.7130, Desviación Estándar = 0.0472
Micro_R        : Media = 0.7130, Desviación Estándar = 0.0472
Micro_F1       : Media = 0.7130, Desviación Estándar = 0.0472
ERDE5          : Media = 0.5639, Desviación Estándar = 0.0170
ERDE30         : Media = 0.3396, Desviación Estándar = 0.0245
latencyTP      : Media = 22.0000, Desviación Estándar = 1.0801
speed          : Media = 0.7941, Desviación Estándar = 0.0103
latency-weightedF1: Media = 0.5595, Desviación Estándar = 0.0602

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9815, Desviación Estándar = 0.0262
Macro_P        : Media = 0.9508, Desviación Estándar = 0.0696
Macro_R        : Media = 0.9508, Desviación Estándar = 0.0696
Macro_F1       : Media = 0.9508, Desviación Estándar = 0.0696
Micro_P        : Media = 0.9815, Desviación Estándar = 0.0262
Micro_R        : Media = 0.9815, Desviación Estándar = 0.0262
Micro_F1       : Media = 0.9815, Desviación Estándar = 0.0262


# 0 CON 164 NEURONAS
#### Mas si, pero claro fijate en la desviacion, esta la achaco a la suerte, comentar eso
ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.7130, Desviación Estándar = 0.0693
Macro_P        : Media = 0.7218, Desviación Estándar = 0.0806
Macro_R        : Media = 0.7130, Desviación Estándar = 0.0693
Macro_F1       : Media = 0.7115, Desviación Estándar = 0.0677
Micro_P        : Media = 0.7130, Desviación Estándar = 0.0693
Micro_R        : Media = 0.7130, Desviación Estándar = 0.0693
Micro_F1       : Media = 0.7130, Desviación Estándar = 0.0693       
ERDE5          : Media = 0.5722, Desviación Estándar = 0.0061
ERDE30         : Media = 0.2876, Desviación Estándar = 0.0327
latencyTP      : Media = 16.1667, Desviación Estándar = 1.8409
speed          : Media = 0.8503, Desviación Estándar = 0.0179
latency-weightedF1: Media = 0.6085, Desviación Estándar = 0.0817

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9907, Desviación Estándar = 0.0131
Macro_P        : Media = 0.9722, Desviación Estándar = 0.0393
Macro_R        : Media = 0.9924, Desviación Estándar = 0.0107
Macro_F1       : Media = 0.9794, Desviación Estándar = 0.0292
Micro_P        : Media = 0.9907, Desviación Estándar = 0.0131
Micro_R        : Media = 0.9907, Desviación Estándar = 0.0131
Micro_F1       : Media = 0.9907, Desviación Estándar = 0.0131

# 0 CON 214

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.6944, Desviación Estándar = 0.0600
Macro_P        : Media = 0.6987, Desviación Estándar = 0.0596
Macro_R        : Media = 0.6944, Desviación Estándar = 0.0600
Macro_F1       : Media = 0.6922, Desviación Estándar = 0.0617
Micro_P        : Media = 0.6944, Desviación Estándar = 0.0600
Micro_R        : Media = 0.6944, Desviación Estándar = 0.0600
Micro_F1       : Media = 0.6944, Desviación Estándar = 0.0600
ERDE5          : Media = 0.5694, Desviación Estándar = 0.0068
ERDE30         : Media = 0.3379, Desviación Estándar = 0.0410
latencyTP      : Media = 19.6667, Desviación Estándar = 4.1096
speed          : Media = 0.8167, Desviación Estándar = 0.0394
latency-weightedF1: Media = 0.5626, Desviación Estándar = 0.0974

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 1.0000, Desviación Estándar = 0.0000
Macro_P        : Media = 1.0000, Desviación Estándar = 0.0000
Macro_R        : Media = 1.0000, Desviación Estándar = 0.0000
Macro_F1       : Media = 1.0000, Desviación Estándar = 0.0000
Micro_P        : Media = 1.0000, Desviación Estándar = 0.0000
Micro_R        : Media = 1.0000, Desviación Estándar = 0.0000
Micro_F1       : Media = 1.0000, Desviación Estándar = 0.0000

# 0 CON 156 


ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.6852, Desviación Estándar = 0.0262
Macro_P        : Media = 0.6926, Desviación Estándar = 0.0292
Macro_R        : Media = 0.6852, Desviación Estándar = 0.0262
Macro_F1       : Media = 0.6823, Desviación Estándar = 0.0258
Micro_P        : Media = 0.6852, Desviación Estándar = 0.0262
Micro_R        : Media = 0.6852, Desviación Estándar = 0.0262
Micro_F1       : Media = 0.6852, Desviación Estándar = 0.0262
ERDE5          : Media = 0.5638, Desviación Estándar = 0.0062
ERDE30         : Media = 0.3381, Desviación Estándar = 0.0354
latencyTP      : Media = 15.6667, Desviación Estándar = 3.3993
speed          : Media = 0.8553, Desviación Estándar = 0.0332
latency-weightedF1: Media = 0.5783, Desviación Estándar = 0.0700

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9815, Desviación Estándar = 0.0262
Macro_P        : Media = 0.9508, Desviación Estándar = 0.0696
Macro_R        : Media = 0.9508, Desviación Estándar = 0.0696
Macro_F1       : Media = 0.9508, Desviación Estándar = 0.0696
Micro_P        : Media = 0.9815, Desviación Estándar = 0.0262
Micro_R        : Media = 0.9815, Desviación Estándar = 0.0262
Micro_F1       : Media = 0.9815, Desviación Estándar = 0.0262


# CON 1 Y 0 CON NEURONAS NORMALES 128

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.6944, Desviación Estándar = 0.0454
Macro_P        : Media = 0.7034, Desviación Estándar = 0.0538
Macro_R        : Media = 0.6944, Desviación Estándar = 0.0454
Macro_F1       : Media = 0.6920, Desviación Estándar = 0.0435
Micro_P        : Media = 0.6944, Desviación Estándar = 0.0454
Micro_R        : Media = 0.6944, Desviación Estándar = 0.0454
Micro_F1       : Media = 0.6944, Desviación Estándar = 0.0454
ERDE5          : Media = 0.5729, Desviación Estándar = 0.0092
ERDE30         : Media = 0.2673, Desviación Estándar = 0.0259
latencyTP      : Media = 12.6667, Desviación Estándar = 2.8964
speed          : Media = 0.8845, Desviación Estándar = 0.0284
latency-weightedF1: Media = 0.6336, Desviación Estándar = 0.0586

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9815, Desviación Estándar = 0.0262
Macro_P        : Media = 0.9508, Desviación Estándar = 0.0696
Macro_R        : Media = 0.9508, Desviación Estándar = 0.0696
Macro_F1       : Media = 0.9508, Desviación Estándar = 0.0696
Micro_P        : Media = 0.9815, Desviación Estándar = 0.0262
Micro_R        : Media = 0.9815, Desviación Estándar = 0.0262
Micro_F1       : Media = 0.9815, Desviación Estándar = 0.0262

# 1 Y 0 CON 164

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.6574, Desviación Estándar = 0.0346
Macro_P        : Media = 0.6626, Desviación Estándar = 0.0368
Macro_R        : Media = 0.6574, Desviación Estándar = 0.0346
Macro_F1       : Media = 0.6549, Desviación Estándar = 0.0341
Micro_P        : Media = 0.6574, Desviación Estándar = 0.0346
Micro_R        : Media = 0.6574, Desviación Estándar = 0.0346
Micro_F1       : Media = 0.6574, Desviación Estándar = 0.0346
ERDE5          : Media = 0.5780, Desviación Estándar = 0.0015
ERDE30         : Media = 0.2918, Desviación Estándar = 0.0103
latencyTP      : Media = 11.6667, Desviación Estándar = 1.4337
speed          : Media = 0.8943, Desviación Estándar = 0.0141
latency-weightedF1: Media = 0.6108, Desviación Estándar = 0.0320

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9815, Desviación Estándar = 0.0262
Macro_P        : Media = 0.9508, Desviación Estándar = 0.0696
Macro_R        : Media = 0.9508, Desviación Estándar = 0.0696
Macro_F1       : Media = 0.9508, Desviación Estándar = 0.0696
Micro_P        : Media = 0.9815, Desviación Estándar = 0.0262
Micro_R        : Media = 0.9815, Desviación Estándar = 0.0262
Micro_F1       : Media = 0.9815, Desviación Estándar = 0.0262

# CON NINGUNO


ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.6667, Desviación Estándar = 0.0393
Macro_P        : Media = 0.6692, Desviación Estándar = 0.0401
Macro_R        : Media = 0.6667, Desviación Estándar = 0.0393
Macro_F1       : Media = 0.6655, Desviación Estándar = 0.0393
Micro_P        : Media = 0.6667, Desviación Estándar = 0.0393
Micro_R        : Media = 0.6667, Desviación Estándar = 0.0393
Micro_F1       : Media = 0.6667, Desviación Estándar = 0.0393
ERDE5          : Media = 0.5747, Desviación Estándar = 0.0249
ERDE30         : Media = 0.2767, Desviación Estándar = 0.0242
latencyTP      : Media = 9.5000, Desviación Estándar = 1.4720
speed          : Media = 0.9156, Desviación Estándar = 0.0145
latency-weightedF1: Media = 0.6269, Desviación Estándar = 0.0433

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9722, Desviación Estándar = 0.0227
Macro_P        : Media = 0.9438, Desviación Estándar = 0.0653
Macro_R        : Media = 0.9091, Desviación Estándar = 0.0649
Macro_F1       : Media = 0.9194, Desviación Estándar = 0.0611
Micro_P        : Media = 0.9722, Desviación Estándar = 0.0227
Micro_R        : Media = 0.9722, Desviación Estándar = 0.0227
Micro_F1       : Media = 0.9722, Desviación Estándar = 0.0227






# Mejora

## rework forward atencion 03/04


### Con clase 0
=== ANÁLISIS FINAL DE LAS 3 EJECUCIONES ===

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.7130, Desviación Estándar = 0.0346
Macro_P        : Media = 0.7148, Desviación Estándar = 0.0342
Macro_R        : Media = 0.7130, Desviación Estándar = 0.0346
Macro_F1       : Media = 0.7123, Desviación Estándar = 0.0350
Micro_P        : Media = 0.7130, Desviación Estándar = 0.0346
Micro_R        : Media = 0.7130, Desviación Estándar = 0.0346
Micro_F1       : Media = 0.7130, Desviación Estándar = 0.0346
ERDE5          : Media = 0.5732, Desviación Estándar = 0.0134
ERDE30         : Media = 0.2811, Desviación Estándar = 0.0532
latencyTP      : Media = 13.0000, Desviación Estándar = 2.5495
speed          : Media = 0.8812, Desviación Estándar = 0.0250
latency-weightedF1: Media = 0.6393, Desviación Estándar = 0.0236

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9815, Desviación Estándar = 0.0262
Macro_P        : Media = 0.9508, Desviación Estándar = 0.0696
Macro_R        : Media = 0.9508, Desviación Estándar = 0.0696
Macro_F1       : Media = 0.9508, Desviación Estándar = 0.0696
Micro_P        : Media = 0.9815, Desviación Estándar = 0.0262
Micro_R        : Media = 0.9815, Desviación Estándar = 0.0262
Micro_F1       : Media = 0.9815, Desviación Estándar = 0.0262


### Con NINGUNA


=== ANÁLISIS FINAL DE LAS 3 EJECUCIONES ===

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.6759, Desviación Estándar = 0.0524
Macro_P        : Media = 0.6777, Desviación Estándar = 0.0517
Macro_R        : Media = 0.6759, Desviación Estándar = 0.0524
Macro_F1       : Media = 0.6749, Desviación Estándar = 0.0530
Micro_P        : Media = 0.6759, Desviación Estándar = 0.0524
Micro_R        : Media = 0.6759, Desviación Estándar = 0.0524
Micro_F1       : Media = 0.6759, Desviación Estándar = 0.0524
ERDE5          : Media = 0.5654, Desviación Estándar = 0.0283
ERDE30         : Media = 0.2791, Desviación Estándar = 0.0501
latencyTP      : Media = 13.3333, Desviación Estándar = 3.8586
speed          : Media = 0.8781, Desviación Estándar = 0.0379
latency-weightedF1: Media = 0.6026, Desviación Estándar = 0.0741

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9722, Desviación Estándar = 0.0227
Macro_P        : Media = 0.9438, Desviación Estándar = 0.0653
Macro_R        : Media = 0.9091, Desviación Estándar = 0.0649
Macro_F1       : Media = 0.9194, Desviación Estándar = 0.0611
Micro_P        : Media = 0.9722, Desviación Estándar = 0.0227
Micro_R        : Media = 0.9722, Desviación Estándar = 0.0227
Micro_F1       : Media = 0.9722, Desviación Estándar = 0.0227

## Con 1 Y 0


ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.7130, Desviación Estándar = 0.0346
Macro_P        : Media = 0.7278, Desviación Estándar = 0.0524
Macro_R        : Media = 0.7130, Desviación Estándar = 0.0346
Macro_F1       : Media = 0.7097, Desviación Estándar = 0.0313
Micro_P        : Media = 0.7130, Desviación Estándar = 0.0346
Micro_R        : Media = 0.7130, Desviación Estándar = 0.0346
Micro_F1       : Media = 0.7130, Desviación Estándar = 0.0346
ERDE5          : Media = 0.5617, Desviación Estándar = 0.0142
ERDE30         : Media = 0.2783, Desviación Estándar = 0.0697
latencyTP      : Media = 14.3333, Desviación Estándar = 2.0548
speed          : Media = 0.8682, Desviación Estándar = 0.0201
latency-weightedF1: Media = 0.6317, Desviación Estándar = 0.0584

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9815, Desviación Estándar = 0.0262
Macro_P        : Media = 0.9508, Desviación Estándar = 0.0696
Macro_R        : Media = 0.9508, Desviación Estándar = 0.0696
Macro_F1       : Media = 0.9508, Desviación Estándar = 0.0696
Micro_P        : Media = 0.9815, Desviación Estándar = 0.0262
Micro_R        : Media = 0.9815, Desviación Estándar = 0.0262
Micro_F1       : Media = 0.9815, Desviación Estándar = 0.0262

## Con clase 1


=== ANÁLISIS FINAL DE LAS 3 EJECUCIONES ===

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.6296, Desviación Estándar = 0.0131
Macro_P        : Media = 0.6509, Desviación Estándar = 0.0103
Macro_R        : Media = 0.6296, Desviación Estándar = 0.0131
Macro_F1       : Media = 0.6158, Desviación Estándar = 0.0178
Micro_P        : Media = 0.6296, Desviación Estándar = 0.0131
Micro_R        : Media = 0.6296, Desviación Estándar = 0.0131
Micro_F1       : Media = 0.6296, Desviación Estándar = 0.0131
ERDE5          : Media = 0.5897, Desviación Estándar = 0.0280
ERDE30         : Media = 0.2505, Desviación Estándar = 0.0193
latencyTP      : Media = 8.6667, Desviación Estándar = 2.0548
speed          : Media = 0.9239, Desviación Estándar = 0.0203
latency-weightedF1: Media = 0.6350, Desviación Estándar = 0.0099

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9722, Desviación Estándar = 0.0227
Macro_P        : Media = 0.9438, Desviación Estándar = 0.0653
Macro_R        : Media = 0.9091, Desviación Estándar = 0.0649
Macro_F1       : Media = 0.9194, Desviación Estándar = 0.0611
Micro_P        : Media = 0.9722, Desviación Estándar = 0.0227
Micro_R        : Media = 0.9722, Desviación Estándar = 0.0227
Micro_F1       : Media = 0.9722, Desviación Estándar = 0.0227

# AÑADIENDO FECHA A LA MEJOR CLASE 0 AUMENTADA ANTERIOR /3 o /4 supongo


ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.7407, Desviación Estándar = 0.0797
Macro_P        : Media = 0.7458, Desviación Estándar = 0.0799
Macro_R        : Media = 0.7407, Desviación Estándar = 0.0797
Macro_F1       : Media = 0.7391, Desviación Estándar = 0.0805
Micro_P        : Media = 0.7407, Desviación Estándar = 0.0797
Micro_R        : Media = 0.7407, Desviación Estándar = 0.0797
Micro_F1       : Media = 0.7407, Desviación Estándar = 0.0797
ERDE5          : Media = 0.5284, Desviación Estándar = 0.0202
ERDE30         : Media = 0.3636, Desviación Estándar = 0.0671
latencyTP      : Media = 31.8333, Desviación Estándar = 11.6500
speed          : Media = 0.7063, Desviación Estándar = 0.1031
latency-weightedF1: Media = 0.5150, Desviación Estándar = 0.1234

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 1.0000, Desviación Estándar = 0.0000
Macro_P        : Media = 1.0000, Desviación Estándar = 0.0000
Macro_R        : Media = 1.0000, Desviación Estándar = 0.0000
Macro_F1       : Media = 1.0000, Desviación Estándar = 0.0000
Micro_P        : Media = 1.0000, Desviación Estándar = 0.0000
Micro_R        : Media = 1.0000, Desviación Estándar = 0.0000
Micro_F1       : Media = 1.0000, Desviación Estándar = 0.0000

## REDUCIENDO LA LATENCY a 3 + ronda/6 (mucho mejor, este modelo tiene mucha seguridad al parecer, vamos a bajar)

=== ANÁLISIS FINAL DE LAS 3 EJECUCIONES ===

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.7685, Desviación Estándar = 0.0131
Macro_P        : Media = 0.7688, Desviación Estándar = 0.0127
Macro_R        : Media = 0.7685, Desviación Estándar = 0.0131
Macro_F1       : Media = 0.7685, Desviación Estándar = 0.0132
Micro_P        : Media = 0.7685, Desviación Estándar = 0.0131
Micro_R        : Media = 0.7685, Desviación Estándar = 0.0131
Micro_F1       : Media = 0.7685, Desviación Estándar = 0.0131
ERDE5          : Media = 0.5236, Desviación Estándar = 0.0106
ERDE30         : Media = 0.3438, Desviación Estándar = 0.0652
latencyTP      : Media = 24.1667, Desviación Estándar = 9.1863
speed          : Media = 0.7754, Desviación Estándar = 0.0853
latency-weightedF1: Media = 0.5985, Desviación Estándar = 0.0728

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 1.0000, Desviación Estándar = 0.0000
Macro_P        : Media = 1.0000, Desviación Estándar = 0.0000
Macro_R        : Media = 1.0000, Desviación Estándar = 0.0000
Macro_F1       : Media = 1.0000, Desviación Estándar = 0.0000
Micro_P        : Media = 1.0000, Desviación Estándar = 0.0000
Micro_R        : Media = 1.0000, Desviación Estándar = 0.0000
Micro_F1       : Media = 1.0000, Desviación Estándar = 0.0000

## REDUCIENDO A 2 + ronda/8

=== ANÁLISIS FINAL DE LAS 3 EJECUCIONES ===

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.7593, Desviación Estándar = 0.0131
Macro_P        : Media = 0.7609, Desviación Estándar = 0.0144
Macro_R        : Media = 0.7593, Desviación Estándar = 0.0131
Macro_F1       : Media = 0.7589, Desviación Estándar = 0.0129
Micro_P        : Media = 0.7593, Desviación Estándar = 0.0131
Micro_R        : Media = 0.7593, Desviación Estándar = 0.0131
Micro_F1       : Media = 0.7593, Desviación Estándar = 0.0131
ERDE5          : Media = 0.5024, Desviación Estándar = 0.0055
ERDE30         : Media = 0.3251, Desviación Estándar = 0.0353
latencyTP      : Media = 21.0000, Desviación Estándar = 8.5245
speed          : Media = 0.8051, Desviación Estándar = 0.0804
latency-weightedF1: Media = 0.6168, Desviación Estándar = 0.0502

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 1.0000, Desviación Estándar = 0.0000
Macro_P        : Media = 1.0000, Desviación Estándar = 0.0000
Macro_R        : Media = 1.0000, Desviación Estándar = 0.0000
Macro_F1       : Media = 1.0000, Desviación Estándar = 0.0000
Micro_P        : Media = 1.0000, Desviación Estándar = 0.0000
Micro_R        : Media = 1.0000, Desviación Estándar = 0.0000
Micro_F1       : Media = 1.0000, Desviación Estándar = 0.0000




## REDUCIENDO A 2 + (ronda_actual // 20) Parece que añadir el tiempo/fecha le enseña a ser paciente "CURIOSO"


ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.7593, Desviación Estándar = 0.0262
Macro_P        : Media = 0.7698, Desviación Estándar = 0.0317
Macro_R        : Media = 0.7593, Desviación Estándar = 0.0262
Macro_F1       : Media = 0.7571, Desviación Estándar = 0.0253
Micro_P        : Media = 0.7593, Desviación Estándar = 0.0262
Micro_R        : Media = 0.7593, Desviación Estándar = 0.0262
Micro_F1       : Media = 0.7593, Desviación Estándar = 0.0262
ERDE5          : Media = 0.5071, Desviación Estándar = 0.0018
ERDE30         : Media = 0.2904, Desviación Estándar = 0.0358
latencyTP      : Media = 17.1667, Desviación Estándar = 7.6848
speed          : Media = 0.8415, Desviación Estándar = 0.0737
latency-weightedF1: Media = 0.6543, Desviación Estándar = 0.0489

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 1.0000, Desviación Estándar = 0.0000
Macro_P        : Media = 1.0000, Desviación Estándar = 0.0000
Macro_R        : Media = 1.0000, Desviación Estándar = 0.0000
Macro_F1       : Media = 1.0000, Desviación Estándar = 0.0000
Micro_P        : Media = 1.0000, Desviación Estándar = 0.0000
Micro_R        : Media = 1.0000, Desviación Estándar = 0.0000
Micro_F1       : Media = 1.0000, Desviación Estándar = 0.0000


## REDUCIENDO A  1 + (ronda_actual // 30)
=== ANÁLISIS FINAL DE LAS 3 EJECUCIONES ===

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.7037, Desviación Estándar = 0.0693
Macro_P        : Media = 0.7269, Desviación Estándar = 0.0649
Macro_R        : Media = 0.7037, Desviación Estándar = 0.0693
Macro_F1       : Media = 0.6934, Desviación Estándar = 0.0766
Micro_P        : Media = 0.7037, Desviación Estándar = 0.0693
Micro_R        : Media = 0.7037, Desviación Estándar = 0.0693
Micro_F1       : Media = 0.7037, Desviación Estándar = 0.0693
ERDE5          : Media = 0.5031, Desviación Estándar = 0.0426
ERDE30         : Media = 0.2438, Desviación Estándar = 0.0229
latencyTP      : Media = 12.6667, Desviación Estándar = 5.2493
speed          : Media = 0.8848, Desviación Estándar = 0.0513
latency-weightedF1: Media = 0.6616, Desviación Estándar = 0.0585

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 1.0000, Desviación Estándar = 0.0000
Macro_P        : Media = 1.0000, Desviación Estándar = 0.0000
Macro_R        : Media = 1.0000, Desviación Estándar = 0.0000
Macro_F1       : Media = 1.0000, Desviación Estándar = 0.0000
Micro_P        : Media = 1.0000, Desviación Estándar = 0.0000
Micro_R        : Media = 1.0000, Desviación Estándar = 0.0000
Micro_F1       : Media = 1.0000, Desviación Estándar = 0.0000



# Probar con 1+0 para este? 2 + (ronda_actual // 20)

=== ANÁLISIS FINAL DE LAS 3 EJECUCIONES ===

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.7315, Desviación Estándar = 0.0262
Macro_P        : Media = 0.7360, Desviación Estándar = 0.0256
Macro_R        : Media = 0.7315, Desviación Estándar = 0.0262
Macro_F1       : Media = 0.7301, Desviación Estándar = 0.0267
Micro_P        : Media = 0.7315, Desviación Estándar = 0.0262
Micro_R        : Media = 0.7315, Desviación Estándar = 0.0262
Micro_F1       : Media = 0.7315, Desviación Estándar = 0.0262
ERDE5          : Media = 0.5069, Desviación Estándar = 0.0184
ERDE30         : Media = 0.2603, Desviación Estándar = 0.0275
latencyTP      : Media = 13.1667, Desviación Estándar = 3.2745
speed          : Media = 0.8797, Desviación Estándar = 0.0321
latency-weightedF1: Media = 0.6579, Desviación Estándar = 0.0285

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9815, Desviación Estándar = 0.0131
Macro_P        : Media = 0.9861, Desviación Estándar = 0.0098
Macro_R        : Media = 0.9167, Desviación Estándar = 0.0589
Macro_F1       : Media = 0.9372, Desviación Estándar = 0.0444
Micro_P        : Media = 0.9815, Desviación Estándar = 0.0131
Micro_R        : Media = 0.9815, Desviación Estándar = 0.0131
Micro_F1       : Media = 0.9815, Desviación Estándar = 0.0131


# PROBANDO AHORA A AÑADIR LA PLATAFORMA(twitch, telegram) bah mastante malos resultados  2 + (ronda_actual / 8)



=== ANÁLISIS FINAL DE LAS 3 EJECUCIONES ===

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.7222, Desviación Estándar = 0.0600
Macro_P        : Media = 0.7238, Desviación Estándar = 0.0610
Macro_R        : Media = 0.7222, Desviación Estándar = 0.0600
Macro_F1       : Media = 0.7218, Desviación Estándar = 0.0599
Micro_P        : Media = 0.7222, Desviación Estándar = 0.0600
Micro_R        : Media = 0.7222, Desviación Estándar = 0.0600
Micro_F1       : Media = 0.7222, Desviación Estándar = 0.0600
ERDE5          : Media = 0.5119, Desviación Estándar = 0.0176
ERDE30         : Media = 0.2948, Desviación Estándar = 0.0816
latencyTP      : Media = 17.0000, Desviación Estándar = 6.7206
speed          : Media = 0.8430, Desviación Estándar = 0.0645
latency-weightedF1: Media = 0.6206, Desviación Estándar = 0.0949

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9907, Desviación Estándar = 0.0131
Macro_P        : Media = 0.9722, Desviación Estándar = 0.0393
Macro_R        : Media = 0.9924, Desviación Estándar = 0.0107
Macro_F1       : Media = 0.9794, Desviación Estándar = 0.0292
Micro_P        : Media = 0.9907, Desviación Estándar = 0.0131
Micro_R        : Media = 0.9907, Desviación Estándar = 0.0131
Micro_F1       : Media = 0.9907, Desviación Estándar = 0.0131

# PRUEBA CON EL MODELO CON EMBEDDING DE 512 TOKENS MAS O MENOS -> REDUCIENDO A 2 + (ronda_actual // 20) Parece que añadir el tiempo/fecha le enseña a ser paciente "CURIOSO"
### Se puede observar que hacer los embeddings por bloque y luego sacar la media ayuda mucho al modelo a enterarse de un contexto mas amplio
=== ANÁLISIS FINAL DE LAS 3 EJECUCIONES ===

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.6389, Desviación Estándar = 0.0600
Macro_P        : Media = 0.6722, Desviación Estándar = 0.0538
Macro_R        : Media = 0.6389, Desviación Estándar = 0.0600
Macro_F1       : Media = 0.6132, Desviación Estándar = 0.0807
Micro_P        : Media = 0.6389, Desviación Estándar = 0.0600
Micro_R        : Media = 0.6389, Desviación Estándar = 0.0600
Micro_F1       : Media = 0.6389, Desviación Estándar = 0.0600
ERDE5          : Media = 0.5666, Desviación Estándar = 0.0227
ERDE30         : Media = 0.2624, Desviación Estándar = 0.0095
latencyTP      : Media = 10.3333, Desviación Estándar = 0.9428
speed          : Media = 0.9074, Desviación Estándar = 0.0093
latency-weightedF1: Media = 0.6425, Desviación Estándar = 0.0231

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9907, Desviación Estándar = 0.0131
Macro_P        : Media = 0.9931, Desviación Estándar = 0.0098
Macro_R        : Media = 0.9583, Desviación Estándar = 0.0589
Macro_F1       : Media = 0.9686, Desviación Estándar = 0.0444
Micro_P        : Media = 0.9907, Desviación Estándar = 0.0131
Micro_R        : Media = 0.9907, Desviación Estándar = 0.0131
Micro_F1       : Media = 0.9907, Desviación Estándar = 0.0131

Estadísticas guardadas en 'resultadosPredictor/estadisticas_ejecuciones.json'

# MULTICLASE UTILIZANDO MENSAJES ACUMULADOS Y CON VOTOS POR RONDA (SUMA) CON 0

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 1.0000, Desviación Estándar = 0.0000
Macro_P        : Media = 1.0000, Desviación Estándar = 0.0000
Macro_R        : Media = 1.0000, Desviación Estándar = 0.0000
Macro_F1       : Media = 1.0000, Desviación Estándar = 0.0000
Micro_P        : Media = 1.0000, Desviación Estándar = 0.0000
Micro_R        : Media = 1.0000, Desviación Estándar = 0.0000
Micro_F1       : Media = 1.0000, Desviación Estándar = 0.0000

#  MULTICLASE UTILIZANDO MENSAJES ACUMULANDO Y CON VOTOS POR RONDA (SUMA) CON 0 Y 1

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9815, Desviación Estándar = 0.0131
Macro_P        : Media = 0.9861, Desviación Estándar = 0.0098
Macro_R        : Media = 0.9167, Desviación Estándar = 0.0589
Macro_F1       : Media = 0.9372, Desviación Estándar = 0.0444
Micro_P        : Media = 0.9815, Desviación Estándar = 0.0131
Micro_R        : Media = 0.9815, Desviación Estándar = 0.0131
Micro_F1       : Media = 0.9815, Desviación Estándar = 0.0131


#  MULTICLASE UTILIZANDO MENSAJES SIN ACUMULAR Y CON VOTOS POR RONDA (SUMA) CON 0 (SE NOTA QUE SIN ACUMULAR)


ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.3611, Desviación Estándar = 0.0454
Macro_P        : Media = 0.5128, Desviación Estándar = 0.0091
Macro_R        : Media = 0.4903, Desviación Estándar = 0.0331
Macro_F1       : Media = 0.3200, Desviación Estándar = 0.0323
Micro_P        : Media = 0.3611, Desviación Estándar = 0.0454
Micro_R        : Media = 0.3611, Desviación Estándar = 0.0454
Micro_F1       : Media = 0.3611, Desviación Estándar = 0.0454


#  MULTICLASE UTILIZANDO MENSAJES SIN ACUMULAR Y CON VOTOS POR RONDA (SUMA) CON 0 Y 1 (SE NOTA QUE SIN ACUMULAR)

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.3981, Desviación Estándar = 0.0571
Macro_P        : Media = 0.4920, Desviación Estándar = 0.0060
Macro_R        : Media = 0.4459, Desviación Estándar = 0.0840
Macro_F1       : Media = 0.3427, Desviación Estándar = 0.0486
Micro_P        : Media = 0.3981, Desviación Estándar = 0.0571
Micro_R        : Media = 0.3981, Desviación Estándar = 0.0571
Micro_F1       : Media = 0.3981, Desviación Estándar = 0.0571




####################

# Entregado


# 1+0 2//20


ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.7130, Desviación Estándar = 0.0346
Macro_P        : Media = 0.7221, Desviación Estándar = 0.0344
Macro_R        : Media = 0.7130, Desviación Estándar = 0.0346
Macro_F1       : Media = 0.7098, Desviación Estándar = 0.0356
Micro_P        : Media = 0.7130, Desviación Estándar = 0.0346
Micro_R        : Media = 0.7130, Desviación Estándar = 0.0346
Micro_F1       : Media = 0.7130, Desviación Estándar = 0.0346
ERDE5          : Media = 0.5168, Desviación Estándar = 0.0241
ERDE30         : Media = 0.2693, Desviación Estándar = 0.0217
latencyTP      : Media = 14.1667, Desviación Estándar = 3.0092
speed          : Media = 0.8698, Desviación Estándar = 0.0295
latency-weightedF1: Media = 0.6439, Desviación Estándar = 0.0421

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 1.0000, Desviación Estándar = 0.0000
Macro_P        : Media = 1.0000, Desviación Estándar = 0.0000
Macro_R        : Media = 1.0000, Desviación Estándar = 0.0000
Macro_F1       : Media = 1.0000, Desviación Estándar = 0.0000
Micro_P        : Media = 1.0000, Desviación Estándar = 0.0000
Micro_R        : Media = 1.0000, Desviación Estándar = 0.0000
Micro_F1       : Media = 1.0000, Desviación Estándar = 0.0000
Cambiada un poco la atencion (igual un poco peor) pero cache se ha perdido


# 0 2//20

=== ANÁLISIS FINAL DE LAS 3 EJECUCIONES ===

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.7593, Desviación Estándar = 0.0131
Macro_P        : Media = 0.7646, Desviación Estándar = 0.0195
Macro_R        : Media = 0.7593, Desviación Estándar = 0.0131
Macro_F1       : Media = 0.7582, Desviación Estándar = 0.0119
Micro_P        : Media = 0.7593, Desviación Estándar = 0.0131
Micro_R        : Media = 0.7593, Desviación Estándar = 0.0131
Micro_F1       : Media = 0.7593, Desviación Estándar = 0.0131
ERDE5          : Media = 0.5133, Desviación Estándar = 0.0214
ERDE30         : Media = 0.3441, Desviación Estándar = 0.0272
latencyTP      : Media = 23.6667, Desviación Estándar = 8.6249
speed          : Media = 0.7797, Desviación Estándar = 0.0824
latency-weightedF1: Media = 0.6005, Desviación Estándar = 0.0574

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9907, Desviación Estándar = 0.0131
Macro_P        : Media = 0.9931, Desviación Estándar = 0.0098
Macro_R        : Media = 0.9583, Desviación Estándar = 0.0589
Macro_F1       : Media = 0.9686, Desviación Estándar = 0.0444
Micro_P        : Media = 0.9907, Desviación Estándar = 0.0131
Micro_R        : Media = 0.9907, Desviación Estándar = 0.0131
Micro_F1       : Media = 0.9907, Desviación Estándar = 0.0131


# 0 3/6

=== ANÁLISIS FINAL DE LAS 3 EJECUCIONES ===

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.7963, Desviación Estándar = 0.0131
Macro_P        : Media = 0.7995, Desviación Estándar = 0.0157
Macro_R        : Media = 0.7963, Desviación Estándar = 0.0131
Macro_F1       : Media = 0.7958, Desviación Estándar = 0.0127
Micro_P        : Media = 0.7963, Desviación Estándar = 0.0131
Micro_R        : Media = 0.7963, Desviación Estándar = 0.0131
Micro_F1       : Media = 0.7963, Desviación Estándar = 0.0131
ERDE5          : Media = 0.5127, Desviación Estándar = 0.0232
ERDE30         : Media = 0.3504, Desviación Estándar = 0.0466
latencyTP      : Media = 28.6667, Desviación Estándar = 10.5699
speed          : Media = 0.7342, Desviación Estándar = 0.0986
latency-weightedF1: Media = 0.5788, Desviación Estándar = 0.0766

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9907, Desviación Estándar = 0.0131
Macro_P        : Media = 0.9931, Desviación Estándar = 0.0098
Macro_R        : Media = 0.9583, Desviación Estándar = 0.0589
Macro_F1       : Media = 0.9686, Desviación Estándar = 0.0444
Micro_P        : Media = 0.9907, Desviación Estándar = 0.0131
Micro_R        : Media = 0.9907, Desviación Estándar = 0.0131
Micro_F1       : Media = 0.9907, Desviación Estándar = 0.0131


# 1 3/6
ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.6944, Desviación Estándar = 0.0786
Macro_P        : Media = 0.7025, Desviación Estándar = 0.0735
Macro_R        : Media = 0.6944, Desviación Estándar = 0.0786
Macro_F1       : Media = 0.6897, Desviación Estándar = 0.0818
Micro_P        : Media = 0.6944, Desviación Estándar = 0.0786
Micro_R        : Media = 0.6944, Desviación Estándar = 0.0786
Micro_F1       : Media = 0.6944, Desviación Estándar = 0.0786
ERDE5          : Media = 0.5591, Desviación Estándar = 0.0462
ERDE30         : Media = 0.2833, Desviación Estándar = 0.0613
latencyTP      : Media = 11.6667, Desviación Estándar = 1.2472
speed          : Media = 0.8943, Desviación Estándar = 0.0123
latency-weightedF1: Media = 0.6487, Desviación Estándar = 0.0528

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9815, Desviación Estándar = 0.0131
Macro_P        : Media = 0.9861, Desviación Estándar = 0.0098
Macro_R        : Media = 0.9167, Desviación Estándar = 0.0589
Macro_F1       : Media = 0.9372, Desviación Estándar = 0.0444
Micro_P        : Media = 0.9815, Desviación Estándar = 0.0131
Micro_R        : Media = 0.9815, Desviación Estándar = 0.0131
Micro_F1       : Media = 0.9815, Desviación Estándar = 0.0131

# 1 y 0 3/6

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.7130, Desviación Estándar = 0.0131
Macro_P        : Media = 0.7157, Desviación Estándar = 0.0112
Macro_R        : Media = 0.7130, Desviación Estándar = 0.0131
Macro_F1       : Media = 0.7120, Desviación Estándar = 0.0139
Micro_P        : Media = 0.7130, Desviación Estándar = 0.0131
Micro_R        : Media = 0.7130, Desviación Estándar = 0.0131
Micro_F1       : Media = 0.7130, Desviación Estándar = 0.0131
ERDE5          : Media = 0.5451, Desviación Estándar = 0.0138
ERDE30         : Media = 0.2971, Desviación Estándar = 0.0186
latencyTP      : Media = 16.1667, Desviación Estándar = 4.3653
speed          : Media = 0.8505, Desviación Estándar = 0.0423
latency-weightedF1: Media = 0.6174, Desviación Estándar = 0.0354

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 1.0000, Desviación Estándar = 0.0000
Macro_P        : Media = 1.0000, Desviación Estándar = 0.0000
Macro_R        : Media = 1.0000, Desviación Estándar = 0.0000
Macro_F1       : Media = 1.0000, Desviación Estándar = 0.0000
Micro_P        : Media = 1.0000, Desviación Estándar = 0.0000
Micro_R        : Media = 1.0000, Desviación Estándar = 0.0000
Micro_F1       : Media = 1.0000, Desviación Estándar = 0.0000



# con ninguno 3/6

=== ANÁLISIS FINAL DE LAS 3 EJECUCIONES ===

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.7778, Desviación Estándar = 0.0227
Macro_P        : Media = 0.7816, Desviación Estándar = 0.0202
Macro_R        : Media = 0.7778, Desviación Estándar = 0.0227
Macro_F1       : Media = 0.7769, Desviación Estándar = 0.0233
Micro_P        : Media = 0.7778, Desviación Estándar = 0.0227
Micro_R        : Media = 0.7778, Desviación Estándar = 0.0227
Micro_F1       : Media = 0.7778, Desviación Estándar = 0.0227
ERDE5          : Media = 0.5193, Desviación Estándar = 0.0139
ERDE30         : Media = 0.2220, Desviación Estándar = 0.0231
latencyTP      : Media = 11.3333, Desviación Estándar = 2.6247
speed          : Media = 0.8976, Desviación Estándar = 0.0258
latency-weightedF1: Media = 0.7086, Desviación Estándar = 0.0083

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9907, Desviación Estándar = 0.0131
Macro_P        : Media = 0.9931, Desviación Estándar = 0.0098
Macro_R        : Media = 0.9583, Desviación Estándar = 0.0589
Macro_F1       : Media = 0.9686, Desviación Estándar = 0.0444
Micro_P        : Media = 0.9907, Desviación Estándar = 0.0131
Micro_R        : Media = 0.9907, Desviación Estándar = 0.0131
Micro_F1       : Media = 0.9907, Desviación Estándar = 0.0131



##### Dividiendo bloques en 6000 caracteres
=== ANÁLISIS FINAL DE LAS 3 EJECUCIONES ===

ESTADÍSTICAS DE MÉTRICAS BINARIAS:
Accuracy       : Media = 0.7963, Desviación Estándar = 0.0262
Macro_P        : Media = 0.7986, Desviación Estándar = 0.0246
Macro_R        : Media = 0.7963, Desviación Estándar = 0.0262
Macro_F1       : Media = 0.7958, Desviación Estándar = 0.0265
Micro_P        : Media = 0.7963, Desviación Estándar = 0.0262
Micro_R        : Media = 0.7963, Desviación Estándar = 0.0262
Micro_F1       : Media = 0.7963, Desviación Estándar = 0.0262
ERDE5          : Media = 0.5050, Desviación Estándar = 0.0160
ERDE30         : Media = 0.2276, Desviación Estándar = 0.0460
latencyTP      : Media = 15.0000, Desviación Estándar = 3.7417
speed          : Media = 0.8618, Desviación Estándar = 0.0366
latency-weightedF1: Media = 0.6866, Desviación Estándar = 0.0511

ESTADÍSTICAS DE MÉTRICAS MULTICLASE:
Accuracy       : Media = 0.9815, Desviación Estándar = 0.0131
Macro_P        : Media = 0.9861, Desviación Estándar = 0.0098
Macro_R        : Media = 0.9167, Desviación Estándar = 0.0589
Macro_F1       : Media = 0.9372, Desviación Estándar = 0.0444
Micro_P        : Media = 0.9815, Desviación Estándar = 0.0131
Micro_R        : Media = 0.9815, Desviación Estándar = 0.0131
Micro_F1       : Media = 0.9815, Desviación Estándar = 0.0131