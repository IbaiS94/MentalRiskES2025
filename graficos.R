library(ggplot2)
library(gridExtra)
library(reshape2)
library(dplyr)
library(scales)

if (!dir.exists("./graficos")) {
  dir.create("./graficos", recursive = TRUE)
}

tabla_entrenamiento <- data.frame(
  ludopatia = c("Baja ludopatía", "Alta ludopatía"),
  entrenamiento = c(77, 83),
  prueba = c(77, 83)
)

tabla_categorias <- data.frame(
  ludopatia = rep(c("Baja ludopatía", "Alta ludopatía"), each = 4),
  categoria = rep(c("Betting", "Online Gaming", "Trading", "Lootboxes"), 2),
  casos = c(41, 51, 76, 14, 46, 55, 61, 13)
)

estadisticas_mensajes <- data.frame(
  tipo = c("Mensajes individuales", "Por usuario"),
  media = c(9.48, 557.49),
  mediana = c(6.00, 527.50),
  desviacion = c(11.35, 467.44)
)

colores_ludopatia <- c("Baja ludopatía" = "#2E8B57", "Alta ludopatía" = "#DC143C")
colores_conjunto <- c("entrenamiento" = "#4682B4", "prueba" = "#FF6347")

tabla_larga <- melt(tabla_entrenamiento, id.vars = "ludopatia")
grafico1 <- ggplot(tabla_larga, aes(x = ludopatia, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8, width = 0.7) +
  geom_text(aes(label = value), position = position_dodge(width = 0.7), 
            vjust = -0.5, size = 4, fontface = "bold") +
  labs(title = "Distribución de Datos: Entrenamiento vs Prueba",
       subtitle = "Comparación entre conjuntos de datos por nivel de ludopatía",
       x = "Nivel de Ludopatía",
       y = "Número de casos",
       fill = "Conjunto de datos") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray50"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 11),
    legend.title = element_text(size = 11, face = "bold"),
    legend.position = "bottom"
  ) +
  scale_fill_manual(values = colores_conjunto) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)))

ggsave("./graficos/01_entrenamiento_vs_prueba.png", grafico1, width = 10, height = 8, dpi = 300)

mapa_calor <- ggplot(tabla_categorias, aes(x = categoria, y = ludopatia, fill = casos)) +
  geom_tile(color = "white", size = 1) +
  geom_text(aes(label = casos), color = "white", size = 5, fontface = "bold") +
  scale_fill_gradient(low = "#87CEEB", high = "#1E3A8A", name = "Casos") +
  labs(title = "Mapa de Calor: Distribución por Tipo de Actividad",
       subtitle = "Intensidad de casos según actividad y nivel de ludopatía",
       x = "Tipo de Actividad",
       y = "Nivel de Ludopatía") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray50"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 11),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.title = element_text(size = 11, face = "bold")
  )

ggsave("./graficos/02_mapa_calor_actividades.png", mapa_calor, width = 12, height = 8, dpi = 300)

total_ludopatia <- data.frame(
  nivel = c("Baja ludopatía", "Alta ludopatía"),
  casos = c(182, 175)
)

grafico_circular <- ggplot(total_ludopatia, aes(x = "", y = casos, fill = nivel)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  geom_text(aes(label = paste0(casos, "\n(", round(casos/sum(casos)*100, 1), "%)")), 
            position = position_stack(vjust = 0.5), 
            size = 5, fontface = "bold", color = "white") +
  labs(title = "Distribución General del Nivel de Ludopatía",
       subtitle = paste("Total de casos analizados:", sum(total_ludopatia$casos)),
       fill = "Nivel de Ludopatía") +
  theme_void() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray50"),
    legend.title = element_text(size = 11, face = "bold"),
    legend.position = "bottom"
  ) +
  scale_fill_manual(values = colores_ludopatia)

ggsave("./graficos/03_distribucion_general.png", grafico_circular, width = 10, height = 8, dpi = 300)

estadisticas_longo <- melt(estadisticas_mensajes, id.vars = "tipo", measure.vars = c("media", "mediana"))
grafico_estadisticas <- ggplot(estadisticas_longo, aes(x = variable, y = value, fill = tipo)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8, width = 0.7) +
  geom_text(aes(label = round(value, 1)), position = position_dodge(width = 0.7), 
            vjust = -0.5, size = 4, fontface = "bold") +
  scale_y_log10(labels = scales::comma) +
  labs(title = "Estadísticas de Longitud de Mensajes",
       subtitle = "Comparación de medias y medianas (escala logarítmica)",
       x = "Métrica Estadística",
       y = "Tokens (escala log)",
       fill = "Tipo de medición") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray50"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 11),
    legend.title = element_text(size = 11, face = "bold"),
    legend.position = "bottom"
  ) +
  scale_fill_manual(values = c("Mensajes individuales" = "#FF7F50", "Por usuario" = "#4169E1"))

ggsave("./graficos/04_estadisticas_mensajes.png", grafico_estadisticas, width = 10, height = 8, dpi = 300)

proporciones <- tabla_categorias %>%
  group_by(categoria) %>%
  mutate(total = sum(casos),
         proporcion = casos / total) %>%
  filter(ludopatia == "Alta ludopatía") %>%
  arrange(desc(proporcion))

colores_actividades <- c(
  "Trading" = "#FF6B6B",        
  "Online Gaming" = "#4ECDC4",  
  "Betting" = "#45B7D1",        
  "Lootboxes" = "#96CEB4"       
)

grafico_proporciones <- ggplot(proporciones, aes(x = reorder(categoria, proporcion), y = proporcion, fill = categoria)) +
  geom_bar(stat = "identity", alpha = 0.8, width = 0.7) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "black", 
             alpha = 0.7, size = 1) +
  geom_text(aes(label = paste0(round(proporcion * 100, 1), "%")), 
            hjust = -0.1, size = 4, fontface = "bold") +
  labs(title = "Proporción de Alta Ludopatía por Tipo de Actividad",
       subtitle = "Línea punteada indica el 50% de referencia",
       x = "Tipo de Actividad",
       y = "Proporción de Alta Ludopatía") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray50"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 11),
    legend.position = "none"  
  ) +
  scale_y_continuous(labels = percent_format(), expand = expansion(mult = c(0, 0.1))) +
  scale_fill_manual(values = colores_actividades) +
  coord_flip()

ggsave("./graficos/05_proporcion_alta_ludopatia.png", grafico_proporciones, width = 10, height = 8, dpi = 300)

grafico_barras <- ggplot(tabla_categorias, aes(x = categoria, y = casos, fill = ludopatia)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8, width = 0.7) +
  geom_text(aes(label = casos), position = position_dodge(width = 0.7), 
            vjust = -0.5, size = 4, fontface = "bold") +
  labs(title = "Distribución de Casos por Tipo de Actividad",
       subtitle = "Comparación entre niveles de ludopatía",
       x = "Tipo de Actividad",
       y = "Número de casos",
       fill = "Nivel de Ludopatía") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray50"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 11),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.title = element_text(size = 11, face = "bold"),
    legend.position = "bottom"
  ) +
  scale_fill_manual(values = colores_ludopatia) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)))

ggsave("./graficos/06_distribucion_categorias.png", grafico_barras, width = 10, height = 8, dpi = 300)

panel_principal <- grid.arrange(grafico1, mapa_calor, grafico_circular, grafico_barras, ncol = 2, nrow = 2)
ggsave("./graficos/00_panel_combinado.png", panel_principal, width = 16, height = 12, dpi = 300)

totales_categoria <- tabla_categorias %>%
  group_by(categoria) %>%
  summarise(total = sum(casos), .groups = 'drop') %>%
  arrange(desc(total))

totales_categoria$porcentaje <- round(totales_categoria$total / sum(totales_categoria$total) * 100, 1)
totales_categoria$categoria <- factor(totales_categoria$categoria, 
                                    levels = totales_categoria$categoria)

grafico_circular_categorias <- ggplot(totales_categoria, aes(x = "", y = total, fill = categoria)) +
  geom_bar(stat = "identity", width = 1, color = "white", size = 2) +
  coord_polar("y", start = 0) +
  geom_text(aes(label = paste0(categoria, "\n", porcentaje, "%")), 
            position = position_stack(vjust = 0.5), 
            size = 10, fontface = "bold", color = "white",
            family = "sans") +
  labs(title = "Distribución por Categorías",
       subtitle = paste("Total de Usuarios:", sum(totales_categoria$total))) +
  theme_void() +
  theme(
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5, margin = margin(b = 10)),
    plot.subtitle = element_text(size = 14, hjust = 0.5, color = "gray50",
                                margin = margin(b = 20)),
    legend.position = "none",
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA)
  ) +
  scale_fill_manual(values = colores_actividades)

ggsave("./graficos/07_distribucion_circular_categorias.png", grafico_circular_categorias, 
       width = 12, height = 10, dpi = 300, bg = "white")

total_casos <- sum(total_ludopatia$casos)
prop_alta_ludopatia <- total_ludopatia$casos[2] / total_casos * 100

cat(sprintf("Total de casos: %d\n", total_casos))
cat(sprintf("Baja ludopatía: %d casos (%.1f%%)\n", total_ludopatia$casos[1], 100-prop_alta_ludopatia))
cat(sprintf("Alta ludopatía: %d casos (%.1f%%)\n", total_ludopatia$casos[2], prop_alta_ludopatia))

resumen_tabla <- tabla_categorias %>%
  group_by(categoria) %>%
  summarise(
    total = sum(casos),
    alta_ludopatia = sum(casos[ludopatia == "Alta ludopatía"]),
    proporcion_alta = alta_ludopatia / total,
    .groups = 'drop'
  ) %>%
  arrange(desc(total))

for(i in 1:nrow(resumen_tabla)) {
  actividad <- resumen_tabla$categoria[i]
  total <- resumen_tabla$total[i]
  prop <- resumen_tabla$proporcion_alta[i] * 100
  cat(sprintf("%s: %d casos total (%.1f%% alta ludopatía)\n", 
              actividad, total, prop))
}

actividad_mas_riesgosa <- resumen_tabla %>% 
  filter(proporcion_alta == max(proporcion_alta)) %>% 
  pull(categoria)
prop_max <- max(resumen_tabla$proporcion_alta) * 100

cat(sprintf("Actividad de mayor riesgo: %s (%.1f%%)\n", 
            actividad_mas_riesgosa, prop_max))

matriz_chi <- matrix(c(41, 46, 51, 55, 76, 61, 14, 13), 
                    nrow = 2, byrow = TRUE)
colnames(matriz_chi) <- c("Betting", "Online Gaming", "Trading", "Lootboxes")
rownames(matriz_chi) <- c("Baja ludopatía", "Alta ludopatía")

resultado_chi <- chisq.test(matriz_chi)
cat(sprintf("Chi-cuadrado: %.3f\n", resultado_chi$statistic))
cat(sprintf("p-valor: %.4f\n", resultado_chi$p.value))

cat(sprintf("Media por mensaje individual: %.2f tokens\n", estadisticas_mensajes$media[1]))
cat(sprintf("Media por usuario: %.2f tokens\n", estadisticas_mensajes$media[2]))
cat(sprintf("Coeficiente de variación: %.1f%%\n", 
            estadisticas_mensajes$desviacion[2]/estadisticas_mensajes$media[2]*100))

datos_rendimiento <- data.frame(
  grupos = 1:9,
  macro_f1_media = c(0.7573, 0.7586, 0.7392, 0.7960, 0.7958, 0.7868, 0.7955, 0.7677, 0.7765),
  macro_f1_sd = c(0.0354, 0.0347, 0.0352, 0.0133, 0.0265, 0.0132, 0.0352, 0.0138, 0.0228),
  erde30_media = c(0.2225, 0.2186, 0.2524, 0.2384, 0.2276, 0.2336, 0.2090, 0.2233, 0.2474),
  erde30_sd = c(0.0476, 0.0175, 0.0304, 0.0282, 0.0460, 0.0311, 0.0227, 0.0105, 0.0413)
)

grafico_macro_f1 <- ggplot(datos_rendimiento, aes(x = grupos, y = macro_f1_media)) +
  geom_line(color = "#2E86AB", size = 1.2) +
  geom_point(color = "#2E86AB", size = 3) +
  geom_errorbar(aes(ymin = macro_f1_media - macro_f1_sd, 
                    ymax = macro_f1_media + macro_f1_sd),
                width = 0.2, color = "#2E86AB", alpha = 0.7) +
  labs(title = "Macro F1 por Número de Grupos",
       subtitle = "Media ± Desviación Estándar",
       x = "Número de Grupos",
       y = "Macro F1") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 12),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10),
        panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = "white", color = NA)) +
  scale_x_continuous(breaks = 1:9) +
  scale_y_continuous(limits = c(0.7, 0.85), labels = scales::number_format(accuracy = 0.001))

grafico_erde30 <- ggplot(datos_rendimiento, aes(x = grupos, y = erde30_media)) +
  geom_line(color = "#A23B72", size = 1.2) +
  geom_point(color = "#A23B72", size = 3) +
  geom_errorbar(aes(ymin = erde30_media - erde30_sd, 
                    ymax = erde30_media + erde30_sd),
                width = 0.2, color = "#A23B72", alpha = 0.7) +
  labs(title = "ERDE30 por Número de Grupos",
       subtitle = "Media ± Desviación Estándar",
       x = "Número de Grupos",
       y = "ERDE30") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 12),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10),
        panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = "white", color = NA)) +
  scale_x_continuous(breaks = 1:9) +
  scale_y_continuous(limits = c(0.15, 0.3), labels = scales::number_format(accuracy = 0.001))

grid.arrange(grafico_macro_f1, grafico_erde30, ncol = 1)

ggsave("./graficos/macro_f1_grupos.png", grafico_macro_f1, width = 10, height = 6, dpi = 300)
ggsave("./graficos/erde30_grupos.png", grafico_erde30, width = 10, height = 6, dpi = 300)

cat("Mejor Macro F1:", max(datos_rendimiento$macro_f1_media), "en", datos_rendimiento$grupos[which.max(datos_rendimiento$macro_f1_media)], "grupos\n")
cat("Mejor ERDE30:", min(datos_rendimiento$erde30_media), "en", datos_rendimiento$grupos[which.min(datos_rendimiento$erde30_media)], "grupos\n")