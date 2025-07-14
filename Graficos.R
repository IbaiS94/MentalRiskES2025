library(ggplot2)
library(gridExtra)
library(reshape2)
library(dplyr)
library(scales)

if (!dir.exists("./graficos")) {
  dir.create("./graficos", recursive = TRUE)
}

tabla1 <- data.frame(
  Riesgo = c("Bajo riesgo", "Alto riesgo"),
  Train = c(182, 175),
  Trial = c(4, 3)
)

tabla2 <- data.frame(
  Riesgo = rep(c("Bajo riesgo", "Alto riesgo"), each = 4),
  Categoria = rep(c("Betting", "Online Gaming", "Trading", "Lootboxes"), 2),
  Casos = c(41, 51, 76, 14, 46, 55, 61, 13)
)

tabla4 <- data.frame(
  Riesgo = c("Bajo riesgo", "Alto riesgo"),
  Twitch = c(65, 68),
  Telegram = c(117, 107)
)

stats_msg <- data.frame(
  Tipo = c("Mensajes individuales", "Por usuario"),
  Media = c(9.58, 615.53),
  Mediana = c(6.00, 592.00),
  Desviacion = c(12.85, 477.74)
)

datos_rendimiento <- data.frame(
  Grupos = 1:9,
  Macro_F1_Media = c(0.7573, 0.7586, 0.7392, 0.7960, 0.7958, 0.7868, 0.7955, 0.7677, 0.7765),
  Macro_F1_SD = c(0.0354, 0.0347, 0.0352, 0.0133, 0.0265, 0.0132, 0.0352, 0.0138, 0.0228),
  ERDE30_Media = c(0.2225, 0.2186, 0.2524, 0.2384, 0.2276, 0.2336, 0.2090, 0.2233, 0.2474),
  ERDE30_SD = c(0.0476, 0.0175, 0.0304, 0.0282, 0.0460, 0.0311, 0.0227, 0.0105, 0.0413)
)

colores_riesgo <- c("Bajo riesgo" = "#2E8B57", "Alto riesgo" = "#DC143C")
colores_conjunto <- c("Train" = "#4682B4", "Trial" = "#FF6347")
colores_actividad <- c("Trading" = "#FF6B6B", "Online Gaming" = "#4ECDC4", 
                      "Betting" = "#45B7D1", "Lootboxes" = "#96CEB4")

tabla1_long <- melt(tabla1, id.vars = "Riesgo")
p1 <- ggplot(tabla1_long, aes(x = Riesgo, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8, width = 0.7) +
  geom_text(aes(label = value), position = position_dodge(width = 0.7), 
            vjust = -0.5, size = 4, fontface = "bold") +
  labs(title = "Distribución de Datos: Entrenamiento vs Evaluación",
       subtitle = "Comparación entre conjuntos de datos por nivel de riesgo",
       x = "Nivel de Riesgo", y = "Número de casos", fill = "Conjunto de datos") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray50"),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 11),
        legend.title = element_text(size = 11, face = "bold"),
        legend.position = "bottom") +
  scale_fill_manual(values = colores_conjunto) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)))

p2 <- ggplot(tabla2, aes(x = Categoria, y = Riesgo, fill = Casos)) +
  geom_tile(color = "white", size = 1) +
  geom_text(aes(label = Casos), color = "white", size = 5, fontface = "bold") +
  scale_fill_gradient(low = "#87CEEB", high = "#1E3A8A", name = "Casos") +
  labs(title = "Mapa de Calor: Distribución por Tipo de Actividad",
       subtitle = "Intensidad de casos según actividad y nivel de riesgo",
       x = "Tipo de Actividad", y = "Nivel de Riesgo") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray50"),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 11),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.title = element_text(size = 11, face = "bold"))

tabla4_long <- melt(tabla4, id.vars = "Riesgo")
p3 <- ggplot(tabla4_long, aes(x = variable, y = value, fill = Riesgo)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8, width = 0.7) +
  geom_text(aes(label = value), position = position_dodge(width = 0.7), 
            vjust = -0.5, size = 4, fontface = "bold") +
  labs(title = "Distribución de Usuarios por Plataforma",
       subtitle = "Comparación entre Twitch y Telegram según nivel de riesgo",
       x = "Plataforma", y = "Número de usuarios", fill = "Nivel de Riesgo") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray50"),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 11),
        legend.title = element_text(size = 11, face = "bold"),
        legend.position = "bottom") +
  scale_fill_manual(values = colores_riesgo) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)))

riesgo_total <- data.frame(
  Nivel = c("Bajo riesgo", "Alto riesgo"),
  Casos = c(182, 175)
)

p4 <- ggplot(riesgo_total, aes(x = "", y = Casos, fill = Nivel)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  geom_text(aes(label = paste0(Casos, "\n(", round(Casos/sum(Casos)*100, 1), "%)")), 
            position = position_stack(vjust = 0.5), 
            size = 5, fontface = "bold", color = "white") +
  labs(title = "Distribución General del Nivel de Riesgo",
       subtitle = paste("Total de casos analizados:", sum(riesgo_total$Casos)),
       fill = "Nivel de Riesgo") +
  theme_void() +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray50"),
        legend.title = element_text(size = 11, face = "bold"),
        legend.position = "bottom") +
  scale_fill_manual(values = colores_riesgo)

stats_long <- melt(stats_msg, id.vars = "Tipo", measure.vars = c("Media", "Mediana"))
p5 <- ggplot(stats_long, aes(x = variable, y = value, fill = Tipo)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8, width = 0.7) +
  geom_text(aes(label = round(value, 1)), position = position_dodge(width = 0.7), 
            vjust = -0.5, size = 4, fontface = "bold") +
  scale_y_log10(labels = scales::comma) +
  labs(title = "Estadísticas de Longitud de Mensajes",
       subtitle = "Comparación de medias y medianas (escala logarítmica)",
       x = "Métrica Estadística", y = "Tokens (escala log)", fill = "Tipo de medición") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray50"),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 11),
        legend.title = element_text(size = 11, face = "bold"),
        legend.position = "bottom") +
  scale_fill_manual(values = c("Mensajes individuales" = "#FF7F50", "Por usuario" = "#4169E1"))

tabla2_prop <- tabla2 %>%
  group_by(Categoria) %>%
  mutate(Total = sum(Casos), Proporcion = Casos / Total) %>%
  filter(Riesgo == "Alto riesgo") %>%
  arrange(desc(Proporcion))

p6 <- ggplot(tabla2_prop, aes(x = reorder(Categoria, Proporcion), y = Proporcion, fill = Categoria)) +
  geom_bar(stat = "identity", alpha = 0.8, width = 0.7) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "black", alpha = 0.7, size = 1) +
  geom_text(aes(label = paste0(round(Proporcion*100, 1), "%")), 
            hjust = -0.1, size = 4, fontface = "bold") +
  labs(title = "Proporción de Alto Riesgo por Tipo de Actividad",
       subtitle = "Línea punteada indica el 50% de referencia",
       x = "Tipo de Actividad", y = "Proporción de Alto Riesgo") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray50"),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 11),
        legend.position = "none") +
  scale_y_continuous(labels = percent_format(), expand = expansion(mult = c(0, 0.1))) +
  scale_fill_manual(values = colores_actividad) +
  coord_flip()

totales_categoria <- tabla2 %>%
  group_by(Categoria) %>%
  summarise(Total = sum(Casos), .groups = 'drop') %>%
  arrange(desc(Total))

totales_categoria$Porcentaje <- round(totales_categoria$Total / sum(totales_categoria$Total) * 100, 1)
totales_categoria$Categoria <- factor(totales_categoria$Categoria, levels = totales_categoria$Categoria)

p7 <- ggplot(totales_categoria, aes(x = "", y = Total, fill = Categoria)) +
  geom_bar(stat = "identity", width = 1, color = "white", size = 2) +
  coord_polar("y", start = 0) +
  geom_text(aes(label = paste0(Categoria, "\n", Porcentaje, "%")), 
            position = position_stack(vjust = 0.5), 
            size = 10, fontface = "bold", color = "white") +
  labs(title = "Distribución por Categorías",
       subtitle = paste("Total de Usuarios:", sum(totales_categoria$Total))) +
  theme_void() +
  theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5, margin = margin(b = 10)),
        plot.subtitle = element_text(size = 14, hjust = 0.5, color = "gray50", margin = margin(b = 20)),
        legend.position = "none",
        plot.background = element_rect(fill = "white", color = NA)) +
  scale_fill_manual(values = colores_actividad)

p8 <- ggplot(datos_rendimiento, aes(x = Grupos, y = Macro_F1_Media)) +
  geom_line(color = "#2E86AB", size = 1.2) +
  geom_point(color = "#2E86AB", size = 3) +
  geom_errorbar(aes(ymin = Macro_F1_Media - Macro_F1_SD, ymax = Macro_F1_Media + Macro_F1_SD),
                width = 0.2, color = "#2E86AB", alpha = 0.7) +
  labs(title = "Macro F1 por Número de Grupos",
       subtitle = "Media ± Desviación Estándar",
       x = "Número de Grupos", y = "Macro F1") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 12),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10),
        panel.grid.minor = element_blank()) +
  scale_x_continuous(breaks = 1:9) +
  scale_y_continuous(limits = c(0.7, 0.85), labels = scales::number_format(accuracy = 0.001))

p9 <- ggplot(datos_rendimiento, aes(x = Grupos, y = ERDE30_Media)) +
  geom_line(color = "#A23B72", size = 1.2) +
  geom_point(color = "#A23B72", size = 3) +
  geom_errorbar(aes(ymin = ERDE30_Media - ERDE30_SD, ymax = ERDE30_Media + ERDE30_SD),
                width = 0.2, color = "#A23B72", alpha = 0.7) +
  labs(title = "ERDE30 por Número de Grupos",
       subtitle = "Media ± Desviación Estándar",
       x = "Número de Grupos", y = "ERDE30") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 12),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10),
        panel.grid.minor = element_blank()) +
  scale_x_continuous(breaks = 1:9) +
  scale_y_continuous(limits = c(0.15, 0.3), labels = scales::number_format(accuracy = 0.001))

ggsave("./graficos/01_train_vs_trial.png", p1, width = 10, height = 8, dpi = 300)
ggsave("./graficos/02_heatmap_actividades.png", p2, width = 12, height = 8, dpi = 300)
ggsave("./graficos/03_distribucion_plataformas.png", p3, width = 10, height = 8, dpi = 300)
ggsave("./graficos/04_distribucion_general.png", p4, width = 10, height = 8, dpi = 300)
ggsave("./graficos/05_estadisticas_mensajes.png", p5, width = 10, height = 8, dpi = 300)
ggsave("./graficos/06_proporcion_alto_riesgo.png", p6, width = 10, height = 8, dpi = 300)
ggsave("./graficos/07_distribucion_categorias.png", p7, width = 12, height = 10, dpi = 300)
ggsave("./graficos/08_macro_f1_grupos.png", p8, width = 10, height = 6, dpi = 300)
ggsave("./graficos/09_erde30_grupos.png", p9, width = 10, height = 6, dpi = 300)

panel_combinado <- grid.arrange(p1, p2, p3, p4, ncol = 2, nrow = 2)
ggsave("./graficos/00_panel_combinado.png", panel_combinado, width = 16, height = 12, dpi = 300)

panel_rendimiento <- grid.arrange(p8, p9, ncol = 1)
ggsave("./graficos/10_panel_rendimiento.png", panel_rendimiento, width = 10, height = 12, dpi = 300)

total_casos <- sum(riesgo_total$Casos)
prop_alto_riesgo <- riesgo_total$Casos[2] / total_casos * 100

cat("RESUMEN ESTADÍSTICO\n")
cat("=" * 50, "\n")
cat("1. DISTRIBUCIÓN GENERAL:\n")
cat(sprintf("   - Total de casos: %d\n", total_casos))
cat(sprintf("   - Bajo riesgo: %d casos (%.1f%%)\n", riesgo_total$Casos[1], 100-prop_alto_riesgo))
cat(sprintf("   - Alto riesgo: %d casos (%.1f%%)\n", riesgo_total$Casos[2], prop_alto_riesgo))

tabla2_summary <- tabla2 %>%
  group_by(Categoria) %>%
  summarise(Total = sum(Casos), Alto_Riesgo = sum(Casos[Riesgo == "Alto riesgo"]),
            Proporcion_Alto = Alto_Riesgo / Total, .groups = 'drop') %>%
  arrange(desc(Total))

cat("\n2. ANÁLISIS POR ACTIVIDAD:\n")
for(i in 1:nrow(tabla2_summary)) {
  cat(sprintf("   - %s: %d casos (%.1f%% alto riesgo)\n", 
              tabla2_summary$Categoria[i], tabla2_summary$Total[i], 
              tabla2_summary$Proporcion_Alto[i] * 100))
}

actividad_mas_riesgosa <- tabla2_summary %>% 
  filter(Proporcion_Alto == max(Proporcion_Alto)) %>% 
  pull(Categoria)

cat(sprintf("\n3. ACTIVIDAD DE MAYOR RIESGO: %s\n", actividad_mas_riesgosa))

total_twitch <- sum(tabla4$Twitch)
total_telegram <- sum(tabla4$Telegram)
prop_alto_twitch <- tabla4$Twitch[2] / total_twitch * 100
prop_alto_telegram <- tabla4$Telegram[2] / total_telegram * 100

cat("\n4. ANÁLISIS POR PLATAFORMA:\n")
cat(sprintf("   - Twitch: %d usuarios (%.1f%% alto riesgo)\n", total_twitch, prop_alto_twitch))
cat(sprintf("   - Telegram: %d usuarios (%.1f%% alto riesgo)\n", total_telegram, prop_alto_telegram))

chi_test_data <- matrix(c(41, 46, 51, 55, 76, 61, 14, 13), nrow = 2, byrow = TRUE)
chi_result <- chisq.test(chi_test_data)

cat("\n5. ANÁLISIS ESTADÍSTICO:\n")
cat(sprintf("   - Chi-cuadrado: %.3f (p = %.4f)\n", chi_result$statistic, chi_result$p.value))
cat(sprintf("   - Significancia: %s\n", ifelse(chi_result$p.value < 0.05, "Significativo", "No significativo")))

cat("\n6. RENDIMIENTO DEL MODELO:\n")
cat(sprintf("   - Mejor Macro F1: %.4f (%d grupos)\n", max(datos_rendimiento$Macro_F1_Media), 
            datos_rendimiento$Grupos[which.max(datos_rendimiento$Macro_F1_Media)]))
cat(sprintf("   - Mejor ERDE30: %.4f (%d grupos)\n", min(datos_rendimiento$ERDE30_Media), 
            datos_rendimiento$Grupos[which.min(datos_rendimiento$ERDE30_Media)]))

cat("\n7. ESTADÍSTICAS DE MENSAJES:\n")
cat(sprintf("   - Media por mensaje: %.2f tokens\n", stats_msg$Media[1]))
cat(sprintf("   - Media por usuario: %.2f tokens\n", stats_msg$Media[2]))
cat(sprintf("   - Variabilidad: CV = %.1f%%\n", stats_msg$Desviacion[2]/stats_msg$Media[2]*100))



