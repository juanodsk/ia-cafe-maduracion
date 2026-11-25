# Corrección del piloto GAN

El 25 de noviembre de 2026 se detectó que el discriminador usaba BatchNorm en modo de entrenamiento al discriminar imágenes reales y falsas, pero cambiaba a modo de evaluación durante el paso del generador. Esto permitía optimizar al generador contra una función distinta y producía pérdidas pequeñas engañosas en ambas redes.

Se corrigió el paso del generador para mantener el discriminador en modo de entrenamiento, con sus parámetros congelados durante ese paso, conservando el uso de estadísticas de lote. Se añadió una prueba de regresión que exige los modos correctos en las tres evaluaciones del discriminador de cada actualización alternada.

El piloto previo de dos épocas se conserva para auditoría en `experimentos/transfer_01/gan_bn_mismatch_pilot/`. No se utiliza para generar los sintéticos de la ablación ni para las métricas del informe. El experimento corregido se reinició desde cero en `experimentos/transfer_01/gan/`, con una clave explícita de configuración que registra el modo del discriminador.

La corrección no modifica los pesos ni las métricas de los cinco clasificadores preentrenados.
