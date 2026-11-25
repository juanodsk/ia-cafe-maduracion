# Probar la madurez del café con fotografías

Coloca aquí una fotografía cercana de un fruto, con iluminación y colores naturales. Si aparecen varios frutos, recorta uno. El modelo clasifica el recorte; no detecta automáticamente los frutos de una rama.

En PowerShell, desde la carpeta del proyecto:

```powershell
.\.venv\Scripts\python.exe -m cafe_ml.predict --imagen ".\pruebas\mi_cafe.jpg" --modelo ensemble
```

El programa imprime la clase y guarda `resultado.png` y `prediccion.json` dentro de `pruebas/resultados/`. La puntuación del modelo no equivale al porcentaje de aciertos garantizado para esa fotografía.

Para registrar una etiqueta conocida de antemano y comprobar si coincide:

```powershell
.\.venv\Scripts\python.exe -m cafe_ml.predict --imagen ".\pruebas\mi_cafe.jpg" --modelo ensemble --esperado ripe
```

Las etiquetas son `unripe` (verde), `semi_ripe` (semimaduro o pintón), `ripe` (maduro), `overripe` (sobremaduro) y `dry` (seco).

Puedes elegir `--modelo resnet50`, `efficientnet_b0`, `densenet121`, `vit_b_16` o `swin_t`. Sin esa opción se utiliza EfficientNet B0 afinado. `--device cpu` permite ejecutar sin GPU. `--bbox X Y ANCHO ALTO` selecciona un fruto por una caja manual, en píxeles después de corregir la orientación EXIF.

## Demostración incluida

`ejemplo_maduro_validacion.png` procede de la validación existente: recorte `valid:4:42`, etiqueta `ripe`. Comprueba que el programa funciona; no constituye una evaluación externa. No se abrió el conjunto de prueba reservado.

```powershell
.\.venv\Scripts\python.exe -m cafe_ml.predict --imagen pruebas\ejemplo_maduro_validacion.png --modelo ensemble --esperado ripe
```

Para evaluar tus condiciones de uso, prueba varias fotos nuevas de todas las clases y etiqueta su madurez antes de mirar las predicciones. Registra aciertos y errores. Una imagen acertada no demuestra el rendimiento general; conserva esas fotografías fuera del entrenamiento durante la evaluación.
