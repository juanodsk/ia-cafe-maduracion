# Dataset preparado para madurez de café

Partición reproducible 70 % / 15 % / 15 %, semilla 42. Los conteos enteros son 1651 / 354 / 354 imágenes.

Cada carpeta train, valid y test contiene imágenes y _annotations.coco.json. Los CSV de csv/ fijan la asignación, las distribuciones, los grupos y la trazabilidad por objeto.

La descarga original se conserva en imagenes/. Se excluyó una copia exacta del conjunto preparado; las imágenes relacionadas permanecen en el mismo conjunto.

Los recortes para CNN y Transformers deben heredar el split de su imagen. No redividir objetos individualmente.

La versión v5 ya contiene aumentos. Esta organización conserva esas variantes dentro de sus grupos y no identifica originales sin aumentos para validación/prueba. Ver preparacion.json y docs/particion_dataset.md.
