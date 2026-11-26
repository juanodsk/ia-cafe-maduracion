"""Build the technical report from recorded experiment artifacts using python-docx."""
from __future__ import annotations
import csv,json
from pathlib import Path
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/documentacion_tecnica_cafe.docx'
REPO='https://github.com/juanodsk/ia-cafe-maduracion'
NAMES=['resnet50','efficientnet_b0','densenet121','vit_b_16','swin_t']
DISPLAY={'resnet50':'ResNet50','efficientnet_b0':'EfficientNet B0','densenet121':'DenseNet121','vit_b_16':'ViT B 16','swin_t':'Swin T'}
LABELS={'dry':'Seco','overripe':'Sobremaduro','ripe':'Maduro','semi_ripe':'Pintón','unripe':'Verde'}
def load(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def rows(path):
    with (ROOT/path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def pct(x):return f'{100*float(x):.2f} %'.replace('.',',')
def dec(x,n=4):return f'{float(x):.{n}f}'.replace('.',',')
def integer(x):return f'{int(x):,}'.replace(',','.')

doc=Document()
sec=doc.sections[0]
sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.top_margin=sec.bottom_margin=Inches(.7)
sec.left_margin=sec.right_margin=Inches(.75)
sec.header_distance=sec.footer_distance=Inches(.3)
for name in ('Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3','Header','Footer'):
    st=doc.styles[name];st.font.name='Calibri';st.font.color.rgb=RGBColor(0,0,0)
    st.element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:ascii'),'Calibri')
    st.element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:hAnsi'),'Calibri')
normal=doc.styles['Normal'];normal.font.size=Pt(11)
normal.paragraph_format.line_spacing=1.1
normal.paragraph_format.space_after=Pt(7)
normal.paragraph_format.widow_control=True
doc.styles['Title'].font.size=Pt(26)
doc.styles['Title'].font.bold=True
doc.styles['Subtitle'].font.size=Pt(12)
for style,size in [('Heading 1',18),('Heading 2',12.5),('Heading 3',11)]:
    st=doc.styles[style];st.font.size=Pt(size);st.font.bold=True
    st.paragraph_format.space_before=Pt(12);st.paragraph_format.space_after=Pt(7)
    st.paragraph_format.keep_with_next=True
header=sec.header.paragraphs[0];header.text='MADUREZ DE CAFÉ   |   DOCUMENTACIÓN TÉCNICA'
header.runs[0].font.size=Pt(8)
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
footer.add_run('Página ').font.size=Pt(9)
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
doc.core_properties.title='Documentación técnica del proyecto de madurez de café'
doc.core_properties.subject='Base técnica y guion para preparar diapositivas'
doc.core_properties.author='Proyecto de madurez de café'

def p(text,style=None):return doc.add_paragraph(text,style)
def h(text):return doc.add_heading(text,2)
def chapter(text):
    item=doc.add_heading(text,1);item.paragraph_format.page_break_before=True
    return item
def bullets(items):
    for item in items:p(item,'List Bullet')
def link(paragraph,text,url):
    rel=paragraph.part.relate_to(url,RT.HYPERLINK,is_external=True)
    node=OxmlElement('w:hyperlink');node.set(qn('r:id'),rel)
    run=OxmlElement('w:r');props=OxmlElement('w:rPr')
    color=OxmlElement('w:color');color.set(qn('w:val'),'174F77');props.append(color)
    run.append(props);content=OxmlElement('w:t');content.text=text;run.append(content)
    node.append(run);paragraph._p.append(node)
def evidence(*paths):
    item=p('Evidencia del proyecto: ')
    item.paragraph_format.space_before=Pt(8)
    for i,path in enumerate(paths):
        if i:item.add_run(' · ')
        link(item,path,REPO+'/blob/main/'+path)
    for run in item.runs:run.font.size=Pt(9)
def table(headers,data,widths):
    t=doc.add_table(rows=1,cols=len(headers));t.autofit=False;t.alignment=WD_TABLE_ALIGNMENT.CENTER
    for col,width in zip(t.columns,widths):col.width=Inches(width)
    for i,head in enumerate(headers):t.rows[0].cells[i].text=str(head)
    for values in data:
        for cell,value in zip(t.add_row().cells,values):cell.text=str(value)
    for ri,row in enumerate(t.rows):
        trPr=row._tr.get_or_add_trPr();trPr.append(OxmlElement('w:cantSplit'))
        if ri==0:trPr.append(OxmlElement('w:tblHeader'))
        for ci,cell in enumerate(row.cells):
            cell.width=Inches(widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcPr=cell._tc.get_or_add_tcPr()
            margins=OxmlElement('w:tcMar')
            for edge,val in [('top','65'),('bottom','65'),('left','95'),('right','95')]:
                el=OxmlElement('w:'+edge);el.set(qn('w:w'),val);el.set(qn('w:type'),'dxa');margins.append(el)
            tcPr.append(margins)
            border=OxmlElement('w:tcBorders')
            for edge in ('top','bottom','left','right'):
                el=OxmlElement('w:'+edge);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');border.append(el)
            tcPr.append(border)
            if ri==0 or ri%2==0:
                shading=OxmlElement('w:shd');shading.set(qn('w:fill'),'243746' if ri==0 else 'F2F5F7');tcPr.append(shading)
            for para in cell.paragraphs:
                para.paragraph_format.space_after=Pt(1);para.paragraph_format.space_before=Pt(1)
                para.paragraph_format.line_spacing=1.04
                if ci>0:para.alignment=WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs:
                    run.font.size=Pt(9.5);run.font.bold=(ri==0)
                    run.font.color.rgb=RGBColor(255,255,255) if ri==0 else RGBColor(0,0,0)
    p('').paragraph_format.space_after=Pt(0)
    return t
def equation(name,numerator,denominator):
    para=p('');para.paragraph_format.space_after=Pt(10)
    math=OxmlElement('m:oMath')
    def mr(text):
        r=OxmlElement('m:r');tt=OxmlElement('m:t');tt.text=text;r.append(tt);return r
    math.append(mr(name+' = '))
    fraction=OxmlElement('m:f')
    num=OxmlElement('m:num');num.append(mr(numerator))
    den=OxmlElement('m:den');den.append(mr(denominator))
    fraction.extend([num,den]);math.append(fraction);para._p.append(math)
def code(text):
    para=p(text);para.paragraph_format.space_after=Pt(8)
    for run in para.runs:run.font.name='Consolas';run.font.size=Pt(9)
def figure(path,caption,width=5.8):
    item=doc.add_paragraph();item.alignment=WD_ALIGN_PARAGRAPH.CENTER
    shape=item.add_run().add_picture(str(ROOT/path),width=Inches(width))
    shape._inline.docPr.set('descr',caption)
    cap=p(caption);cap.paragraph_format.space_after=Pt(9)
    for run in cap.runs:run.font.size=Pt(9)

results={n:load(f'experimentos/transfer_01/{n}/resultado.json') for n in NAMES}
extensions=load('experimentos/transfer_01/resumen_extensiones.json')['phases']
fine=rows('experimentos/finetune_01/comparacion.csv')
counts=rows('dataset_preparado/csv/distribucion.csv')

doc.add_paragraph('Documentación técnica del proyecto de madurez de café','Title')
doc.add_paragraph('Base para preparar diapositivas y sustentar los resultados','Subtitle')
p('Corte de evidencia 25 de noviembre de 2026')
p('Se desarrolló un sistema de visión artificial para localizar frutos de café y clasificar su estado visual en cinco categorías. El trabajo integra preparación reproducible de datos, tres CNN, dos Transformers, optimización con Hyperband, validación cruzada por grupos, ensembles y pilotos de GAN, aprendizaje federado y entrenamiento distribuido. El resultado se utiliza desde cámara, consola y una aplicación React con un backend Python desplegable en Docker y Dokploy.')
p('El ensemble actualizado obtuvo 92,42 % de accuracy, 77,98 % de precisión macro y 0,7503 de F1 macro en 5.116 recortes de validación. Estos valores no son resultados del conjunto de prueba ni una medición del sistema completo de detección y clasificación en fotografías nuevas. El 15 % de prueba permanece reservado.')
table(['Dato principal','Estado comprobado'],[
('Imágenes preparadas','2.359 archivos y 884 grupos'),('Partición fija','1.651 train / 354 valid / 354 test'),
('Clasificadores','ResNet50, EfficientNet B0, DenseNet121, ViT B 16 y Swin T'),
('Modelo predeterminado de la app','EfficientNet B0 con ajuste fino'),('Método de comparación','Validación fija y cinco folds por grupos dentro de train'),
('Uso de GAN en los pesos desplegados','No se incorporaron los sintéticos a los modelos finales')],[2.3,4.7])
p('Cómo usar este informe. Las secciones técnicas explican las decisiones y sus límites. Al final se propone un guion de 20 diapositivas con mensajes, recursos visuales y notas de exposición. Las tablas se calcularon a partir de los JSON y CSV guardados, dando prioridad a los artefactos numéricos frente a resúmenes redactados.')
link(p('Repositorio: '),'juanodsk/ia-cafe-maduracion',REPO)

chapter('1 Problema y alcance del sistema')
p('La tarea consiste en estimar la madurez visible del fruto de Coffea arabica mediante imágenes. El criterio operativo se representa con las clases verde, pintón o semimaduro, maduro, sobremaduro y seco. La etiqueta no mide grados Brix, humedad, calidad de taza ni condiciones internas del fruto; esas variables no fueron anotadas ni validadas en este proyecto.')
h('Dos tareas supervisadas diferentes')
p('Localización. Un detector YOLO11n identifica cajas que contienen frutos. Se entrenó como detector de una sola clase, fruit, para proponer regiones de interés sin decidir todavía su madurez.')
p('Clasificación. Cada caja se recorta y se entrega a una CNN, un Transformer o al ensemble. La salida contiene cinco puntuaciones softmax y una clase candidata. La aplicación dibuja el rectángulo y muestra la madurez junto a la puntuación del clasificador.')
bullets(['Unidad de partición: fotografía y grupo de captura; sus recortes heredan la misma partición.','Unidad de entrenamiento de los clasificadores: un recorte de fruto anotado.','Unidad de evaluación de clasificación: recorte de validación con caja de referencia.','Unidad de inferencia de la app: caja predicha por el detector sobre una foto nueva.'])
p('Esta separación explica por qué un buen resultado del clasificador no garantiza el mismo rendimiento en cámara. El detector puede omitir frutos o recortar regiones incompletas; además cambian iluminación, distancia, oclusión y fondo. No se ha calculado una métrica final que reúna ambos errores en una prueba independiente.')
h('Objetivos implementados')
p('Comparar las cinco arquitecturas con una partición común; reducir fugas identificables; seleccionar hiperparámetros sin utilizar test; conservar predicciones y configuraciones; explorar las extensiones solicitadas y construir una demostración funcional. El sistema produce una ayuda visual de clasificación, sin accionar una máquina de selección física.')
evidence('cafe_ml/camera.py','cafe_ml/web_api.py','docs/metodologia_entrenamiento.md')

chapter('2 Dataset de origen y etiquetas')
p('Se utilizó Coffee Fruit Maturity, publicado por Ciencia Cafeto en Roboflow, versión v5 de octubre de 2023. La documentación de la descarga declara especie Coffea arabica y licencia CC BY 4.0. El formato exportado fue COCO Segmentation; aunque conserva polígonos, los experimentos operativos emplean cajas y recortes, no un modelo de segmentación por píxel.')
table(['Etiqueta','Interpretación','Recortes train','Recortes valid','Recortes test'],[(key,LABELS[key],integer(next(r['annotations'] for r in rows('dataset_preparado/csv/distribucion_clases.csv') if r['split']=='train' and r['label']==key)),integer(next(r['annotations'] for r in rows('dataset_preparado/csv/distribucion_clases.csv') if r['split']=='valid' and r['label']==key)),integer(next(r['annotations'] for r in rows('dataset_preparado/csv/distribucion_clases.csv') if r['split']=='test' and r['label']==key))) for key in LABELS],[1.1,1.5,1.45,1.45,1.5])
p('La descarga contenía 2.360 archivos: 2.061 de entrenamiento, 201 de validación y 98 de prueba, mostrados como 87 %, 9 % y 4 %. Tras eliminar una copia exacta quedaron 2.359 imágenes y 34.111 anotaciones. Una fotografía puede contener muchos frutos y varias clases; los conteos de objetos no equivalen a capturas independientes ni a frutos físicos distintos.')
h('Preprocesamiento heredado de Roboflow')
p('Autoorientación aplicada; estiramiento a 640 × 640 píxeles; cero clases reasignadas y una clase eliminada cuyo nombre no aparece en la captura. Entrenamiento se exportó con tres salidas por ejemplo, cizallamiento horizontal y vertical entre −15° y +15° y exposición entre −12 % y +12 %. No se debe atribuir un nombre a la clase eliminada sin evidencia.')
p('La clase verde domina: representa alrededor del 75 % de los recortes de train y validación. Este desbalance motiva reportar F1 macro y métricas por clase. Una accuracy alta por sí sola podría ocultar errores importantes en sobremaduros y pintones.')
link(p('Fuente de datos: '),'Coffee Fruit Maturity de Ciencia Cafeto','https://universe.roboflow.com/ciencia-cafeto/coffee-fruit-maturity-befkg')
evidence('docs/dataset.md','dataset_preparado/csv/distribucion_clases.csv')

chapter('3 Partición fija y prevención de fugas de datos')
table(['Conjunto','Imágenes','Porcentaje real','Grupos','Anotaciones'],[(r['split'],integer(r['images']),dec(r['image_percentage'],4)+' %',r['groups'],integer(r['annotations'])) for r in counts],[1.3,1.2,1.6,1.1,1.8])
p('La proporción objetivo fue 70 % / 15 % / 15 %, con semilla 42. Los números enteros 1.651 / 354 / 354 se obtuvieron mediante restos mayores: se asigna la parte entera de cada cuota y después se distribuyen las imágenes restantes según la mayor fracción. Los pequeños desvíos porcentuales son consecuencia de trabajar con archivos indivisibles.')
h('Controles realizados')
bullets(['Calcular SHA-256 de contenido para encontrar duplicados exactos.','Unir variantes con el mismo nombre anterior a .rf. y nombres que comparten una marca AAAAMMDD_HHMMSS.','Asignar cada grupo completo a un solo conjunto, aproximando también las proporciones por clase.','Comprobar que nombres fuente, fechas identificadas, grupos y hashes no se crucen entre train, valid y test.','Conservar una sola copia del duplicado detectado y sus 13 anotaciones; no sumar dos versiones del mismo trazado.'])
p('La fuga inicial se observó en una imagen idéntica compartida por train y validación. Las pequeñas diferencias entre sus anotaciones no convertían la imagen en una observación independiente. Se eligió una copia de validación y se preservaron los originales para auditoría.')
h('Alcance de la protección')
p('Estos controles reducen fugas identificables, pero no garantizan independencia por finca, planta o fruto físico, porque faltan esos identificadores. La v5 ya contenía aumentos y no permite recuperar de manera fiable todos los originales sin transformación; por ello, validación y test pueden incluir variantes heredadas. Una futura prueba externa debe usar fotos nuevas sin aumentos y grupos de finca o sesión definidos de antemano.')
evidence('docs/particion_dataset.md','scripts/preparar_dataset.cjs','scripts/verificar_dataset.cjs')

chapter('4 Reproducibilidad y preparación por lotes')
p('La asignación se conserva en CSV y COCO para que todas las arquitecturas utilicen los mismos ejemplos. Los manifiestos no se recalculan al elegir otro modelo ni al cargar una imagen en la app. Test se conserva para una única evaluación final después de fijar todas las decisiones experimentales.')
table(['Archivo','Función'],[
('train.csv valid.csv test.csv','Partición fija de imágenes'),('imagenes.csv y grupos.csv','Procedencia, SHA-256, grupos y clases por imagen'),('anotaciones.csv','Una fila por fruto con imagen padre, caja y clase'),('distribucion.csv y distribucion_clases.csv','Conteos por conjunto y categoría'),('duplicados_excluidos.csv','Trazabilidad del duplicado y la copia conservada'),('preparacion.json y COCO por conjunto','Parámetros, semilla, IDs y geometrías actualizados')],[3.05,3.95])
h('Carga de datos')
p('CoffeeFruitDataset abre la fotografía bajo demanda, limita la caja al área visible y recorta el fruto. Un DataLoader agrupa recortes y una caché pequeña evita guardar todas las imágenes en RAM. Se auditaron 531 cajas de train y 111 de validación que excedían ligeramente el borde; se recortaron al límite. Treinta recortes de train tenían un lado menor de ocho píxeles, una limitación de resolución que no desaparece al reescalar.')
h('Transformaciones')
p('Se usa RGB y el preprocesamiento oficial de los pesos TorchVision: redimensionamiento, recorte central, tensor y normalización ImageNet. La entrada final es 224 × 224. Las medias son 0,485 / 0,456 / 0,406 y las desviaciones 0,229 / 0,224 / 0,225. No se cambian colores al azar, porque el color es parte de la señal de madurez. En el ajuste fino se permite volteo horizontal y rotación geométrica moderada.')
p('Durante la primera ronda se extrajeron embeddings por lotes en GPU y se guardaron matrices numéricas. La media y escala de características se ajustan exclusivamente con el subconjunto de entrenamiento de cada ajuste o fold; normalizar con todo el dataset introduciría información de la evaluación.')
evidence('cafe_ml/data.py','cafe_ml/models.py','experimentos/auditoria_recortes.json')

chapter('5 Arquitecturas y aprendizaje por transferencia')
table(['Familia','Modelo','Pesos ImageNet','Dimensión del embedding'],[
('CNN','ResNet50','1K V2','2.048'),('CNN','EfficientNet B0','1K V1','1.280'),('CNN','DenseNet121','1K V1','1.024'),('Transformer','ViT B 16','1K V1','768'),('Transformer','Swin T','1K V1','768')],[1.0,2.1,1.6,2.3])
p('ResNet usa conexiones residuales que facilitan el paso de información y gradientes [R1]. DenseNet concatena características de capas anteriores para reutilizarlas [R3]. EfficientNet estudia el escalamiento conjunto de profundidad, ancho y resolución; aquí se eligió su variante B0 [R2]. Estas propiedades motivan la comparación, pero no demuestran por sí mismas superioridad en café.')
p('ViT B 16 divide la imagen en parches de 16 × 16 y aplica atención sobre sus representaciones [R4]. Con entrada 224 × 224 se obtienen 14 × 14 parches. Swin T incorpora atención en ventanas desplazadas y una organización jerárquica [R5]. Los dos usan pesos preentrenados; no se entrenaron Transformers desde cero con 2.359 fotografías.')
h('Qué se entrenó realmente')
p('En la ronda inicial el backbone ImageNet permanece congelado, con BatchNorm, dropout y stochastic depth del extractor en evaluación. Se reemplaza su clasificador por una cabeza para cinco clases. La cabeza aprende una estandarización basada en train y una red con dropout, capa oculta opcional, GELU y salida lineal. Los ensayos ganadores de los cinco modelos usaron 256 unidades ocultas.')
p('El optimizador AdamW minimiza entropía cruzada ponderada. Los pesos de clase se calculan con la frecuencia de train y un exponente elegido de 0,5 o 1,0. La extracción compartida de embeddings reduce el costo de repetir búsquedas y folds. Cada backbone es distinto y tiene su propio preprocesamiento fijado por la versión de pesos.')
evidence('cafe_ml/models.py','cafe_ml/head_training.py','experimentos/transfer_01/configuracion.json')

chapter('6 Optimización con Hyperband')
p('Se ejecutó un estudio Optuna por arquitectura con muestreo aleatorio y semilla 42. Cada estudio tuvo 12 configuraciones candidatas, para un total de 60 ensayos. Hyperband asigna presupuesto de épocas y poda configuraciones menos prometedoras; así evita gastar el máximo en todas ellas [R6]. El criterio de selección fue F1 macro de validación, no accuracy ni rendimiento en test.')
table(['Variable de la cabeza','Espacio de búsqueda'],[
('Learning rate','0,0001 a 0,03 en escala logarítmica'),('Batch size','128 o 256 embeddings'),('Weight decay','0,000001 a 0,01 en escala logarítmica'),('Dropout','0 o 0,2'),('Dimensión oculta','Sin capa oculta o 256 con GELU'),('Exponente de pesos de clase','0,5 o 1,0'),('Recurso de Hyperband','Hasta 12 épocas; mínimo 2; reducción 3')],[2.5,4.5])
h('Configuraciones seleccionadas')
table(['Modelo','LR','Batch','Weight decay','Dropout','Época elegida'],[(DISPLAY[n],f"{results[n]['best_params']['lr']:.3g}",results[n]['best_params']['batch_size'],f"{results[n]['best_params']['weight_decay']:.3g}",results[n]['best_params']['dropout'],results[n]['best_epoch']) for n in NAMES],[1.6,1.0,.8,1.25,1.0,1.35])
p('El batch 128 de la tabla corresponde al entrenamiento de la cabeza sobre embeddings, no al número de fotos servidas simultáneamente por la app. Las épocas elegidas pueden ser menores que el presupuesto máximo por poda o por selección del mejor checkpoint. La configuración ganadora se vuelve a ajustar con la misma semilla para verificar que reproduce el resultado antes de guardarse.')
p('Se preservaron parámetros, historial, estados de ensayos completos o podados y bases SQLite locales. Las bases de estudio y embeddings regenerables se excluyeron de Git; los resúmenes de resultados y los checkpoints permiten auditar la selección publicada.')
evidence('docs/metodologia_entrenamiento.md','cafe_ml/head_training.py')

chapter('7 Validación cruzada y evaluación independiente')
p('Además del holdout fijo, se ejecutó StratifiedGroupKFold con cinco folds dentro del 70 % de entrenamiento. La estratificación intenta conservar clases y la restricción de grupos mantiene juntas las variantes y capturas relacionadas. Se guardaron asignaciones por fruto y grupo, predicciones fuera de fold y métricas de cada ajuste.')
bullets(['Separar train en cinco subconjuntos por grupos.','Entrenar una cabeza nueva con cuatro folds y evaluar en el restante.','Recalcular normalización y pesos de clase solamente con el train de ese fold.','Repetir cinco veces para que cada fold actúe una vez como evaluación.','Resumir F1 macro con media y desviación estándar muestral entre folds.'])
p('Los hiperparámetros y el número de épocas de la cabeza se fijaron usando la validación externa de la primera búsqueda y después se reutilizaron en los folds. Por tanto, la CV es condicionada a esa selección, no una validación cruzada anidada de todo el procedimiento de optimización. La desviación entre folds tampoco es un intervalo de confianza ni una estimación entre cinco entrenamientos completos independientes del backbone.')
table(['Técnica','Uso en este proyecto'],[
('Holdout','Implementado como partición fija 70 / 15 / 15'),('K fold estratificado por grupos','Implementado con cinco folds dentro de train'),('Stratified k fold sin grupos','Concepto relacionado; se añadió la restricción de captura'),('Leave one out y leave p out','Explicados, sin experimentos ejecutados'),('Monte Carlo o shuffle split','Explicado, sin experimentos ejecutados'),('Rolling de series temporales','No ejecutado; fecha de exportación no prueba fecha real de captura')],[2.65,4.35])
p('Test contiene 354 imágenes y 5.117 anotaciones; su existencia y sus conteos están documentados, pero no se utilizaron predicciones de test para seleccionar modelos, ensambles o mejoras. El siguiente cierre experimental debe fijar el candidato y evaluar ese conjunto una sola vez.')
evidence('cafe_ml/head_training.py','dataset_preparado/csv/test.csv')

chapter('8 Métricas y resultados de los cinco modelos')
p('Accuracy mide la proporción total de aciertos. Precisión de una clase indica qué parte de sus predicciones positivas eran correctas. Recall indica qué parte de sus ejemplos reales se recuperaron. F1 combina precisión y recall; macro F1 promedia las cinco clases con el mismo peso. Balanced accuracy es el promedio del recall por clase, y no debe rotularse como precisión macro.')
equation('Precisión','TP','TP + FP')
equation('Recall','TP','TP + FN')
equation('F1','2 × Precisión × Recall','Precisión + Recall')
table(['Modelo','Accuracy valid','Precisión macro','F1 macro valid','F1 macro CV'],[(DISPLAY[n],pct(results[n]['validation_metrics']['accuracy']),pct(results[n]['validation_metrics']['macro_precision']),dec(results[n]['validation_metrics']['macro_f1']),next(dec(r['cv_macro_f1_mean'])+' ± '+dec(r['cv_macro_f1_std']) for r in rows('experimentos/transfer_01/comparacion_validacion.csv') if r['model']==n)) for n in NAMES],[1.6,1.25,1.35,1.25,1.55])
p('Los valores corresponden a 5.116 recortes de validación con cajas anotadas. EfficientNet B0 fue el mejor individual por F1 macro del holdout, con 0,7280, mientras DenseNet121 obtuvo la mayor accuracy individual inicial, 90,85 %. Un ranking puede cambiar al utilizar otra métrica; ese cambio no es una contradicción.')
p('Siempre predecir verde acertaría alrededor del 75 % en esta validación debido al desbalance. Por eso se informa F1 macro y la matriz de confusión por clase, y no se presenta una única cifra de accuracy como prueba suficiente de buen comportamiento. Las cinco métricas de clasificación son diferentes de las métricas del detector y de los porcentajes softmax que ve el usuario.')
evidence(*[f'experimentos/transfer_01/{n}/resultado.json' for n in ('resnet50','efficientnet_b0','densenet121','vit_b_16','swin_t')])

chapter('9 Ajuste fino y cambios observados')
p('La segunda ronda restauró los pesos y cabezas guardados de EfficientNet B0 y Swin T. Se descongelaron las últimas etapas del extractor y la cabeza, manteniendo congeladas las capas iniciales y las estadísticas de BatchNorm. Se utilizaron exclusivamente las mismas particiones de train y validación, sin datos sintéticos ni test.')
table(['Modelo','Accuracy antes','Accuracy después','F1 antes','F1 después'],[(DISPLAY.get(r['model'],'Ensemble de cinco'),pct(r['accuracy_before']),pct(r['accuracy_after']),dec(r['macro_f1_before']),dec(r['macro_f1_after'])) for r in fine],[1.8,1.3,1.4,1.25,1.25])
table(['Modelo','Precisión macro antes','Precisión macro después'],[(DISPLAY.get(r['model'],'Ensemble de cinco'),pct(r['macro_precision_before']),pct(r['macro_precision_after'])) for r in fine],[2.5,2.25,2.25])
p('AdamW utilizó LR 0,00001 en el backbone y 0,00003 en la cabeza, batch 32, máximo seis épocas y paciencia de dos sin mejora de F1 macro. EfficientNet ejecutó cuatro épocas y conservó la segunda. Swin ejecutó dos épocas sin superar el punto inicial y conservó la época cero; no se presenta como mejora un ajuste que no ganó en la métrica de selección.')
p('EfficientNet mejoró aproximadamente 1,27 puntos porcentuales de accuracy y 4,62 puntos de precisión macro; F1 macro aumentó de 0,7280 a 0,7301. El ensemble pasó de 92,22 % a 92,42 % de accuracy, pero F1 macro bajó de 0,7510 a 0,7503. Ganar accuracy no implica mejorar todas las clases ni todas las métricas.')
p('La CV de la ronda inicial no se reasigna a los pesos afinados. No se ha repetido una CV completa del ajuste fino ni se ha estimado todavía una mejora sobre test. Los pesos afinados de EfficientNet son los utilizados por defecto en la app por el equilibrio observado entre costo de inferencia y rendimiento.')
evidence('experimentos/finetune_01/comparacion.csv','experimentos/finetune_01/efficientnet_b0/configuracion.json')

chapter('10 Ensembles y comportamiento por clase')
p('El soft voting combina vectores de probabilidades para el mismo fruto y con idéntico orden de clases. El heterogéneo promedia cinco arquitecturas diferentes con peso 1/5 cada una. El homogéneo usa cinco cabezas de EfficientNet entrenadas por fold con el mismo extractor congelado. El híbrido de familias reserva 50 % del peso a CNN y 50 % a Transformers: cada CNN pesa 1/6 y cada Transformer 1/4.')
ensembles=[('Heterogéneo de cinco','heterogeneo_cinco'),('Tres CNN','familia_cnn'),('Dos Transformers','familia_transformer'),('Híbrido por familia','hibrido_familias'),('Homogéneo por folds','homogeneo_cinco_folds')]
ensemble_rows=[]
for display,key in ensembles:
    metrics=load(f'experimentos/transfer_01/ensembles/{key}/validacion/metrics.json')
    ensemble_rows.append((display,pct(metrics['accuracy']),dec(metrics['macro_f1'])))
table(['Ensemble inicial','Accuracy valid','F1 macro valid'],ensemble_rows,[3.2,1.9,1.9])
p('Los promedios de las tres CNN y de los dos Transformers son heterogéneos por arquitectura; pertenecer a la misma familia no los hace homogéneos. El híbrido no es una red adicional entrenada ni un algoritmo genético: es una combinación jerárquica de probabilidades con pesos fijados.')
fine_metrics=load('experimentos/finetune_01/ensembles/heterogeneo_cinco/validacion/metrics.json') if (ROOT/'experimentos/finetune_01/ensembles/heterogeneo_cinco/validacion/metrics.json').exists() else None
if fine_metrics:
    table(['Clase','F1 ensemble actualizado','Recall actualizado'],[(LABELS[r['label']],dec(r['f1']),pct(r['recall'])) for r in fine_metrics['per_class']],[2.4,2.3,2.3])
else:
    p('En el ensemble afinado, sobremaduro conservó una dificultad importante: F1 0,4271 y recall 33,61 %. La clase verde alcanzó F1 0,9803. Este contraste demuestra que la accuracy global no resume el comportamiento de cada estado de madurez.')
p('El ensemble actualizado se usa como opción de la app, pero su F1 macro ligeramente menor debe comunicarse. No se dispone de una medición independiente que establezca su superioridad general frente al ensemble inicial.')
evidence('docs/resultados_finetune_01.md','cafe_ml/ensembles.py')

chapter('11 Experimento con redes generativas adversarias')
p('Una GAN entrena un generador y un discriminador en competencia: el primero produce imágenes y el segundo intenta distinguirlas de las reales [R7]. Se implementó una DCGAN condicional de 64 × 64 para explorar aumento de clases minoritarias, no para sustituir los datos reales ni demostrar automáticamente una mejora de clasificación.')
g=extensions['gan']['result'];cfg=g['configuration']
table(['Parámetro','Valor ejecutado'],[
('Presupuesto','5 épocas × 80 pasos = 400 actualizaciones'),('Batch y espacio latente','64 y dimensión latente 64'),('Condicionamiento','Cinco clases y embedding de 16 dimensiones'),('Optimizador','Adam LR 0,0002 y betas 0,5 / 0,999'),('Rango de imágenes','−1 a 1'),('Muestreo real','Ponderado por inversa de frecuencia de clase'),('Sintéticos producidos','1.024; 256 por cada una de cuatro clases minoritarias')],[2.7,4.3])
h('Corrección y ablación')
p('El piloto inicial mezclaba modos de BatchNorm del discriminador entre sus propias actualizaciones y el paso del generador. Se conservó para auditoría y se reinició el experimento corregido. Durante el paso del generador, el discriminador mantiene el modo de entrenamiento con parámetros congelados; una prueba de regresión comprueba esta decisión.')
ab=extensions['gan_ablation']['result']
p(f"La ablación comparó EfficientNet con datos reales frente a reales más sintéticos usando semilla 43, los mismos hiperparámetros y tres épocas fijas. Macro F1 pasó de {dec(ab['baseline_validation_metrics']['macro_f1'])} a {dec(ab['validation_metrics']['macro_f1'])}, un cambio de {dec(ab['macro_f1_change'])}. La calidad visual mostró ruido y estructuras poco definidas; las etiquetas sintéticas no quedaron validadas agronómicamente.")
p('Los sintéticos se restringieron a train y no se incorporaron a los clasificadores desplegados. Tampoco se usaron en los folds, porque el generador ya había visto todo train y contaminaría el fold de evaluación. Un estudio válido con GAN dentro de CV requeriría entrenar un generador nuevo solo con el train de cada fold. Las pérdidas de GAN no equivalen a exactitud ni a calidad agronómica.')
evidence('experimentos/transfer_01/gan/resultado.json','experimentos/transfer_01/resumen_extensiones.json','docs/correccion_gan.md')

chapter('12 Aprendizaje federado y distribuido')
p('Federado y distribuido resuelven problemas distintos. FedAvg alterna aprendizaje local por cliente y agregación de parámetros en un servidor [R8]. DDP reparte ejemplos entre procesos y sincroniza gradientes para entrenar réplicas del mismo modelo. En ambos pilotos del proyecto se actualiza la cabeza de clasificación, conservando el backbone preentrenado.')
fed=extensions['federated']['result'];ddp=extensions['distributed']['result']
table(['Aspecto','FedAvg','DDP'],[
('Ejecución','Tres clientes simulados en un equipo','Dos procesos CPU reales en un equipo'),('Presupuesto','Cinco rondas y dos épocas locales por ronda','Dos épocas con backend Gloo'),('Datos por participante','8.000 / 8.561 / 7.317 recortes','DistributedSampler sobre train'),('Batch','128 local','128 por proceso y 256 global'),('Accuracy valid',pct(fed['validation_metrics']['accuracy']),pct(ddp['validation_metrics']['accuracy'])),('F1 macro valid',dec(fed['validation_metrics']['macro_f1']),dec(ddp['validation_metrics']['macro_f1']))],[1.6,2.7,2.7])
p('En FedAvg se asignaron grupos completos tras un barajado con semilla 43 y se promediaron estados según la cantidad de muestras. Las estadísticas de normalización se agregaron mediante sumas locales ordinarias. No se implementaron agregación segura, privacidad diferencial, cifrado entre fincas ni clientes físicos remotos. La distribución observada no permite afirmar que se modeló un escenario fuertemente no IID.')
p('DDP se ejecutó con PyTorch 2.7.1 CPU en un entorno separado para disponer de Gloo en Windows. Los dos ranks terminaron con hashes de estado idénticos, evidencia de sincronización. No hubo filas de padding del sampler en ese presupuesto. Esto demuestra la ejecución técnica, no aceleración multi GPU o escalabilidad entre máquinas.')
p('Las métricas bajas de FedAvg y la diferencia de presupuesto entre pilotos y modelos optimizados deben mostrarse. No es correcto concluir que una técnica es inherentemente inferior a partir de estas corridas cortas; tampoco se debe presentar la simulación como un despliegue real en tres fincas.')
evidence('cafe_ml/federated.py','cafe_ml/distributed.py','experimentos/transfer_01/resumen_extensiones.json')

chapter('13 Localización y demostración con cámara')
p('Para dibujar cajas sobre fotos completas se añadió YOLO11n como localizador de una sola clase. Se entrenó durante 15 épocas con AdamW, batch 16 e imágenes de 640, partiendo de pesos preentrenados y congelando los primeros diez módulos. Solo se utilizaron train y validación de la partición fijada.')
det=load('experimentos/detector_01/resumen.json')['metrics']
table(['Métrica de localización','Validación'],[(name,pct(det[key])) for name,key in [('Precisión de cajas','metrics/precision(B)'),('Recall de cajas','metrics/recall(B)'),('mAP a IoU 0,50','metrics/mAP50(B)'),('mAP a IoU 0,50 a 0,95','metrics/mAP50-95(B)')]],[4.3,2.7])
p('La conversión a YOLO conservó 23.641 cajas de train y 5.007 de validación después de deduplicar cajas equivalentes; los manifiestos originales de clasificación conservan 23.878 y 5.116 recortes. Son tareas y conteos distintos. Estas métricas evalúan encontrar frutos, no su estado de madurez.')
h('Flujo de inferencia')
bullets(['Capturar un frame BGR con OpenCV o decodificar la foto como RGB.','Ejecutar detección con confianza mínima 0,30, IoU de NMS 0,50 y hasta 100 cajas.','Limitar cajas al área de imagen, recortar y aplicar el transform del clasificador.','Clasificar por lotes y convertir logits a softmax; para ensemble, promediar los cinco vectores.','Marcar Incierto si la puntuación ganadora es menor de 55 %; conservar clase candidata y puntuación.'])
p('La cámara USB se comprobó localmente. Q o Escape cierran la ventana y S guarda una captura. También existe una zona manual para un fruto cuando no se utiliza el detector. El umbral del 55 % es una regla operativa, no una calibración probabilística. Los recuadros son rectángulos y no trazados exactos del contorno de cada fruto.')
evidence('experimentos/detector_01/resumen.json','cafe_ml/camera.py','docs/camara.md')

chapter('14 Aplicación React y despliegue en Dokploy')
p('La interfaz React permite seleccionar o arrastrar una foto, elegir cualquiera de los cinco modelos o el ensemble, ejecutar el análisis, resaltar frutos y descargar JPG anotado e informe JSON. FastAPI sirve el frontend compilado y la inferencia desde el mismo servicio, evitando una API externa o una base de datos obligatoria.')
table(['Ruta','Función'],[('GET /','Interfaz React compilada'),('GET /api/health','Disponibilidad y modo local o servidor'),('GET /api/example','Fotografía de validación para demostración'),('POST /api/analyze','Archivo multipart y modelo; devuelve cajas, clases, puntuaciones y recuentos')],[2.2,4.8])
p('Se aceptan JPG, PNG y WEBP de hasta 15 MB y 20 megapíxeles. Se corrige la orientación EXIF y se limita el lado mayor a 1.920 píxeles conservando proporciones. La foto se procesa en el servidor de la aplicación y no se guarda automáticamente. Los pesos se cargan de forma diferida y se libera el clasificador anterior al cambiar de modelo.')
p('Docker usa una etapa Node para compilar React y una imagen Python con PyTorch CPU para servir la app. Incluye nueve archivos de inferencia, 590,8 MiB de pesos, comprobados por SHA-256. El contenedor se ejecuta sin root, con sistema de archivos de solo lectura y almacenamiento temporal. Compose fija un proceso, reinicio automático y límites configurables de dos CPU y cuatro GB de RAM.')
p('En Dokploy se configura un servicio Docker Compose desde main, archivo ./compose.yaml, dominio cafe.juanodsk.dev, servicio app y puerto interno 8765. Los dominios se encaminan mediante Traefik y sus cambios requieren redeploy. Git LFS descarga los modelos grandes durante el clonado; el dataset completo no se necesita para servir la app.')
p('Una prueba remota inicial obtuvo HTTP 200 y nueve frutos con EfficientNet CPU en aproximadamente 1,9 segundos. Una foto posterior reveló un error de interpretación de respuestas no JSON. La corrección 4b01ae8 añade mensajes HTTP claros, exclusión de inferencias simultáneas y lotes pequeños; requiere Redeploy para activarse en Dokploy. Un reinicio observado no confirma por sí solo falta de memoria.')
evidence('Dockerfile','compose.yaml','web/src/api.js','docs/docker_vps.md')

chapter('15 Verificación funcional y reproducibilidad operativa')
p('Las pruebas de funcionamiento comprueban que la aplicación procesa una imagen y entrega una respuesta coherente; no sustituyen una evaluación supervisada de precisión. La foto integrada pertenece a validación y debe anunciarse así. Para demostrar generalización hay que recolectar y etiquetar fotos externas que no participaron en entrenamiento ni selección.')
table(['Comprobación','Evidencia'],[
('Prueba Docker de los seis modos','Cinco clasificadores y ensemble procesaron nueve frutos del ejemplo'),('Prueba sin Internet','El ensemble infirió en CPU con red Docker deshabilitada'),('Corrección de errores de respuesta','Once pruebas JavaScript de JSON, HTML, respuestas vacías y estados HTTP'),('Pipeline y API revisados','Trece pruebas Python incluyendo lotes, orden de detecciones, tamaño y concurrencia'),('Foto densa de 2.048 × 1.536','Cien cajas producidas con la misma distribución en la prueba anterior y revisada')],[2.8,4.2])
p('En una prueba puntual con límite Docker de dos CPU y cuatro GB, la configuración anterior registró 1.061,5 MiB de RSS máximo del proceso y 19,1 segundos; la revisada, con batch cuatro y dos hilos, registró 628,2 MiB y 15,2 segundos. No son mediciones de memoria total de la VPS ni un benchmark estadístico. Cambiaron dos parámetros y no se atribuye causalidad exclusiva a uno de ellos.')
h('Comandos esenciales')
code('git lfs pull\ndocker compose up -d --build\ndocker compose logs --tail=100 app\ndocker compose exec app python deploy/verify_models.py')
code('python -m cafe_ml.predict --imagen foto.jpg --modelo efficientnet_b0\npython -m cafe_ml.camera --fuente 0\npython -m cafe_ml.web_api')
p('La configuración inicial, semillas, versiones, manifiestos, distribuciones, pesos, historiales y predicciones permiten seguir la procedencia de los resultados. Git conserva datos de partición y modelos LFS, pero excluye fotos del dataset, entornos virtuales y embeddings regenerables. Repetir el entrenamiento exige restaurar las fotos originales además del repositorio.')
evidence('deploy/model_assets.json','scripts/verificar_app_fotos.py','requirements.lock.txt','web/tests/api.test.js')

chapter('16 Conclusiones y trabajo pendiente')
bullets(['Se completó una comparación reproducible de tres CNN y dos Transformers con la misma partición y control de grupos identificables.','El ensemble inicial obtuvo el mejor F1 macro de validación de la primera ronda, 0,7510.','El ajuste fino mejoró EfficientNet y elevó la accuracy del ensemble, pero no mejoró su F1 macro.','La GAN fue implementada y evaluada, pero no produjo una mejora en la ablación ni muestras con calidad suficiente para adoptarlas.','FedAvg y DDP se ejecutaron como pilotos de cabeza; sus límites de simulación y presupuesto deben acompañar las métricas.','El prototipo integra detección, clasificación, cámara, carga de fotos y despliegue en CPU.'])
h('Límites que deben aparecer en la exposición')
p('La v5 contiene aumentos heredados y no dispone de identificadores completos de finca, planta o fruto. Los estados de madurez no fueron revisados aquí por expertos agrónomos. La clase sobremaduro mantiene bajo recall; las fotos densas, lejanas u ocluidas plantean problemas de detección. El porcentaje softmax no está calibrado y una demostración exitosa con una imagen no acredita exactitud general.')
h('Siguiente fase experimental propuesta')
bullets(['Fijar un único modelo o ensemble y el protocolo completo antes de abrir test.','Evaluar test una vez y guardar métricas por clase, matrices de confusión y predicciones.','Medir el sistema completo sobre fotos externas con cajas y madurez verificadas.','Obtener nuevas capturas sin aumentos, separadas por finca, planta y sesión.','Calibrar puntuaciones en un conjunto independiente de la prueba final y estudiar umbrales por costo del error.','Repetir experimentos con varias semillas y presupuestos comparables; mejorar clases minoritarias con datos reales antes de adoptar sintéticos.'])
p('Conclusión defendible. El proyecto demuestra un flujo experimental y operativo funcional, con resultados prometedores de validación y limitaciones registradas. Aún no demuestra generalización final entre fincas, equivalencia agronómica de etiquetas sintéticas ni desempeño validado en un sistema industrial de selección.')
evidence('docs/resultados_transfer_01.md','docs/resultados_finetune_01.md','experimentos/transfer_01/resumen_extensiones.json')

slides=[
('Problema y objetivo','Reconocer cinco estados visuales de madurez por fruto.','Foto propia o ejemplo de café con estados distintos.','Aclarar que se estima apariencia visual; no se mide calidad de taza ni azúcar. Presentar detección y clasificación como etapas distintas.'),
('Dataset y clases','Coffee Fruit Maturity v5 y cinco etiquetas supervisadas.','Cinco recortes reales etiquetados y el nombre del publicador.','Distinguir 2.359 archivos preparados de 34.111 anotaciones. Las variantes exportadas no equivalen a capturas independientes.'),
('Partición y datos disponibles','1.651 train, 354 valid y 354 test.','Tabla de imágenes, grupos y recortes de la sección 3.','Explicar el redondeo de 70 / 15 / 15 y que los recortes heredan la partición de su fotografía.'),
('Prevención de data leakage','Agrupar variantes y eliminar una copia exacta compartida.','Esquema de una captura y sus variantes dentro de un único conjunto.','Describir SHA-256, nombres fuente y marcas de tiempo. Señalar que faltan IDs físicos para garantizar independencia completa.'),
('Preprocesamiento y generador','Recorte bajo demanda y lotes que caben en memoria.','Fotografía, caja de referencia, recorte RGB y tensor 224 × 224.','Explicar caché pequeña, normalización ImageNet y ausencia de jitter de color. No confundir el tamaño original 640 con la entrada 224 del clasificador.'),
('Las tres CNN','ResNet50, EfficientNet B0 y DenseNet121.','Tabla breve de embeddings y conexiones residuales o densas.','Presentar su mecanismo principal y el uso común de transferencia ImageNet. Ninguna arquitectura se entrenó desde cero en la primera ronda.'),
('Los dos Transformers','ViT B 16 y Swin T.','Parches de ViT y ventanas desplazadas de Swin.','Explicar atención, representación global y jerarquía local. No prometer que atención supera a convoluciones con este dataset.'),
('Qué parámetros se aprendieron','Backbones congelados y cabezas para cinco clases.','Diagrama de extractor congelado y cabeza entrenable.','Justificar embeddings en disco, normalización solo de train, entropía cruzada ponderada y AdamW.'),
('Hyperband','Sesenta ensayos con presupuesto máximo de doce épocas.','Embudo de poda y tabla de espacios de búsqueda.','El criterio fue F1 macro. Aclarar que batch 128 es de embeddings y no de usuarios atendidos por la app.'),
('Validación cruzada','Cinco folds estratificados por grupos dentro de train.','Cinco filas con cuatro folds de ajuste y uno de evaluación.','Repetir normalización por fold. La búsqueda usa holdout y la CV resultante es condicionada, sin búsqueda anidada.'),
('Resultados iniciales','F1 macro individual entre 0,7006 y 0,7280.','Tabla de la sección 8 con accuracy, precisión macro y CV.','Destacar que EfficientNet gana por F1 del holdout y DenseNet por accuracy inicial. Indicar 5.116 recortes de validación, no test.'),
('Ajuste fino','EfficientNet mejora; Swin conserva el checkpoint previo.','Comparación antes y después de la sección 9.','Mostrar que el ensemble gana 0,20 puntos de accuracy pero pierde ligeramente F1. Evitar afirmar mejora general sin test.'),
('Ensembles','Combinación de probabilidades con diferentes estructuras.','Tabla heterogéneo, homogéneo e híbrido.','Pesos 1/5 para cinco modelos y 1/6 por CNN más 1/4 por Transformer en el híbrido. Promedio por familia no implica homogeneidad.'),
('GAN y ablación','La generación se implementó, pero no mejoró la clasificación.','Muestras reales y sintéticas identificadas; F1 antes y después.','Explicar 1.024 sintéticos, corrección de BatchNorm y cambio de F1 de aproximadamente −0,0116. No presentar pérdidas GAN como precisión.'),
('Aprendizaje federado','FedAvg con tres clientes simulados.','Servidor central y clientes con sus tamaños reales.','Cinco rondas y dos épocas locales. F1 0,3050. Aclarar que no hubo tres fincas, agregación segura ni privacidad diferencial.'),
('Aprendizaje distribuido','Dos procesos CPU con sincronización Gloo.','Dos ranks y una operación de promedio de gradientes.','Dos épocas, batch global 256 y hashes finales idénticos. F1 0,7029. No hubo benchmark multi GPU ni escalabilidad entre máquinas.'),
('Detector y cámara','YOLO11n localiza; el clasificador decide la madurez.','Captura anotada de pruebas/camara/demo_auto.jpg.','mAP50 95,02 % corresponde a localización. Explicar umbrales 0,30 y 0,55, cajas rectangulares y límite de cien detecciones.'),
('Aplicación y despliegue','React y FastAPI empaquetados con pesos en Docker.','Captura de la app con resultados y diagrama cliente servidor.','Servicio app en puerto interno 8765, Git LFS, CPU y Dokploy. El porcentaje es una puntuación softmax, no precisión global.'),
('Demostración y verificaciones','Subir una foto y seleccionar un fruto.','Foto nueva más el ejemplo de validación identificado como tal.','Enseñar JPG y JSON descargables. Explicar que una demo confirma funcionamiento y que las pruebas automáticas cubren errores y concurrencia.'),
('Conclusiones y próximos pasos','Evaluar test y validar generalización externa.','Lista de tres logros y tres límites medibles.','Cerrar con el resultado validado de 92,42 % de accuracy y 0,7503 de F1 del ensemble actualizado. Reiterar que no son métricas de test ni del pipeline completo.'),
]
for start in range(0,len(slides),4):
    chapter('Guion de diapositivas '+str(start+1)+' a '+str(min(start+4,len(slides))))
    for idx,(title,message,visual,note) in enumerate(slides[start:start+4],start+1):
        h(str(idx)+' '+title)
        p('Mensaje principal. '+message)
        p('Recurso visual. '+visual)
        p('Notas para exponer. '+note)

chapter('Referencias y fuentes para la exposición')
references=[
('R1','He y colaboradores','Deep Residual Learning for Image Recognition','https://arxiv.org/abs/1512.03385'),
('R2','Tan y Le','EfficientNet Rethinking Model Scaling for Convolutional Neural Networks','https://arxiv.org/abs/1905.11946'),
('R3','Huang y colaboradores','Densely Connected Convolutional Networks','https://arxiv.org/abs/1608.06993'),
('R4','Dosovitskiy y colaboradores','An Image is Worth 16x16 Words','https://arxiv.org/abs/2010.11929'),
('R5','Liu y colaboradores','Swin Transformer Hierarchical Vision Transformer using Shifted Windows','https://arxiv.org/abs/2103.14030'),
('R6','Li y colaboradores','Hyperband A Novel Bandit Based Approach to Hyperparameter Optimization','https://arxiv.org/abs/1603.06560'),
('R7','Goodfellow y colaboradores','Generative Adversarial Networks','https://arxiv.org/abs/1406.2661'),
('R8','McMahan y colaboradores','Communication Efficient Learning of Deep Networks from Decentralized Data','https://arxiv.org/abs/1602.05629'),
]
for key,author,title,url in references:
    para=p(f'[{key}] {author}. {title}. ');link(para,'Artículo original',url)
p('Las referencias fundamentan los métodos, no las métricas del café. Los valores experimentales se obtienen de los artefactos del repositorio enlazados al final de cada sección. Las cifras de ensayos, épocas y recursos describen lo ejecutado; los trabajos futuros se identifican como propuestas.')
h('Archivos recomendados para preparar gráficos')
bullets(['experimentos/transfer_01/comparacion_validacion.csv y figuras/comparacion_validacion.png','experimentos/finetune_01/comparacion.csv y comparacion.png','experimentos/transfer_01/ensembles/heterogeneo_cinco/validacion/confusion_matrix.png','dataset_preparado/csv/distribucion.csv y distribucion_clases.csv','experimentos/transfer_01/resumen_extensiones.json','pruebas/camara/demo_auto.jpg y pruebas/docker_app/demo_resultado.jpg'])
link(p('Código y documentación pública: '),REPO,REPO)
OUT.parent.mkdir(parents=True,exist_ok=True)
# The bundled default template carries a blue Title paragraph border.
# Remove paragraph borders from styles and content; table borders remain.
for xml in (doc.styles.element,doc.element):
    for border in xml.xpath('.//w:pBdr'):
        border.getparent().remove(border)
doc.save(OUT)
print(json.dumps({'document':str(OUT),'paragraphs':len(doc.paragraphs),'tables':len(doc.tables),'slides':len(slides)},ensure_ascii=False))
