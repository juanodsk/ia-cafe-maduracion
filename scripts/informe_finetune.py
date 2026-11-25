"""Render before/after results from completed fine-tuning artifacts only."""
import csv,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def main():
    run=ROOT/'experimentos/finetune_01'
    result=json.loads((run/'comparacion.json').read_text(encoding='utf-8'))
    lines=['# Mejoras mediante fine tuning','',
        'Se ajustaron las últimas etapas de EfficientNet B0 y Swin T usando las particiones guardadas. '
        'No se utilizaron datos de prueba ni imágenes sintéticas. Las métricas corresponden a los mismos 5.116 frutos de validación de la primera ronda.','',
        '| Modelo | Accuracy antes | Accuracy después | F1 macro antes | F1 macro después | Precisión macro antes | Precisión macro después |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    comparisons=[]
    for name,row in result['models'].items():
        comparisons.append((name,row['baseline_metrics'],row['validation_metrics']))
    comparisons.append(('ensemble_cinco',result['baseline_ensemble'],result['finetuned_ensemble']))
    rows=[]
    for name,old,new in comparisons:
        lines.append(f"| {name} | {100*old['accuracy']:.2f} % | {100*new['accuracy']:.2f} % | {old['macro_f1']:.4f} | {new['macro_f1']:.4f} | {old['macro_precision']:.4f} | {new['macro_precision']:.4f} |")
        rows.append({'model':name,'accuracy_before':old['accuracy'],'accuracy_after':new['accuracy'],
            'macro_f1_before':old['macro_f1'],'macro_f1_after':new['macro_f1'],
            'macro_precision_before':old['macro_precision'],'macro_precision_after':new['macro_precision']})
    lines.extend(['', 'EfficientNet B0 conservó la segunda época de cuatro ejecutadas. Swin T detuvo el ajuste tras dos épocas sin mejora de F1 macro y conservó los pesos de referencia (época cero). '
        'El ensemble ganó aproximadamente 0,20 puntos porcentuales de accuracy y 0,92 puntos de precisión macro; F1 macro bajó 0,0008. '
        'Es una mejora pequeña en esta validación, sin evidencia todavía de una mejora sobre el conjunto de prueba.'])
    with (run/'comparacion.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    figure,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    x=list(range(len(comparisons)))
    for ax,metric,title in zip(axes,['accuracy','macro_f1'],['Accuracy','F1 macro']):
        ax.bar([i-.18 for i in x],[old[metric] for _,old,_ in comparisons],width=.36,label='Antes')
        ax.bar([i+.18 for i in x],[new[metric] for _,_,new in comparisons],width=.36,label='Después')
        ax.set_xticks(x,[n for n,_,_ in comparisons],rotation=10);ax.set_ylim(0,1);ax.set_title(title);ax.legend();ax.grid(axis='y',alpha=.2)
    figure.savefig(run/'comparacion.png',dpi=160);plt.close(figure)
    lines+=['','## Resultados por clase del ensemble','',
        '| Clase | F1 antes | F1 después | Recall antes | Recall después |','| --- | ---: | ---: | ---: | ---: |']
    for old,new in zip(result['baseline_ensemble']['per_class'],result['finetuned_ensemble']['per_class']):
        assert old['label']==new['label']
        lines.append(f"| {old['label']} | {old['f1']:.4f} | {new['f1']:.4f} | {old['recall']:.4f} | {new['recall']:.4f} |")
    lines+=['','## Protocolo y límites','',
        'Se partió de los pesos preentrenados y las cabezas guardadas. Se mantuvieron congeladas las capas iniciales y las estadísticas de BatchNorm; '
        'las últimas etapas y la cabeza recibieron gradientes. AdamW utilizó learning rate de 0,00001 para el backbone y 0,00003 para la cabeza, '
        'batch 32, máximo seis épocas y parada tras dos épocas sin mejora de F1 macro. La normalización de características conservó las estadísticas de entrenamiento de la primera ronda.','',
        'La selección de checkpoint se hizo por F1 macro de validación; la accuracy y precisión macro se registran por separado. '
        'El ensemble sustituye únicamente las predicciones de los dos modelos afinados y mantiene las de los otros tres, con pesos uniformes.','',
        'La CV de la primera ronda no se presenta como CV de estos nuevos pesos. Esta comparación utiliza validación fija repetida; '
        'el 15 % de prueba sigue reservado para una evaluación final independiente. Se conservan las limitaciones de la descarga v5 y sus aumentos heredados.','',
        '[Comparación numérica](../experimentos/finetune_01/comparacion.csv) · '
        '[Gráfica](../experimentos/finetune_01/comparacion.png) · '
        '[Resultados y checkpoints](../experimentos/finetune_01/comparacion.json)','']
    (ROOT/'docs/resultados_finetune_01.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
