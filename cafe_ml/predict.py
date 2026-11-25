"""Classify an isolated coffee fruit, or a user-specified box, using saved weights."""
from __future__ import annotations
import argparse, gc, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from .experiment import ROOT, json_write
from .data import CLASS_NAMES, crop_bounds
from .finetune import load_finetuned
from .head_training import load_head_checkpoint
from .models import build_model, set_head, get_transforms, MODEL_NAMES

SPANISH={'dry':'Seco','overripe':'Sobremaduro','ripe':'Maduro',
         'semi_ripe':'Semimaduro / pintón','unripe':'Verde / inmaduro'}

def load_classifier(name, device):
    tuned=ROOT/'experimentos/finetune_01'/name/'checkpoint.pt'
    if name in ('efficientnet_b0','swin_t') and tuned.exists():
        return load_finetuned(tuned,device),str(tuned)
    checkpoint=ROOT/'experimentos/transfer_01'/name/'checkpoint.pt'
    model=build_model(name,pretrained=True)
    set_head(model,name,load_head_checkpoint(checkpoint,device))
    return model.to(device).eval(),str(checkpoint)

@torch.inference_mode()
def classify(image, model_name='efficientnet_b0', device='cpu'):
    names=MODEL_NAMES if model_name=='ensemble' else (model_name,)
    probabilities=[]; checkpoints=[]
    for name in names:
        model,checkpoint=load_classifier(name,device)
        tensor=get_transforms(name)(image).unsqueeze(0).to(device)
        with torch.autocast('cuda',dtype=torch.float16,enabled=device=='cuda'):
            logits=model(tensor)
        probabilities.append(logits.float().softmax(1).cpu().numpy()[0])
        checkpoints.append(checkpoint)
        del model,tensor
        gc.collect()
        if device=='cuda': torch.cuda.empty_cache()
    values=np.mean(probabilities,axis=0)
    winner=int(values.argmax())
    return {'model':model_name,'class':CLASS_NAMES[winner],'label':SPANISH[CLASS_NAMES[winner]],
        'score':float(values[winner]),'probabilities':dict(zip(CLASS_NAMES,map(float,values))),
        'checkpoints':checkpoints,'note':'Softmax scores are not calibrated certainty. Classification of a fruit crop, not automatic detection.'}

def save_preview(image,result,path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    axes[0].imshow(image);axes[0].axis('off')
    axes[0].set_title(result['label'])
    labels=[SPANISH[c] for c in CLASS_NAMES]
    values=[100*result['probabilities'][c] for c in CLASS_NAMES]
    axes[1].barh(labels,values,color=['#d77d31' if c==result['class'] else '#28659a' for c in CLASS_NAMES])
    axes[1].set(xlim=(0,100),xlabel='Puntuación del modelo (%)',title=result['model'])
    figure.savefig(path,dpi=150);plt.close(figure)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imagen',type=Path,required=True)
    parser.add_argument('--modelo',choices=(*MODEL_NAMES,'ensemble'),default='efficientnet_b0')
    parser.add_argument('--bbox',type=float,nargs=4,metavar=('X','Y','ANCHO','ALTO'))
    parser.add_argument('--esperado',choices=CLASS_NAMES)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--salida',type=Path)
    args=parser.parse_args()
    if not args.imagen.is_file(): parser.error(f'No existe la imagen: {args.imagen}')
    torch.set_num_threads(4)
    with Image.open(args.imagen) as original:
        image=ImageOps.exif_transpose(original).convert('RGB')
        if args.bbox: image=image.crop(crop_bounds(args.bbox,*image.size))
    result=classify(image,args.modelo,args.device)
    result.update(image=str(args.imagen.resolve()),bbox=args.bbox,device=args.device)
    if args.esperado:
        result.update(expected_class=args.esperado,correct=result['class']==args.esperado)
    identifier=hashlib.sha256(args.imagen.read_bytes()).hexdigest()[:8]
    output=args.salida or ROOT/'pruebas'/'resultados'/f'{args.imagen.stem}_{identifier}_{args.modelo}'
    output.mkdir(parents=True,exist_ok=True)
    json_write(output/'prediccion.json',result)
    save_preview(image,result,output/'resultado.png')
    print(f"Resultado: {result['label']} ({100*result['score']:.2f}% de puntuacion)")
    if args.esperado: print('Coincide con tu etiqueta:' , 'SI' if result['correct'] else 'NO')
    print(f'Vista: {output / "resultado.png"}')
    print(f'Detalles: {output / "prediccion.json"}')

if __name__=='__main__': main()
