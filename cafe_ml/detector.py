"""Train a one-class coffee-fruit localizer using the fixed train/valid COCO split."""
import argparse,csv,json,os,shutil
from pathlib import Path
from .experiment import ROOT,json_write,digest_file
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.cache/ultralytics'))
os.environ.setdefault('YOLO_AUTOINSTALL','false')
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True,exist_ok=True)

def yolo_box(box,width,height):
    x,y,w,h=map(float,box)
    left=max(0.,x);top=max(0.,y);right=min(float(width),x+w);bottom=min(float(height),y+h)
    if right<=left or bottom<=top: raise ValueError('Box outside image')
    return ((left+right)/(2*width),(top+bottom)/(2*height),(right-left)/width,(bottom-top)/height)

def prepare(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    identity={s:digest_file(source/s/'_annotations.coco.json') for s in ('train','valid')}
    summary_path=output/'preparacion.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text(encoding='utf-8'))
        if summary['source_sha256']!=identity: raise ValueError('Detector dataset changed')
        return output/'data.yaml'
    rows=[];counts={}
    for split in ('train','valid'):
        coco=json.loads((source/split/'_annotations.coco.json').read_text(encoding='utf-8'))
        by_image={im['id']:[] for im in coco['images']}
        for ann in coco['annotations']: by_image[ann['image_id']].append(ann)
        images=output/'images'/split;labels=output/'labels'/split
        images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
        for im in coco['images']:
            file=im['file_name'];shutil.copy2(source/split/file,images/file)
            lines=[]
            for ann in by_image[im['id']]:
                box=yolo_box(ann['bbox'],im['width'],im['height'])
                lines.append('0 '+' '.join(f'{v:.8f}' for v in box))
            (labels/(Path(file).stem+'.txt')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
            rows.append({'split':split,'image_id':im['id'],'file_name':file,'fruit_boxes':len(lines)})
        counts[split]={'images':len(coco['images']),'boxes':len(coco['annotations'])}
    (output/'data.yaml').write_text(f'path: "{output.as_posix()}"\ntrain: images/train\nval: images/valid\nnames:\n  0: fruit\n',encoding='utf-8')
    with (output/'distribucion.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    json_write(summary_path,{'source_sha256':identity,'counts':counts,'test_used':False,
        'purpose':'One fruit localization class; maturity remains classified by the five project classifiers.'})
    return output/'data.yaml'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','download','train'))
    p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    data=ROOT/'dataset_deteccion';run=ROOT/'experimentos/detector_01'
    if args.phase=='prepare': print(prepare(ROOT/'dataset_preparado',data));return
    from ultralytics import YOLO,settings
    settings.update({'sync':False,'hub':False})
    weights=ROOT/'.cache/torch/yolo11n.pt'
    if args.phase=='download':
        weights.parent.mkdir(parents=True,exist_ok=True)
        model=YOLO(str(weights))
        print('Detector initialization ready:',weights);return
    yaml=prepare(ROOT/'dataset_preparado',data)
    if not weights.exists(): raise FileNotFoundError('Download first: python -m cafe_ml.detector download')
    checkpoint=run/'weights/last.pt'
    model=YOLO(str(checkpoint if args.resume else weights))
    if args.resume:
        results=model.train(resume=True)
    else:
        if checkpoint.exists(): raise FileExistsError('Saved training exists; use --resume')
        results=model.train(data=str(yaml),epochs=args.epochs,imgsz=640,batch=args.batch,
            workers=args.workers,device=0,project=str(run.parent),name=run.name,exist_ok=True,
            seed=42,deterministic=True,optimizer='AdamW',lr0=.001,lrf=.01,patience=5,
            freeze=10,amp=True,cache=False,plots=True,save=True,max_det=200,
            hsv_h=0.,hsv_s=0.,hsv_v=0.,mosaic=0.,mixup=0.,close_mosaic=0,
            fliplr=.5,translate=.05,scale=.2)
    json_write(run/'resumen.json',{'stage':'complete','task':'coffee_fruit_localization',
        'best_checkpoint':str(run/'weights/best.pt'),'metrics':results.results_dict,
        'test_used':False,'class_names':{'0':'fruit'},'epochs_requested':args.epochs})
    print(json.dumps(results.results_dict,indent=2))
if __name__=='__main__':main()
