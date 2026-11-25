"""USB/IP camera or video -> coffee localization -> saved maturity classifier."""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
from collections import Counter
from datetime import datetime
from .experiment import ROOT
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.cache/ultralytics'))
os.environ.setdefault('YOLO_AUTOINSTALL','false')
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True,exist_ok=True)
import cv2
import numpy as np
from PIL import Image
import torch
from .predict import load_classifier
from .models import MODEL_NAMES,get_transforms
from .data import CLASS_NAMES,crop_bounds

# OpenCV uses BGR. Grey indicates an uncertain prediction.
COLORS={'unripe':(40,190,40),'semi_ripe':(0,210,240),'ripe':(35,35,240),
        'overripe':(180,40,150),'dry':(45,85,130),'unknown':(170,170,170)}
LABELS={'unripe':'Verde','semi_ripe':'Pinton','ripe':'Maduro','overripe':'Sobremaduro',
        'dry':'Seco','unknown':'Incierto'}

class CameraPipeline:
    def __init__(self,detector_path,model_name='efficientnet_b0',device='cuda',
                 detector_conf=.3,classifier_conf=.55,manual=False):
        self.device=device;self.manual=manual;self.detector_conf=detector_conf
        self.classifier_conf=classifier_conf
        self.models=[]
        for name in MODEL_NAMES if model_name=='ensemble' else (model_name,):
            model,_=load_classifier(name,device)
            self.models.append((name,model.eval(),get_transforms(name)))
        self.detector=None
        if not manual:
            if not Path(detector_path).is_file():
                raise FileNotFoundError('Detector sin entrenar. Ejecuta python -m cafe_ml.detector train o usa --manual para un fruto en una zona fija.')
            from ultralytics import YOLO,settings
            settings.update({'sync':False,'hub':False})
            self.detector=YOLO(str(detector_path))

    @torch.inference_mode()
    def process(self,frame,roi=None):
        h,w=frame.shape[:2]
        if self.manual:
            x,y,bw,bh=roi or (int(w*.3),int(h*.2),int(w*.4),int(h*.6))
            boxes=[crop_bounds((x,y,bw,bh),w,h)];detection_scores=[None]
        else:
            result=self.detector.predict(frame,imgsz=640,conf=self.detector_conf,iou=.5,max_det=100,
                device=0 if self.device=='cuda' else 'cpu',verbose=False)[0]
            boxes=[];detection_scores=[]
            for box,score in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy()):
                x1,y1,x2,y2=map(float,box)
                try: bounds=crop_bounds((x1,y1,x2-x1,y2-y1),w,h)
                except ValueError: continue
                boxes.append(bounds);detection_scores.append(float(score))
        annotations=[]
        for start in range(0,len(boxes),32):
            chunk=boxes[start:start+32]
            crops=[Image.fromarray(cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2RGB)) for x1,y1,x2,y2 in chunk]
            probabilities=[]
            for _,model,transform in self.models:
                tensor=torch.stack([transform(im) for im in crops]).to(self.device)
                with torch.autocast('cuda',dtype=torch.float16,enabled=self.device=='cuda'):
                    logits=model(tensor)
                probabilities.append(logits.float().softmax(1).cpu().numpy())
            values=np.mean(probabilities,axis=0)
            for i,(box,p) in enumerate(zip(chunk,values)):
                index=int(p.argmax());score=float(p[index]);label=CLASS_NAMES[index] if score>=self.classifier_conf else 'unknown'
                annotations.append({'bbox':list(box),'class':label,'candidate_class':CLASS_NAMES[index],
                    'score':score,'detector_score':detection_scores[start+i]})
        return annotations

def draw(frame,annotations,fps=0.,manual=False):
    output=frame.copy()
    for ann in annotations:
        x1,y1,x2,y2=ann['bbox'];color=COLORS[ann['class']]
        text=f"{LABELS[ann['class']]} {100*ann['score']:.0f}%"
        cv2.rectangle(output,(x1,y1),(x2,y2),color,2)
        y=max(20,y1-5)
        cv2.putText(output,text,(x1,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(output,text,(x1,y),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1,cv2.LINE_AA)
    title=f"{'ZONA MANUAL' if manual else 'DETECCION AUTOMATICA'} | {fps:.1f} FPS | frutos: {len(annotations)} | Q: salir  S: guardar"
    cv2.rectangle(output,(0,0),(output.shape[1],32),(25,25,25),-1)
    cv2.putText(output,title,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
    if not annotations:
        cv2.putText(output,'No se detectaron frutos',(15,65),cv2.FONT_HERSHEY_SIMPLEX,.65,(220,220,220),2)
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fuente',default='0',help='USB index, video path, or IP/RTSP URL')
    p.add_argument('--imagen',type=Path,help='Process a still image instead of opening a camera')
    p.add_argument('--detector',type=Path,default=ROOT/'experimentos/detector_01/weights/best.pt')
    p.add_argument('--modelo',choices=(*MODEL_NAMES,'ensemble'),default='efficientnet_b0')
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--conf-deteccion',type=float,default=.3)
    p.add_argument('--conf-madurez',type=float,default=.55)
    p.add_argument('--manual',action='store_true',help='Classify one fruit inside a fixed box; no automatic detection')
    p.add_argument('--roi',nargs=4,type=int,metavar=('X','Y','ANCHO','ALTO'))
    p.add_argument('--sin-ventana',action='store_true')
    p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--salida',type=Path,help='Annotated still image output, when using --imagen')
    p.add_argument('--grabar',type=Path,help='Explicitly save annotated video as MP4')
    p.add_argument('--ancho',type=int,default=1280);p.add_argument('--alto',type=int,default=720)
    args=p.parse_args()
    if args.roi and not args.manual: p.error('--roi requiere --manual')
    if not 0<=args.conf_deteccion<=1 or not 0<=args.conf_madurez<=1: p.error('Thresholds must be in [0,1]')
    torch.set_num_threads(4)
    print('Cargando detector y clasificador...',flush=True)
    pipeline=CameraPipeline(args.detector,args.modelo,args.device,args.conf_deteccion,args.conf_madurez,args.manual)
    if args.imagen:
        frame=cv2.imread(str(args.imagen))
        if frame is None: p.error('Cannot read image')
        annotations=pipeline.process(frame,args.roi)
        output=args.salida or ROOT/'pruebas/camara/imagen_resultado.jpg'
        output.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(output),draw(frame,annotations,manual=args.manual)): raise OSError('Cannot save output')
        output.with_suffix('.json').write_text(json.dumps(annotations,indent=2),encoding='utf-8')
        print(f'Resultado: {output}\nFrutos localizados: {len(annotations)}');return
    source=int(args.fuente) if args.fuente.isdecimal() else args.fuente
    cap=cv2.VideoCapture(source,cv2.CAP_DSHOW if os.name=='nt' and isinstance(source,int) else cv2.CAP_ANY)
    if isinstance(source,int) and not cap.isOpened():
        cap.release();cap=cv2.VideoCapture(source)
    writer=None;count=0;last_print=0
    try:
        if not cap.isOpened(): raise RuntimeError('No se pudo abrir la camara. Prueba --fuente 1, cierra otras apps o habilita permisos de camara de Windows.')
        if isinstance(source,int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,args.ancho);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,args.alto)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        print('Camara lista. Q/Esc: salir. S: guardar captura. Ctrl+C: detener consola.',flush=True)
        while True:
            started=time.monotonic()
            ok,frame=cap.read()
            if not ok:
                if isinstance(source,int): raise RuntimeError('La camara dejo de entregar imagenes')
                break
            annotations=pipeline.process(frame,args.roi)
            annotated=draw(frame,annotations,1/max(time.monotonic()-started,.001),args.manual)
            if args.grabar:
                if writer is None:
                    args.grabar.parent.mkdir(parents=True,exist_ok=True)
                    fps=cap.get(cv2.CAP_PROP_FPS)
                    writer=cv2.VideoWriter(str(args.grabar),cv2.VideoWriter_fourcc(*'mp4v'),fps if 1<fps<121 else 20.,(frame.shape[1],frame.shape[0]))
                    if not writer.isOpened(): raise OSError('No se pudo crear el video de salida')
                writer.write(annotated)
            if time.monotonic()-last_print>2:
                print(json.dumps({'frame':count,'frutos':len(annotations),'madurez':dict(Counter(a['class'] for a in annotations))}),flush=True)
                last_print=time.monotonic()
            if not args.sin_ventana:
                cv2.imshow('Madurez del cafe',annotated)
                key=cv2.waitKey(1)&255
                if key in (27,ord('q')) or cv2.getWindowProperty('Madurez del cafe',cv2.WND_PROP_VISIBLE)<1: break
                if key==ord('s'):
                    target=ROOT/'pruebas/camara'/f'captura_{datetime.now():%Y%m%d_%H%M%S}.jpg'
                    target.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(target),annotated);print(f'Captura: {target}')
            count+=1
            if args.max_frames and count>=args.max_frames: break
    except KeyboardInterrupt: print('Camara detenida.')
    finally:
        cap.release()
        if writer is not None: writer.release()
        if not args.sin_ventana: cv2.destroyAllWindows()
if __name__=='__main__':main()
