"""Fine-tune the last backbone stages without changing the saved data split."""
from __future__ import annotations
import argparse, csv, gc, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .experiment import ROOT, seed_all, seed_worker, json_write, digest_file
from .data import CLASS_NAMES, CoffeeFruitDataset, load_records
from .models import build_model, set_head, get_head, get_transforms, get_model_spec, get_feature_dim
from .head_training import load_head_checkpoint, compute_class_weights, build_head
from .evaluation import evaluate_predictions, save_evaluation
from .ensembles import ensemble_predictions

def load_finetuned(checkpoint_path, device='cpu'):
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if checkpoint['class_labels']!=list(CLASS_NAMES): raise ValueError('Unexpected checkpoint classes')
    name=checkpoint['configuration']['model']
    model=build_model(name,pretrained=False)
    set_head(model,name,build_head(get_feature_dim(name),checkpoint['head_params']))
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    return model.to(device).eval()

def configure_partial(model, name):
    model.requires_grad_(False)
    blocks = list(model.features.children())[-3 if name == 'efficientnet_b0' else -2:]
    for block in blocks: block.requires_grad_(True)
    if name == 'swin_t': model.norm.requires_grad_(True)
    get_head(model, name).requires_grad_(True)
    return blocks

def training_mode(model, name, blocks):
    model.eval()
    for block in blocks: block.train()
    get_head(model, name).train()
    # Small, correlated batches must not rewrite pretrained BN statistics.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm): module.eval()

@torch.inference_mode()
def predict(model, loader, device):
    model.eval(); probabilities=[]
    for x, _ in loader:
        with torch.autocast('cuda', dtype=torch.float16, enabled=device=='cuda'):
            logits=model(x.to(device, non_blocking=True))
        probabilities.append(logits.float().softmax(1).cpu().numpy())
    return np.concatenate(probabilities)

def train_one(name, records, validation, args):
    baseline_dir=args.baseline/name
    baseline=json.loads((baseline_dir/'resultado.json').read_text(encoding='utf-8'))
    out=args.output/name; out.mkdir(parents=True,exist_ok=True)
    if (out/'resultado.json').exists(): return json.loads((out/'resultado.json').read_text(encoding='utf-8'))
    seed_all(42)
    model=build_model(name,pretrained=True)
    set_head(model,name,load_head_checkpoint(baseline['checkpoint_path']))
    blocks=configure_partial(model,name)
    model.to(args.device)
    generator=torch.Generator().manual_seed(42)
    train_loader=DataLoader(CoffeeFruitDataset(records,args.dataset,get_transforms(name,train=True),cache_size=8),
        batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True,
        persistent_workers=args.workers>0,worker_init_fn=seed_worker,generator=generator)
    valid_loader=DataLoader(CoffeeFruitDataset(validation,args.dataset,get_transforms(name),cache_size=8),
        batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True,
        persistent_workers=args.workers>0,worker_init_fn=seed_worker)
    head_ids={id(p) for p in get_head(model,name).parameters()}
    backbone=[p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    optimizer=torch.optim.AdamW([{'params':backbone,'lr':args.backbone_lr},
        {'params':get_head(model,name).parameters(),'lr':args.head_lr}],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-6)
    weights=compute_class_weights(np.array([r['class_index'] for r in records]),.5)
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,device=args.device))
    scaler=torch.amp.GradScaler('cuda',enabled=args.device=='cuda')
    y=[r['class_index'] for r in validation]
    best_prob=predict(model,valid_loader,args.device)
    initial=evaluate_predictions(y,best_prob)
    if abs(initial['macro_f1']-baseline['validation_metrics']['macro_f1'])>1e-4:
        raise RuntimeError('Reconstructed baseline differs from saved predictions.')
    best=initial; best_epoch=0; stale=0; history=[]
    checkpoint=out/'checkpoint.pt'
    config={'model':name,'seed':42,'epochs_max':args.epochs,'patience':args.patience,
        'backbone_lr':args.backbone_lr,'head_lr':args.head_lr,'batch_size':args.batch_size,
        'trainable_backbone_parameters':sum(p.numel() for p in backbone),
        'batchnorm':'frozen_running_statistics','class_weights':weights.tolist(),
        'model_spec':get_model_spec(name),'test_used':False}
    json_write(out/'configuracion.json',config)
    def save(epoch):
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        torch.save({'state_dict':state,'head_params':baseline['best_params'],'configuration':config,
            'epoch':epoch,'validation_metrics':best,'class_labels':list(CLASS_NAMES)},checkpoint)
    save(0)
    for epoch in range(1,args.epochs+1):
        training_mode(model,name,blocks); total=0.; count=0; started=time.monotonic()
        json_write(args.output/'estado.json',{'model':name,'epoch':epoch,'stage':'training','test_reserved':True})
        for step,(x,target) in enumerate(train_loader,1):
            x=x.to(args.device,non_blocking=True); target=target.to(args.device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=args.device=='cuda'):
                loss=criterion(model(x),target)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite fine-tuning loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            scaler.step(optimizer); scaler.update()
            total+=float(loss.detach())*len(x);count+=len(x)
            if step%150==0: print(f'[{name}] epoch {epoch}: {count}/{len(records)} crops',flush=True)
        probabilities=predict(model,valid_loader,args.device)
        metrics=evaluate_predictions(y,probabilities)
        history.append({'epoch':epoch,'train_loss':total/count,'accuracy':metrics['accuracy'],
            'macro_precision':metrics['macro_precision'],'macro_f1':metrics['macro_f1'],
            'seconds':time.monotonic()-started,'backbone_lr':optimizer.param_groups[0]['lr']})
        with (out/'history.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
        print(f"[{name}] epoch {epoch}: accuracy={metrics['accuracy']:.4f}, macro F1={metrics['macro_f1']:.4f}",flush=True)
        if metrics['macro_f1']>best['macro_f1']+1e-5:
            best=metrics;best_epoch=epoch;best_prob=probabilities;stale=0;save(epoch)
        else: stale+=1
        scheduler.step()
        if stale>=args.patience: break
    result_metrics=save_evaluation(out/'validacion',validation,best_prob,model_name=name+'_finetuned')
    result={'model':name,'best_epoch':best_epoch,'epochs_run':len(history),'baseline_metrics':initial,
        'validation_metrics':result_metrics,'macro_f1_change':best['macro_f1']-initial['macro_f1'],
        'accuracy_change':best['accuracy']-initial['accuracy'],'checkpoint':str(checkpoint.resolve()),'test_used':False}
    json_write(out/'resultado.json',result)
    del model,train_loader,valid_loader,optimizer;gc.collect();torch.cuda.empty_cache()
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT/'dataset_preparado')
    p.add_argument('--baseline',type=Path,default=ROOT/'experimentos/transfer_01')
    p.add_argument('--output',type=Path,default=ROOT/'experimentos/finetune_01')
    p.add_argument('--epochs',type=int,default=6);p.add_argument('--patience',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--backbone-lr',type=float,default=1e-5);p.add_argument('--head-lr',type=float,default=3e-5)
    p.add_argument('--device',default='cuda',choices=['cuda','cpu']);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=digest_file(args.dataset/'csv/imagenes.csv')
    source=json.loads((args.baseline/'configuracion.json').read_text(encoding='utf-8'))
    assert manifest==source['dataset_manifest_sha256']
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config['dataset_manifest_sha256']=manifest
    if (args.output/'configuracion.json').exists():
        assert json.loads((args.output/'configuracion.json').read_text(encoding='utf-8'))==config
    json_write(args.output/'configuracion.json',config)
    records=load_records(args.dataset,'train'); validation=load_records(args.dataset,'valid')
    results={name:train_one(name,records,validation,args) for name in ('efficientnet_b0','swin_t')}
    paths=[args.output/name/'validacion/predictions.csv' if name in results else args.baseline/name/'validacion/predictions.csv'
        for name in ('resnet50','efficientnet_b0','densenet121','vit_b_16','swin_t')]
    ensemble=ensemble_predictions(paths,args.output/'ensemble/validacion',model_name='ensemble_finetuned')
    old=json.loads((args.baseline/'ensembles/heterogeneo_cinco/validacion/metrics.json').read_text(encoding='utf-8'))
    json_write(args.output/'comparacion.json',{'models':results,'baseline_ensemble':old,'finetuned_ensemble':ensemble,
        'ensemble_macro_f1_change':ensemble['macro_f1']-old['macro_f1'],
        'ensemble_accuracy_change':ensemble['accuracy']-old['accuracy'],'test_reserved':True})
    json_write(args.output/'estado.json',{'stage':'complete','test_reserved':True})
    print(json.dumps({'ensemble_accuracy':ensemble['accuracy'],'ensemble_macro_f1':ensemble['macro_f1']},indent=2))

if __name__=='__main__': main()
