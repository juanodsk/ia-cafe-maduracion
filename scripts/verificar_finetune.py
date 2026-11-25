"""Check saved fine-tuned checkpoints and before/after prediction alignment."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from cafe_ml.finetune import load_finetuned
from cafe_ml.models import get_transforms
from cafe_ml.data import load_records,CoffeeFruitDataset
from cafe_ml.experiment import digest_file,json_write
from cafe_ml.evaluation import evaluate_predictions

def main():
    torch.set_num_threads(2)
    run=ROOT/'experimentos/finetune_01'
    result=json.loads((run/'comparacion.json').read_text(encoding='utf-8'))
    config=json.loads((run/'configuracion.json').read_text(encoding='utf-8'))
    assert digest_file(ROOT/'dataset_preparado/csv/imagenes.csv')==config['dataset_manifest_sha256']
    assert not (run/'evaluacion_final.json').exists()
    records=load_records(ROOT/'dataset_preparado','valid')
    for name,row in result['models'].items():
        predictions=list(csv.DictReader((run/name/'validacion/predictions.csv').open(encoding='utf-8')))
        assert len(predictions)==len(records)
        for record,pred in zip(records,predictions):
            assert all(str(record[k])==pred[k] for k in ('split','image_id','annotation_id','group_id'))
        probs=np.array([[float(p['prob_'+c]) for c in ('dry','overripe','ripe','semi_ripe','unripe')] for p in predictions])
        metric=evaluate_predictions([r['class_index'] for r in records],probs)
        assert abs(metric['macro_f1']-row['validation_metrics']['macro_f1'])<1e-12
        model=load_finetuned(row['checkpoint'])
        dataset=CoffeeFruitDataset(records[:3],ROOT/'dataset_preparado',get_transforms(name))
        with torch.inference_mode():
            actual=model(torch.stack([dataset[i][0] for i in range(3)])).softmax(1).numpy()
        np.testing.assert_allclose(actual,probs[:3],atol=.01,rtol=.01)
        del model
    audit={'status':'OK','saved_full_models_reloaded':list(result['models']),
        'validation_prediction_alignment':True,'metrics_recomputed':True,
        'same_dataset_manifest':True,'test_reserved':True}
    json_write(run/'verificacion.json',audit)
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
