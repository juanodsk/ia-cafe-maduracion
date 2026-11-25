import unittest
import torch
from torch import nn
from cafe_ml.models import build_model, get_head
from cafe_ml.finetune import configure_partial, training_mode

class FineTuneTests(unittest.TestCase):
    def test_partial_updates_preserve_stem_and_batchnorm(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model=build_model('efficientnet_b0',pretrained=False)
        blocks=configure_partial(model,'efficientnet_b0')
        training_mode(model,'efficientnet_b0',blocks)
        self.assertFalse(any(p.requires_grad for p in model.features[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.features[-1].parameters()))
        self.assertTrue(all(p.requires_grad for p in get_head(model).parameters()))
        bns=[m for m in model.modules() if isinstance(m,nn.BatchNorm2d)]
        self.assertTrue(all(not m.training for m in bns))
        old_stem=model.features[0][0].weight.detach().clone()
        old_last=model.features[-1][0].weight.detach().clone()
        old_stats=[m.running_mean.clone() for m in bns]
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
        loss=nn.CrossEntropyLoss()(model(torch.randn(2,3,224,224)),torch.tensor([0,4]))
        loss.backward();optimizer.step()
        self.assertTrue(torch.equal(old_stem,model.features[0][0].weight))
        self.assertFalse(torch.equal(old_last,model.features[-1][0].weight))
        self.assertTrue(all(torch.equal(before,m.running_mean) for before,m in zip(old_stats,bns)))

    def test_swin_only_last_stage_and_norm_trainable(self):
        model=build_model('swin_t',pretrained=False)
        configure_partial(model,'swin_t')
        self.assertFalse(any(p.requires_grad for p in model.features[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.features[-1].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.norm.parameters()))

if __name__=='__main__': unittest.main()
