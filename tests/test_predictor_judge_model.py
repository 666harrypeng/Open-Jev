import unittest
from types import SimpleNamespace
import torch
from torch import nn
from jev.predictor_judge_model import PredictorJudgeModel

class Processor:
    def __init__(self):
        self.tokenizer=SimpleNamespace(pad_token='[PAD]',pad_token_id=0,padding_side='right')
        self.contents=[]
    def apply_chat_template(self,messages,**kwargs):
        self.contents.append(messages[0]['content']);return 'fixture'
    def __call__(self,text,images,**kwargs):
        self.images=images
        b=len(text)
        return {'input_ids':torch.tensor([[1]+[0]*9+[2,0]]*b),
                'attention_mask':torch.tensor([[1]*11+[0]]*b),
                'pixel_values':torch.tensor([sum(im.getpixel((0,0))) for im in images],dtype=torch.float32),
                'image_grid_thw':torch.ones(len(images),3,dtype=torch.long),
                'mm_token_type_ids':torch.zeros(b,12,dtype=torch.long)}
class Backbone(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(3,8);self.visual=nn.Linear(1,1);self.language_model=nn.Linear(8,8)
    def get_input_embeddings(self):return self.embedding
    def get_rope_index(self,input_ids,**kwargs):return torch.arange(input_ids.shape[1])[None,None].expand(3,len(input_ids),-1),None
    def forward(self,inputs_embeds,attention_mask,pixel_values,**kwargs):
        x=(inputs_embeds*attention_mask.unsqueeze(-1)).cumsum(1)
        return SimpleNamespace(last_hidden_state=self.language_model(x)+pixel_values.mean()/765)
class PredictorJudgeModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model=PredictorJudgeModel(Backbone(),Processor(),nn.Linear(8,1),
            {'max_length':128,'history_frames':3,'state_dim':16,'max_actions':8,'camera_names':['overview','wrist'],'lora_rank':0})
        self.inputs=dict(questions=['Will it tilt?'],observations={c:torch.zeros(1,3,3,8,8,dtype=torch.uint8) for c in ['overview','wrist']},
            robot_state=torch.zeros(1,16),remaining_actions=torch.ones(1,8,8),action_mask=torch.tensor([[1,1,1,0,0,0,0,0]],dtype=torch.bool),
            action_dt_s=torch.tensor([.05]),history_mask=torch.tensor([[0,1,1]],dtype=torch.bool),constraint_context=[{'source':'none','text':''}])
    def test_prepared_cpu_inputs_match_raw_path_and_keep_projection_gradients(self):
        from jev.predictor_judge_model import encode_predictor_judge_visual_inputs
        raw=self.model(**self.inputs)
        visual={k:self.inputs[k] for k in ['questions','observations','action_mask','action_dt_s','history_mask','constraint_context']}
        encoded=encode_predictor_judge_visual_inputs(self.model.processor,self.model.model_config,**visual)
        numbers={k:self.inputs[k] for k in ['robot_state','remaining_actions','action_mask','action_dt_s','history_mask']}
        result=self.model(encoded=encoded,**numbers)
        torch.testing.assert_close(raw,result,rtol=0,atol=0)
        result.sum().backward()
        self.assertGreater(self.model.action_projection[0].weight.grad.abs().sum(),0)
        self.assertTrue(all(not x.is_cuda for x in encoded.values() if isinstance(x,torch.Tensor)))

    def test_padding_is_masked_and_does_not_change_logits(self):
        expected=self.model(**self.inputs);x=self.inputs['remaining_actions'].clone();x[:,3:]=float('nan')
        changed=self.model(**{**self.inputs,'remaining_actions':x});torch.testing.assert_close(expected,changed)
        self.assertEqual(expected.shape,(1,2));self.assertEqual(expected[0,0],0)
    def test_valid_action_and_state_affect_output_and_receive_gradients(self):
        baseline=self.model(**self.inputs)
        changed=self.model(**{**self.inputs,'remaining_actions':self.inputs['remaining_actions']*2,'robot_state':torch.ones(1,16)})
        self.assertFalse(torch.allclose(baseline,changed));changed.sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in self.model.action_projection.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in self.model.state_projection.parameters()))
    def test_history_has_only_adjacent_available_views(self):
        self.inputs['observations']['overview'][:,0]=255
        self.model(**self.inputs)
        self.assertEqual(len(self.model.processor.images),4)
        self.assertTrue(all(im.getpixel((0,0))==(0,0,0) for im in self.model.processor.images))
        content=str(self.model.processor.contents[-1]);self.assertIn('step -1',content);self.assertIn('step 0',content)
    def test_mask_must_be_nonempty_contiguous_prefix(self):
        for mask in [[0]*8,[1,0,1,0,0,0,0,0]]:
            with self.assertRaises(ValueError):self.model(**{**self.inputs,'action_mask':torch.tensor([mask],dtype=torch.bool)})
    def test_oracle_context_is_not_allowed(self):
        with self.assertRaises(ValueError):self.model(**{**self.inputs,'constraint_context':[{'source':'oracle','text':'closed=true'}]})
    def test_each_remaining_length_runs(self):
        for h in range(1,9):
            out=self.model(**{**self.inputs,'action_mask':(torch.arange(8)<h)[None]})
            self.assertTrue(torch.isfinite(out).all())
