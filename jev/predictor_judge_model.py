"""Native visual Noul safety judgment conditioned on current state and unexecuted commands."""
import json
from pathlib import Path
import torch
from torch import nn
from .visual_model import VisualDecisionModel, last_valid_hidden


class PredictorJudgeModel(VisualDecisionModel):
    def __init__(self, backbone, processor, head, model_config):
        super().__init__(backbone,processor,head,model_config)
        width=head.in_features;device=head.weight.device
        self.state_projection=nn.Sequential(nn.Linear(model_config.get('state_dim',16),128),nn.SiLU(),nn.Linear(128,width)).to(device)
        self.action_projection=nn.Sequential(nn.Linear(10,128),nn.SiLU(),nn.Linear(128,width)).to(device)
        self.register_buffer('state_mean',torch.zeros(model_config.get('state_dim',16),device=device))
        self.register_buffer('state_std',torch.ones_like(self.state_mean))
        self.register_buffer('action_mean',torch.zeros(8,device=device))
        self.register_buffer('action_std',torch.ones_like(self.action_mean))

    @classmethod
    def from_pretrained(cls,model_id,revision,*,history_frames=3,state_dim=16,max_actions=8,**kwargs):
        base=VisualDecisionModel.from_pretrained(model_id,revision,**kwargs)
        cfg={**base.model_config,'method':'native_qwen_action_conditioned_noul',
             'history_frames':history_frames,'state_dim':state_dim,'max_actions':max_actions}
        return cls(base.backbone,base.processor,base.head,cfg)

    def set_normalization(self,state_mean,state_std,action_mean,action_std):
        for name,values in [('state_mean',state_mean),('state_std',state_std),('action_mean',action_mean),('action_std',action_std)]:
            buffer=getattr(self,name);value=torch.as_tensor(values,device=buffer.device,dtype=buffer.dtype)
            if value.shape!=buffer.shape or not torch.isfinite(value).all() or (name.endswith('std') and (value<=0).any()):
                raise ValueError('Invalid '+name)
            buffer.copy_(value)

    def prepare_predictor_judge_inputs(self,questions,observations,robot_state,remaining_actions,action_mask,action_dt_s,history_mask,constraint_context=None):
        from PIL import Image
        b=len(questions);f=self.model_config['history_frames'];h=self.model_config['max_actions'];device=self.head.weight.device
        if b<1 or any(not isinstance(q,str) or not q.strip() for q in questions):raise ValueError('Nonempty NL questions required')
        if set(observations)!=set(self.camera_names):raise ValueError('Missing camera view')
        for images in observations.values():
            if images.dtype!=torch.uint8 or images.ndim!=5 or images.shape[:3]!=(b,f,3):raise ValueError('Expected B,F,3,H,W uint8 history')
        state=robot_state.to(device=device,dtype=torch.float32);actions=remaining_actions.to(device=device,dtype=torch.float32)
        mask=action_mask.to(device);hm=history_mask.to(device);dt=action_dt_s.to(device=device,dtype=torch.float32).reshape(-1)
        if state.shape!=(b,self.model_config['state_dim']) or actions.shape!=(b,h,8):raise ValueError('State/action shape mismatch')
        if mask.dtype!=torch.bool or mask.shape!=(b,h) or hm.dtype!=torch.bool or hm.shape!=(b,f):raise ValueError('Boolean action/history masks required')
        count=mask.sum(-1)
        if (count==0).any() or not torch.equal(mask,torch.arange(h,device=device)[None]<count[:,None]):raise ValueError('Actions must form a nonempty valid prefix')
        hc=hm.sum(-1)
        if (hc==0).any() or not torch.equal(hm,torch.arange(f,device=device)[None]>=f-hc[:,None]):raise ValueError('History must be an adjacent suffix ending now')
        if dt.shape!=(b,) or not torch.isfinite(dt).all() or (dt<=0).any():raise ValueError('Positive per-command interval required')
        if not torch.isfinite(state).all() or not torch.isfinite(actions[mask]).all():raise ValueError('Nonfinite valid numeric input')
        contexts=constraint_context or [{'source':'none','text':''} for _ in range(b)]
        if len(contexts)!=b:raise ValueError('Context batch mismatch')
        for c in contexts:
            if set(c)!={'source','text'} or c['source'] not in ('none','observed_history') or not isinstance(c['text'],str):raise ValueError('Only causal observed context is allowed; oracle is supervision')
            if c['source']=='none' and c['text']:raise ValueError('Absent context cannot contain text')
        marker=self.processor.tokenizer.pad_token
        if not marker or any(marker in q or marker in c['text'] for q,c in zip(questions,contexts)):raise ValueError('Reserved numeric marker in text')
        images=[];prompts=[]
        for i,q in enumerate(questions):
            content=[]
            for t in range(f):
                if not hm[i,t]:continue
                for camera in self.camera_names:
                    im=Image.fromarray(observations[camera][i,t].detach().cpu().permute(1,2,0).numpy())
                    images.append(im);content.extend([{'type':'text','text':f'{camera}, relative step {t-f+1}, dt={float(dt[i]):.6g}s:'},{'type':'image','image':im}])
            text=(f'{q}\nPredict a NEW violation during the {int(count[i])} valid remaining commands. '
                  f'Command interval: {float(dt[i]):.6g} seconds.\n')
            if contexts[i]['text']:text+='Observed context: '+contexts[i]['text']+'\n'
            text+='Current robot state token: '+marker+'\nRemaining command tokens: '+' '.join([marker]*h)+'\nAnswer yes or no.'
            content.append({'type':'text','text':text})
            prompts.append(self.processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False))
        encoded=self.processor(text=prompts,images=images,padding=True,return_tensors='pt')
        if 'pixel_values' not in encoded:raise ValueError('Missing native visual inputs')
        encoded={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in encoded.items()}
        ids=encoded.pop('input_ids');attention=encoded['attention_mask'].clone()
        slots=(ids==self.processor.tokenizer.pad_token_id)&attention.bool()
        if not torch.all(slots.sum(-1)==h+1):raise ValueError('Numeric marker tokenization is not one token per slot')
        # The padding ID reserves slots only; learned continuous embeddings replace it.
        embeds=self.backbone.get_input_embeddings()(ids).clone()
        clean=torch.where(mask.unsqueeze(-1),actions,torch.zeros_like(actions))
        normalized=(clean-self.action_mean)/self.action_std
        normalized=torch.where(mask.unsqueeze(-1),normalized,torch.zeros_like(normalized))
        times=torch.arange(h,device=device)[None]*dt[:,None]
        features=torch.cat([normalized,times[:,:,None],dt[:,None,None].expand(-1,h,1)],-1)
        action_tokens=self.action_projection(features)
        state_tokens=self.state_projection((state-self.state_mean)/self.state_std)
        action_tokens=torch.where(mask.unsqueeze(-1),action_tokens,torch.zeros_like(action_tokens))
        for i in range(b):
            ix=slots[i].nonzero().flatten()
            embeds[i,ix[0]]=state_tokens[i].to(embeds.dtype)
            embeds[i,ix[1:]]=action_tokens[i].to(embeds.dtype)
            attention[i,ix[1:]]=mask[i].to(attention.dtype)
        if ids.shape[1]>self.model_config['max_length']:raise ValueError('PredictorJudge input exceeds token budget')
        position_ids,_=self.backbone.get_rope_index(ids,image_grid_thw=encoded.get('image_grid_thw'),
            attention_mask=attention,mm_token_type_ids=encoded.get('mm_token_type_ids'))
        encoded.update(inputs_embeds=embeds,attention_mask=attention,position_ids=position_ids)
        self.last_input_tokens=int(attention.sum());self.last_sequence_length=ids.shape[1]
        return encoded

    def forward(self,questions,observations,robot_state,remaining_actions,action_mask,action_dt_s,history_mask,constraint_context=None):
        encoded=self.prepare_predictor_judge_inputs(questions,observations,robot_state,remaining_actions,action_mask,action_dt_s,history_mask,constraint_context)
        output=self.backbone(**encoded,use_cache=False,return_dict=True)
        score=self.head(last_valid_hidden(output.last_hidden_state,encoded['attention_mask']).float()).squeeze(-1)
        return torch.stack([torch.zeros_like(score),score],-1)

    @torch.inference_mode()
    def probabilities(self,**inputs):
        self.eval();return (self(**inputs)/self.temperature).softmax(-1)

    def save(self,output):
        super().save(output)
        torch.save({'state_projection':self.state_projection.state_dict(),'action_projection':self.action_projection.state_dict(),
                    'normalization':{k:getattr(self,k).detach().cpu() for k in ['state_mean','state_std','action_mean','action_std']}},Path(output)/'predictor_judge.pt')

    @classmethod
    def load(cls,output,*,device='cuda:0',trainable=False):
        from transformers import AutoProcessor
        path=Path(output);cfg=json.loads((path/'model.json').read_text())
        if cfg.get('method')!='native_qwen_action_conditioned_noul':raise ValueError('Not an action-conditioned predictor judge checkpoint')
        model=cls.from_pretrained(cfg['model_id'],cfg['revision'],device=device,dtype=cfg['dtype'],lora_rank=0,
            max_length=cfg['max_length'],min_pixels=cfg['min_pixels'],max_pixels=cfg['max_pixels'],gradient_checkpointing=False,
            history_frames=cfg['history_frames'],state_dim=cfg['state_dim'],max_actions=cfg['max_actions'])
        model.processor=AutoProcessor.from_pretrained(path/'processor');model.processor.tokenizer.padding_side='right'
        if cfg['lora_rank']:
            from peft import PeftModel
            model.backbone.language_model=PeftModel.from_pretrained(model.backbone.language_model,path/'adapter',is_trainable=trainable)
        model.head.load_state_dict(torch.load(path/'head.pt',map_location=device,weights_only=True))
        payload=torch.load(path/'predictor_judge.pt',map_location=device,weights_only=True)
        for name in ['state_projection','action_projection']:getattr(model,name).load_state_dict(payload[name])
        model.set_normalization(**payload['normalization']);model.model_config=cfg;model.set_temperature(cfg.get('temperature',1.))
        if trainable:model._gradient_checkpointing(cfg['gradient_checkpointing']);return model.train()
        return model.eval()
