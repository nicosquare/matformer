import sys,tempfile,yaml
sys.path[:0]=['.','tests']
from pathlib import Path
from pytest import MonkeyPatch
from test_optimizer_ownership_campaign import audited_inputs
from src.evaluation import optimizer_ownership as c
with tempfile.TemporaryDirectory() as d:
 p=Path(d); m=MonkeyPatch(); audited_inputs.__wrapped__(p,m)
 refs=c.inspect_linear_calr_references('/nfs-stor/ivo.navarrete/results/elasticnn')
 old=refs['counterparts']['S1']['resolved_config']
 runs=c.expand_campaign(yaml.safe_load(Path('configs/controlled_exps/tinystories_instruct_linear_calr.yaml').read_text()),prepared_corpus_dir=old['dataset']['prepared_corpus_dir'],tokenizer_dir=old['model']['tokenizer_dir'],run_output_root=p/'runs')
 checks=c.inspect_campaign_models(runs)
 for run,check in zip(runs,checks):
  from src.utils.reproducibility import build_optimizer_ownership_signature
  for config in (run['resolved_config'],run['executable_config']):
   config['model']['tokenizer_name']=old['model']['tokenizer_name']
   contract=config['optimizer_ownership_contract']
   contract['model']['tokenizer_name']=old['model']['tokenizer_name']
   topology=c.stable_hash(dict(parameters=check['parameter_descriptors'],owners=check['owners']))
   contract['parameter_topology_hash']=topology
   contract['clipping']={**run['clipping'],'schema_version':1,'topology_hash':topology}
   config['optimizer_ownership_contract_hash']=build_optimizer_ownership_signature(contract)[0]
  original=refs['counterparts'][run['reference_arm_id']]['resolved_config']
  print(run['arm_id'], c.audit_linear_calr_pair(original,run['resolved_config'],kind='counterpart')['status'])
