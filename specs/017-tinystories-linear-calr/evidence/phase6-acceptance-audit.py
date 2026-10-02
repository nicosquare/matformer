import csv,json,hashlib,math,collections
from pathlib import Path
P=Path('/home/ivo.navarrete/ElasticNN/matformer/specs/017-tinystories-linear-calr/evidence')
D=P/'phase5-final-comparison'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rows(n):
 with (D/(n+'.csv')).open() as f:return list(csv.DictReader(f))
m=read(D/'plot_sources.json');t=read(P/'phase5-production-terminals.json');report=read(D/'comparison_report.json')
assert all(report[k]=='complete' for k in ['new_terminals','references','trajectories','comparison'])
e=rows('endpoints');pairs=rows('paired_differences');inter=rows('interactions');supp=rows('supplemental_004_differences');by={r['endpoint_id']:r for r in e}
assert len(e)==len(by)==36 and len(pairs)==48 and len(inter)==4 and len(supp)==16
for r in e:assert math.isclose(math.exp(float(r['loss'])),float(r['perplexity']),rel_tol=1e-10)
for r in pairs+supp:
 a,b=by[r['left_endpoint_id']],by[r['right_endpoint_id']];delta=float(a['loss'])-float(b['loss']);ratio=float(a['perplexity'])/float(b['perplexity'])
 assert math.isclose(delta,float(r['delta_loss']),abs_tol=1e-12)
 assert math.isclose(ratio,float(r['perplexity_ratio']),rel_tol=1e-10)
 assert math.isclose(ratio-1,float(r['relative_gap']),abs_tol=1e-10)
for r in inter:
 loss=lambda k:float(by[r[k+'_endpoint_id']]['loss'])
 assert math.isclose(loss('S2_CaLR')-loss('S2_poly')-loss('S1_CaLR')+loss('S1_poly'),float(r['interaction_loss']),abs_tol=1e-12)
new=[r for r in e if r['historical_reference']=='false'];assert len(new)==16
assert {r['width'] for r in new}=={'g250','g500','g750','g1000'}
for r in new:
 assert int(r['actual_updates'])==348528 and int(r['actual_tokens'])==2855141376
 counts=json.loads(r['width_selection_counts']);assert sum(counts.values())==348528
 assert json.loads(r['applied_schedule_watermark'])['step']==348528
assert t['campaign_updates']==1394112 and t['campaign_training_tokens']==11420565504
for a in t['arms']:
 assert a['scheduler']['state']=='COMPLETED' and a['scheduler']['exit_code']=='0:0'
 assert a['resources']['measurement_complete'] and not a['resources']['incomplete_attempts'] and not a['resources']['unobserved_attempts']
 assert a['resources']['failed_attempt_seconds']==0
outputs={n:sha(D/n) for n in m['output_sha256']}
assert outputs==m['output_sha256'];assert len([n for n in outputs if n.endswith(('.png','.pdf'))])==16
sources={}
for r in m['sources']+[m['code_source']]:sources[r['path']]=r['sha256']
for a in t['arms']:
 for r in a['sources']+a['terminal']['sources']:sources[r['path']]=r['sha256']
checked=[]
for path,expected in sources.items():
 assert sha(Path(path))==expected,path
 checked.append(dict(path=path,sha256=expected))
from scripts import run_tinystories_linear_calr as ops
repo=Path('/home/ivo.navarrete/ElasticNN/matformer');files={k:v for k,v in ops.source_files(repo).items() if k.split('/')[0] in ('src','scripts','configs','tests') or k in ('train.py','pyproject.toml','requirements.txt','build_backend.py')}
result=dict(status='passed',endpoint_count=len(e),required_pair_count=len(pairs),pair_families=dict(collections.Counter(r['family'] for r in pairs)),interaction_count=len(inter),supplemental_pair_count=len(supp),figure_count=16,campaign_updates=t['campaign_updates'],campaign_training_tokens=t['campaign_training_tokens'],output_sha256=outputs,checked_sources=checked,production_bindings=t['production_bindings'],tested_source_sha256=ops.stable_hash(files),tested_source_files=files,recipe_sha256=sha(repo/'configs/controlled_exps/tinystories_instruct_linear_calr.yaml'),limits='Read-only hash/arithmetic/resource-watermark audit of saved validated evidence; no new GPU run, scheduler query or training; prior whole-bundle and full-trace validation retained in phase5 evidence.')
(P/'phase6-acceptance-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['checked_sources','tested_source_files','output_sha256','production_bindings']}))
