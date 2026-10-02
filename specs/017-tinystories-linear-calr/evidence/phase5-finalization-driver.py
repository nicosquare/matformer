import sys,json,hashlib,shutil,time
from pathlib import Path
repo=Path('/home/ivo.navarrete/ElasticNN/matformer')
root=Path('/nfs-stor/ivo.navarrete/results/elasticnn/tinystories-linear-s1-s2-calr-v1')
source=root/'reports/reporting-source-final-20261002'
source.mkdir(parents=True,exist_ok=False)
records=[]
for directory in ('src','scripts','configs','tests'):
    for path in sorted((repo/directory).rglob('*')):
        if path.is_file() and path.suffix in ('.py','.yaml','.yml','.json') and '__pycache__' not in path.parts:
            relative=path.relative_to(repo);out=source/relative;out.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,out)
            records.append(dict(path=str(relative),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
manifest=dict(purpose='reporting-only; production snapshot unchanged',files=records,
    sha256=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),created_at=time.time())
(source/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
sys.path.insert(0,str(source))
from scripts import run_tinystories_linear_calr as ops
plan=ops.verify_plan(root,gpu=True)
intents=ops.read(root/'launchers/submissions.json');evidence=[]
for intent in intents['jobs']:
    print('Reconciling full terminal',intent['arm_id'],flush=True)
    account=ops.scheduler_history(root,intent)
    intent['scheduler_accounting']=account
    proof=ops.execution_evidence(root,intent,plan)
    intent['execution_evidence']=proof;intent['status']='completed'
    ops.save(root/'reports'/('terminal-'+intent['arm_id']+'.json'),dict(arm_id=intent['arm_id'],job_id=intent['job_id'],scheduler=account,**proof))
    evidence.append(dict(arm_id=intent['arm_id'],job_id=intent['job_id'],scheduler=account,**proof))
if len(evidence)!=4:raise RuntimeError('Four successful full-budget arms required')
ops.save(root/'launchers/submissions.json',intents)
status=ops.read(root/'launchers/status.json')
status.update(production='completed',submission='complete',execution='complete',terminals=dict.fromkeys(ops.ARMS,'complete'),
    monitoring='stopped_at_user_request',automatic_report='stopped_at_user_request',
    campaign_updates=1394112,campaign_training_tokens=11420565504)
ops.save(root/'launchers/status.json',status)
proof=dict(status='complete',production_bindings=plan['bindings'],reporting_source=manifest,arms=evidence,
    campaign_updates=1394112,campaign_training_tokens=11420565504)
ops.save(root/'reports/production-terminal-reconciliation.json',proof)
out=repo/'specs/017-tinystories-linear-calr/evidence/phase5-production-terminals.json'
ops.save(out,proof)
print('T031 terminal reconciliation complete; publishing comparison',flush=True)
code=ops.report(root,source)
print('Report return code',code,flush=True)
if code:raise SystemExit(code)
report=root/'reports/comparison'
target=repo/'specs/017-tinystories-linear-calr/evidence/phase5-final-comparison'
shutil.copytree(report,target)
shutil.copyfile(source/'source-manifest.json',target/'reporting-source-manifest.json')
print(json.dumps(ops.read(report/'comparison_report.json'),indent=2),flush=True)
