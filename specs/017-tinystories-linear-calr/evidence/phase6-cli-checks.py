import subprocess,tempfile,json,os
from pathlib import Path
repo=Path.cwd(); py='/home/ivo.navarrete/.conda/envs/elasticnn/bin/python';env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
inputs=json.loads(Path('/nfs-stor/ivo.navarrete/results/elasticnn/tinystories-linear-s1-s2-calr-v1/diagnostics/inputs.json').read_text());records=[]
with tempfile.TemporaryDirectory(prefix='calr-phase6-cli-') as d:
 root=Path(d)/'campaign'
 commands=[([py,'scripts/preflight_tinystories_linear_calr.py','snapshot','--campaign-root',str(root),'--prepared-corpus-dir',inputs['prepared_corpus_dir'],'--tokenizer-dir',inputs['tokenizer_dir'],'--reference-root',inputs['reference_root']],0),([py,str(root/'source/scripts/preflight_tinystories_linear_calr.py'),'snapshot','--campaign-root',str(root)],0),([py,str(root/'source/scripts/run_tinystories_linear_calr.py'),'prepare','--campaign-root',str(root)],1)]
 for cmd,expected in commands:
  r=subprocess.run(cmd,env=env,text=True,capture_output=True);records.append(dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert (r.returncode==0)==(expected==0),r.stderr
 assert not (root/'launchers/submissions.json').exists()
Path('specs/017-tinystories-linear-calr/evidence/phase6-cli-checks.json').write_text(json.dumps(dict(status='passed',records=records,limitations='Actual CLI snapshot/idempotency and missing-CPU rejection in a temporary root; successful CPU/prepare/diagnostic/queue/report paths validated by existing temporary-root fixtures with mocked Slurm, not GPU submission.'),indent=2)+'\n')
