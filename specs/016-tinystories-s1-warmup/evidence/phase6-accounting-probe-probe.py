import os,socket,shutil,subprocess,json
from pathlib import Path
record={'host':socket.gethostname(),'environment':{k:os.environ.get(k) for k in ('SLURM_CONF','SLURM_CONF_SERVER','SLURM_JOB_ID','PATH')},'sacct_executable':shutil.which('sacct'),'checks':[]}
cmd=['sacct','-X','--noheader','--parsable2','--user=ivo.navarrete','--starttime=2026-09-22','--name=s1w-diagnostic-a2-450634f8','--format=JobIDRaw,JobName%100,State,ElapsedRaw,ExitCode']
for name,env in [('inherited',dict(os.environ)),('explicit_client',dict(os.environ,SLURM_CONF='/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1/launchers/slurm-client.conf'))]:
 try:
  r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=30);record['checks'].append(dict(mode=name,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
 except Exception as error:record['checks'].append(dict(mode=name,error=repr(error)))
Path(__file__).with_name('result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
