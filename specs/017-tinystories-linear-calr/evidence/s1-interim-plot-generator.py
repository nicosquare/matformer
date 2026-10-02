"""Read-only terminal admission and S1-only interim plots from frozen helpers."""
import sys,json,csv,math,time
from pathlib import Path
root=Path('/nfs-stor/ivo.navarrete/results/elasticnn/tinystories-linear-s1-s2-calr-v1')
sys.path.insert(0,str(root/'source'))
from scripts import run_tinystories_linear_calr as ops
from src.evaluation import optimizer_ownership as c
# Use the corrected report admission helper without changing frozen sources.
import ast
report_source=Path('/home/ivo.navarrete/ElasticNN/matformer/src/evaluation/optimizer_ownership.py').read_text()
node=next(n for n in ast.parse(report_source).body if isinstance(n,ast.FunctionDef) and n.name=='_linear_calr_native_terminal')
exec(ast.get_source_segment(report_source,node),c.__dict__)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
output=Path('/home/ivo.navarrete/ElasticNN/matformer/specs/017-tinystories-linear-calr/evidence/s1-interim-20261002')
plan=ops.verify_plan(root,gpu=True)
manifest=c._read_preflight_manifest(root/'campaign/campaign_manifest.json')
base=Path('/nfs-stor/ivo.navarrete/results/elasticnn')
refmanifest=c._read_preflight_manifest(base/'optimizer-ownership-v1/campaign/campaign_manifest.json')
terminals=[];execution=[];sources=[c._source_record(root/'campaign/campaign_manifest.json'),c._source_record(base/'optimizer-ownership-v1/campaign/campaign_manifest.json')]
# Plot admission validates the saved runtime config; the queue continuation
# helper reconstructs a config whose comparison signature differs post-run.
# Keep full native model, optimizer, RNG, terminal, and trace validation.
admitted={}
def saved_config_completion(root_path,arm):
    definition=next(r for r in manifest['runs'] if r['arm_id']==arm)
    terminal=c._linear_calr_native_terminal(root_path/'runs'/arm,definition,manifest['expected_traces'][arm])
    admitted[arm]=terminal
    return dict(mode='completion_only',checkpoint=dict(sha256=terminal['checkpoint_sha256']))
ops.continuation=saved_config_completion
intents=ops.read(root/'launchers/submissions.json')['jobs']
for arm in ('S1-linear-poly','S1-linear-CaLR'):
    print('Validating new terminal',arm,flush=True)
    intent=dict(next(j for j in intents if j['arm_id']==arm));account=ops.scheduler_history(root,intent)
    if not account or account['state']!='COMPLETED' or account['exit_code']!='0:0':raise RuntimeError('S1 job not completed successfully')
    intent['scheduler_accounting']=account
    proof=ops.execution_evidence(root,intent,plan);execution.append(dict(arm_id=arm,job_id=intent['job_id'],scheduler=account,resources=proof['resources']))
    sources.extend(proof['sources'])
    definition=next(r for r in manifest['runs'] if r['arm_id']==arm)
    terminal=admitted[arm]
    terminal.update(series='S1-carl' if arm.endswith('CaLR') else 'S1-polynomial',peak_lr=.008,historical_reference=False)
    terminals.append(terminal);sources.extend(terminal['sources'])
    print('Admitted',arm,flush=True)
for arm in ('S1',*('ST-'+w for w in c.WIDTH_LABELS)):
    print('Validating reference',arm,flush=True)
    definition=next(r for r in refmanifest['runs'] if r['arm_id']==arm)
    terminal=c._linear_calr_native_terminal(base/'optimizer-ownership-v1/runs'/arm,definition,refmanifest['expected_traces'][arm])
    terminal.update(series='S1-cosine' if arm=='S1' else 'standalone-cosine',peak_lr=.008,historical_reference=True)
    terminals.append(terminal);sources.extend(terminal['sources'])
reference=c.LINEAR_CALR_REFERENCES['S1-cosine-004']
terminal=c._linear_calr_legacy_terminal(base/reference['path'],reference,next(r for r in refmanifest['runs'] if r['arm_id']=='S1'))
terminal.update(series='S1-cosine',peak_lr=.004,historical_reference=True)
terminals.append(terminal);sources.extend(terminal['sources'])
rows=[r for t in terminals for r in c._linear_calr_endpoint_rows(t)]
if len(rows)!=20:raise RuntimeError('Expected 20 S1-only endpoints including supplemental .004')
progress=[]
for terminal in terminals:
    print('Reading measured curves',terminal['run_id'],flush=True)
    p,lr,notes=c.linear_calr_measured_trajectories(terminal)
    if notes['missing_validation']:raise RuntimeError('Missing measured validation trajectory')
    progress.extend(p);sources.extend(notes['sources'])
labels=('standalone-cosine','S1-cosine','S1-polynomial','S1-carl')
names={'standalone-cosine':'Standalone cosine','S1-cosine':'S1 cosine baseline','S1-polynomial':'S1 polynomial','S1-carl':'S1 CaLR'}
def selected(peak):return [r for r in rows if r['peak_lr']==(peak if r['series']=='S1-cosine' else .008)]
def caption(peak):return f'Seed 42; warmup 64. S1 cosine peak {peak:g}; new arms and standalone peak .008.\nS1: 348,528 updates / 2,855,141,376 training tokens; standalone: 87,132 updates / 713,785,344 tokens.'
def write(stage,final):
    with (stage/'endpoints.csv').open('w',newline='') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row));writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:c.endpoint_csv_value(v) for k,v in row.items()} for row in rows)
    c.write_json_artifact(stage/'endpoints.json',dict(scope='S1-only interim; full campaign incomplete',endpoints=rows))
    files=[]
    for peak,panel in ((.008,'primary'),(.004,'supplemental_004')):
        current=selected(peak)
        fig,axes=plt.subplots(1,2,figsize=(13,5.5))
        for ax,metric in zip(axes,('loss','perplexity')):
            for series in labels:
                subset=sorted((r for r in current if r['series']==series),key=lambda r:r['non_embedding_parameters'])
                x=[r['non_embedding_parameters'] for r in subset];y=[r[metric] for r in subset];color,marker=c.LINEAR_CALR_STYLES[series]
                if series=='standalone-cosine':ax.scatter(x,y,color=color,marker=marker,s=65,label=names[series],zorder=4)
                else:ax.plot(x,y,color=color,marker=marker,linewidth=2,label=names[series])
            ax.set(xlabel='Active non-embedding parameters',ylabel='Terminal ordinary-validation '+metric,title=metric.capitalize())
            ax.set_xticks([w['non_embedding_parameters'] for w in c.WIDTHS],c.WIDTH_LABELS);ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('Completed S1 runs versus cosine baseline and standalone')
        fig.text(.5,.015,caption(peak),ha='center',fontsize=8);fig.tight_layout(rect=(0,.09,1,.94))
        for suffix in ('png','pdf'):
            name=f'{panel}_terminal_loss_perplexity.{suffix}';fig.savefig(stage/name,dpi=180,bbox_inches='tight');files.append(name)
        plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
        for ax,width in zip(axes.flat,c.WIDTH_LABELS):
            for series in labels:
                color,marker=c.LINEAR_CALR_STYLES[series]
                if series=='standalone-cosine':
                    endpoint=next(r for r in current if r['series']==series and r['width']==width)
                    ax.scatter([87132],[endpoint['loss']],color=color,marker=marker,s=65,label=names[series],zorder=4)
                else:
                    lr=peak if series=='S1-cosine' else .008
                    curve=sorted((r for r in progress if r['series']==series and r['peak_lr']==lr and r['width']==width),key=lambda r:r['step'])
                    ax.plot([r['step'] for r in curve],[r['loss'] for r in curve],color=color,linewidth=1,label=names[series])
            ax.set(title=width,xlabel='Global optimizer updates',ylabel='Recorded ordinary-validation loss');ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8);fig.suptitle('S1 measured validation progress; standalone terminal point only')
        fig.text(.5,.015,caption(peak),ha='center',fontsize=8);fig.tight_layout(rect=(0,.09,1,.94))
        for suffix in ('png','pdf'):
            name=f'{panel}_validation_progress.{suffix}';fig.savefig(stage/name,dpi=180,bbox_inches='tight');files.append(name)
        plt.close(fig)
    c._check_sources(sources)
    binding=dict(scope='S1-only interim',new_s1_terminal_status='complete',full_campaign_status='incomplete',execution=execution,
        sources=sources,figure_sha256={name:ops.digest(stage/name) for name in files},
        endpoint_sha256=ops.digest(stage/'endpoints.csv'),generator_source=c._source_record(Path(__file__)))
    c.write_json_artifact(stage/'plot_sources.json',binding)
    table=[]
    for width in c.WIDTH_LABELS:
        values={series:next(r['loss'] for r in selected(.008) if r['width']==width and r['series']==series) for series in labels}
        table.append(dict(width=width,**values))
    c.write_json_artifact(stage/'summary.json',dict(output_dir=str(final),endpoint_loss_table=table,figures=files))
    c._check_sources(sources)
    return dict(output_dir=str(final),endpoint_loss_table=table,figures=files)
print(json.dumps(c._publish_directory(output,write),indent=2),flush=True)
