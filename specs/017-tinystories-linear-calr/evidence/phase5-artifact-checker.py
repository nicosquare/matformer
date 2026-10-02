import csv,json,math,hashlib
from pathlib import Path
p=Path('/home/ivo.navarrete/ElasticNN/matformer/specs/017-tinystories-linear-calr')
out=p/'evidence/phase5-final-comparison'
def load(name):return list(csv.DictReader((out/(name+'.csv')).open()))
report=json.loads((out/'comparison_report.json').read_text());manifest=json.loads((out/'plot_sources.json').read_text())
assert report['status']=='complete',report
assert all(report[k]=='complete' for k in ('new_terminals','references','trajectories','comparison'))
for name,digest in manifest['output_sha256'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest,name
rows=load('endpoints');pairs=load('paired_differences');inter=load('interactions');supp=load('supplemental_004_differences')
assert (len(rows),len(pairs),len(inter),len(supp))==(36,48,4,16)
idx={r['endpoint_id']:r for r in rows};assert len(idx)==36
for r in rows:assert math.isclose(float(r['perplexity']),math.exp(float(r['loss'])),rel_tol=1e-10)
for r in pairs+supp:
 a,b=idx[r['left_endpoint_id']],idx[r['right_endpoint_id']]
 for key,value in [('delta_loss',float(a['loss'])-float(b['loss'])),('perplexity_ratio',float(a['perplexity'])/float(b['perplexity'])),('relative_gap',float(a['perplexity'])/float(b['perplexity'])-1)]:assert math.isclose(float(r[key]),value,rel_tol=1e-10,abs_tol=1e-12)
for r in inter:
 values=[float(idx[r[k+'_endpoint_id']]['loss']) for k in ('S2_CaLR','S2_poly','S1_CaLR','S1_poly')]
 assert math.isclose(float(r['interaction_loss']),(values[0]-values[1])-(values[2]-values[3]),abs_tol=1e-12)
figures=[n for n in manifest['output_sha256'] if n.endswith(('.png','.pdf'))];assert len(figures)==16
widths=('g250','g500','g750','g1000');series=('standalone-cosine','S1-cosine','S2-cosine','S1-polynomial','S1-carl','S2-polynomial','S2-carl')
assert manifest['plot_labels']==['standalone-cosine','S1-cosine','S2-cosine','S1-polynomial','S2-polynomial','S1-carl','S2-carl']
def loss(s,w,peak=.008):return float(next(r['loss'] for r in rows if r['series']==s and r['width']==w and float(r['peak_lr'])==peak))
table=['| Width | Standalone | S1 cosine | S2 cosine | S1 poly | S1 CaLR | S2 poly | S2 CaLR |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for w in widths:table.append('| '+w+' | '+' | '.join(f'{loss(s,w):.6f}' for s in series)+' |')
text='''# TinyStories linear S1/S2 polynomial/CaLR experiment

Phase 5 is complete. All four new arms reached 348528 optimizer updates and 2855141376 training tokens each: 1394112 updates and 11420565504 tokens in total. Slurm, worker, CUDA BF16, resource, own-checkpoint, sampler and committed trace checks passed. All eight historical runs were revalidated read-only, contributing 20 reference endpoints.

The real report has **36 unique endpoints, 48 required directed comparisons, four interactions, 16 separate supplemental .004 comparisons and 16 PNG/PDF figures**. New-terminal, reference, trajectory and comparison states are all complete. Every published table/figure hash matches its plot-source manifest; endpoint perplexities, all pair directions/ratios/gaps and all four interactions were independently recomputed from exported CSV files.

[Terminal loss plot](evidence/phase5-final-comparison/primary_loss.png) · [Perplexity plot](evidence/phase5-final-comparison/primary_perplexity.png) · [Measured validation curves](evidence/phase5-final-comparison/primary_validation_progress.png) · [Measured applied LR](evidence/phase5-final-comparison/primary_applied_lr.png) · [Tuned-baseline comparison](evidence/phase5-final-comparison/supplemental_loss.png).

## Terminal ordinary-validation loss

Primary cosine peaks are .008; new arms and standalones also use .008. Lower loss is better.

'''+ '\n'.join(table)+'''\n\n## All-width findings

'''+(out/'findings.md').read_text()+'''
## Evidence and practical limits

The canonical report is `/nfs-stor/ivo.navarrete/results/elasticnn/tinystories-linear-s1-s2-calr-v1/reports/comparison`; a reviewable copy is [saved here](evidence/phase5-final-comparison/comparison_report.json). [Plot sources](evidence/phase5-final-comparison/plot_sources.json) bind actual configs, checkpoints, ordinary evaluations, recorded metrics and reporter code; [reporting snapshot manifest](evidence/phase5-final-comparison/reporting-source-manifest.json) binds the separate corrected reporting snapshot. [Production terminal proof](evidence/phase5-production-terminals.json) binds scheduler/worker/resource evidence and the unchanged production source/config/CPU/GPU gates.

All curves are measured; no observed LR was reconstructed analytically. LR exports retain the first65, every128th global commit and last per width after validating the entire trace. Standalone progress plots contain only the terminal point at87132. Primary .008 and supplemental .004 baselines occupy separate panels. Historical files were not modified or retrained.

Each production arm has one successful attempt and zero failed production-process seconds. Ordinary-validation and update costs overlap process time, which overlaps Slurm allocation; they must not be summed. The earlier failed GPU diagnostic's18 allocation seconds remain separately documented in verification.md. Reporting-only source fixes do not relabel or replace production readiness. Continuous monitoring remains stopped.

T031 and T042 are complete. Phase6 tasks T043–T045 remain pending and are outside this Phase5 finalization.
'''
(p/'experiment-report.md').write_text(text)
tasks=(p/'tasks.md').read_text().replace('- [ ] T042','- [X] T042');(p/'tasks.md').write_text(tasks)
with (p/'verification.md').open('a') as f:f.write('\n### T042 complete: real publication\n\nThe launcher report returned0 and atomically published `ROOT/reports/comparison`; all four independent states are complete. Eight historical references and20 endpoints were revalidated read-only. The review copy in `evidence/phase5-final-comparison` has36 endpoints,48 required pairs,4 interactions,16 supplemental pairs and16 PNG/PDF artifacts. Independent CSV checks recomputed exp(loss), every pair difference/ratio/gap and every interaction; all output SHA-256 values match `plot_sources.json`. The final reporting source manifest and code hash are retained alongside the artifacts. See experiment-report.md for measured all-width findings. T042 is checked complete; Phase6 remains pending.\n')
summary=dict(status='complete',counts=dict(endpoints=36,pairs=48,interactions=4,supplemental_pairs=16,figures=16),output_hashes_verified=True,arithmetic_verified=True,reporting_snapshot=json.loads((out/'reporting-source-manifest.json').read_text())['sha256'])
(p/'evidence/phase5-final-artifact-checks.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2));print('\n'.join(table))
