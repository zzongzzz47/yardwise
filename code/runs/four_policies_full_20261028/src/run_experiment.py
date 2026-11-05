"""Four independent collection policies, all decisions and legal destinations.

Original files and the prior experiment remain unchanged. Probability sampling
and behavior RNGs are independent. All repeated states remain in the dataset.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import sklearn, scipy
from sklearn.ensemble import ExtraTreesRegressor
import baseline_experiment as b

RUN = Path(__file__).resolve().parents[1]
PROJECT = RUN.parents[1]
OUT = RUN / 'results'
CHECK = RUN / 'checkpoints'
b.ROOT = PROJECT
b.OUT = OUT
ROUTES = ('height', 'mean', 'risk', 'random')
SAMPLES = 12
BASE_SEED = 20261028

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))

def seed_for(*parts):
    payload=json.dumps([BASE_SEED,*parts],separators=(',',':'),sort_keys=True)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'little')

def state_key(case, state, target):
    payload=json.dumps([case,state,target],separators=(',',':'))
    return hashlib.sha256(payload.encode()).hexdigest()

def choose(route, state, src, height, mu, pair, rng):
    candidates=b.legal(state,src,height)
    if route=='height':
        return min(candidates,key=lambda j:(len(state[j]),abs(src-j),j))
    if route=='mean':return b.expected_rule(state,src,height,mu)
    if route=='risk':return b.probability_rule(state,src,height,pair)
    if route=='random':return int(rng.choice(candidates))
    raise ValueError(route)

def initialize():
    for p in [OUT,CHECK,RUN/'deliverables',RUN/'logs']:p.mkdir(exist_ok=True)
    # Declare the design before generating labels or evaluating the new model.
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),
          'revision':'four_policies_full_20261028',
          'collection_routes':list(ROUTES),'routes_per_training_case':4,
          'training_cases':336,'validation_cases':112,'evaluation_cases':224,
          'split':'Original suffix: train 0..5; validation 6..7; evaluation 8..11.',
          'decision_stride':1,'all_legal_destinations_labeled':True,
          'sampling':{'training_futures':SAMPLES,'comparison_lookahead_futures':24,
                      'common_futures_across_destinations':True,
                      'independent_container_draws':True,'within_period_ties':'uniform random',
                      'behavior_rng':'separate fixed stream per original case and route',
                      'label_rng':'stable hash of case, layout, current target; equal states receive equal futures'},
          'duplicate_states':'Retain every occurrence and every candidate row; no deduplication or reweighting.',
          'random_route':'One reproducible uniformly random legal-destination trajectory per training case.',
          'teacher':'One chosen relocation plus greedy rollout under each sampled complete future; mean cost centered among legal destinations.',
          'known_information':'Layout, capacity, current request and supplied fixed probabilities; no actual future tail is passed to features, route selectors or label generation.',
          'model':'ExtraTrees: 100 trees, depth 16, min leaf 8, max_features .9, seed 20261028, n_jobs 1; unchanged.',
          'validation_role':'Check the fixed model; no tuning or favorable-result selection.',
          'test_role':'Reanalysis of the previously examined 224-case benchmark, not a fresh unseen external holdout.',
          'old_data_collection':'One mixed route per case, 25% random / 37.5% mean / 37.5% risk; every other decision labeled.',
          'collection_counts_are_observed_not_targets':True,
          'baseline_snapshot_sha256':sha(RUN/'src/baseline_experiment.py'),
          'legacy_model_sha256':sha(PROJECT/'results/move_value_model.joblib'),
          'legacy_test_sha256':sha(PROJECT/'results/test_records.json'),
          'new_runner_sha256':sha(__file__)}
    path=OUT/'study_plan.json'
    if path.exists():
        prior=json.loads(path.read_text())
        assert prior['new_runner_sha256']==plan['new_runner_sha256'],'Runner changed: use a new run directory.'
    else:dump(path,plan)
    dump(OUT/'environment.json',{'python':sys.version,'executable':sys.executable,
         'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,
         'joblib':joblib.__version__,'seed':BASE_SEED})

def process_case(path_string):
    path=Path(path_string);cid=path.stem
    cache=CHECK/f'{cid}.npz';detail_path=CHECK/f'{cid}.json'
    if cache.exists() and detail_path.exists():
        d=json.loads(detail_path.read_text())
        assert d['input_sha256']==sha(path)
        return {'case':cid,'rows':d['rows'],'decisions':len(d['decisions']),'cached':True}
    initial,h,order,p=b.load_case(path);mu,pair=b.stats(p)
    X=[];Y=[];C=[];R=[];K=[];DEST=[];decisions=[];trajectories=[]
    for route in ROUTES:
        state=[s.copy() for s in initial]
        behavior_seed=seed_for('behavior',cid,route)
        rng=np.random.default_rng(behavior_seed)
        trail=[];number=0;row_start=len(Y)
        for request_index,target in enumerate(order):
            src=next(j for j,s in enumerate(state) if target in s)
            while state[src][-1]!=target:
                number+=1
                candidates,x=b.features(state,src,target,h,p,mu,pair)
                assert candidates==b.legal(state,src,h) and candidates
                key=state_key(cid,state,target)
                label_seed=seed_for('labels',key)
                cc,cost=b.action_costs(state,src,target,h,p,np.random.default_rng(label_seed),SAMPLES)
                assert cc==candidates and len(cost)==len(x)
                y=cost-cost.mean();start=len(Y)
                dest=choose(route,state,src,h,mu,pair,rng)
                assert dest in candidates
                record={'case':cid,'route':route,'number':number,'request_index':request_index,
                        'target':target,'blocker':state[src][-1],'source_column':src+1,
                        'layout':[s.copy() for s in state],'state_key':key,
                        'candidate_columns':[j+1 for j in candidates],
                        'sample_mean_cost':cost.tolist(),'relative_cost':y.tolist(),
                        'label_seed':label_seed,'futures':SAMPLES,
                        'selected_column':dest+1,'case_row_start':start,'case_row_stop':start+len(y)}
                decisions.append(record)
                X.extend(x);Y.extend(y);C.extend([cid]*len(y));R.extend([route]*len(y))
                K.extend([key]*len(y));DEST.extend(candidates)
                c=state[src].pop();state[dest].append(c)
                assert len(state[dest])<=h
                trail.append({'type':'move','container':c,'from':src,'to':dest,'target':target,'moves':number})
            state[src].pop()
            trail.append({'type':'retrieve','container':target,'from':src,'to':-1,'target':target,'moves':number})
        assert not any(state)
        trajectories.append({'case':cid,'route':route,'behavior_seed':behavior_seed,
             'decisions':number,'labeled_decisions':number,'candidate_rows':len(Y)-row_start,
             'retrievals':len(order),'trace':trail})
    np.savez_compressed(cache,X=np.asarray(X).reshape(-1,len(b.FEATURES)),Y=np.asarray(Y),
                        case=np.asarray(C,dtype=str),route=np.asarray(R,dtype=str),
                        state=np.asarray(K,dtype=str),destination=np.asarray(DEST,dtype=int))
    dump(detail_path,{'case':cid,'input_sha256':sha(path),'rows':len(Y),
                     'decisions':decisions,'trajectories':trajectories})
    return {'case':cid,'rows':len(Y),'decisions':len(decisions),'cached':False}

def collect(workers):
    started=time.monotonic();files=b.paths('train')
    counts={'cases':0,'decisions':0,'rows':0}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(process_case,map(str,files),chunksize=1):
            counts['cases']+=1;counts['decisions']+=result['decisions'];counts['rows']+=result['rows']
            if counts['cases']%12==0 or counts['cases']==len(files):
                print('COLLECT',json.dumps({**counts,'seconds':round(time.monotonic()-started,1)}),flush=True)
    arrays={name:[] for name in ['X','Y','case','route','state','destination']}
    trajectories=[];hist=Counter();unique=set();label_fingerprints={};offset=0
    with (OUT/'training_decisions.jsonl').open('w') as log:
        for path in files:
            with np.load(CHECK/f'{path.stem}.npz') as arr:
                for name in arrays:
                    value=arr[name]
                    if name=='X':value=value.reshape(-1,len(b.FEATURES))
                    if name in ['case','route','state']:value=value.astype(str)
                    if name=='destination':value=value.astype(int)
                    arrays[name].append(value)
            detail=json.loads((CHECK/f'{path.stem}.json').read_text())
            trajectories.extend(detail['trajectories'])
            for d in detail['decisions']:
                count=len(d['candidate_columns']);hist[count]+=1;unique.add(d['state_key'])
                fingerprint=json.dumps([d['candidate_columns'],d['sample_mean_cost']])
                assert label_fingerprints.setdefault(d['state_key'],fingerprint)==fingerprint
                d['global_row_start']=offset+d['case_row_start'];d['global_row_stop']=offset+d['case_row_stop']
                log.write(json.dumps(d,separators=(',',':'))+'\n')
            offset+=detail['rows']
    joined={name:np.concatenate(parts) for name,parts in arrays.items()}
    assert joined['X'].shape==(counts['rows'],40) and len(joined['Y'])==counts['rows']
    np.savez_compressed(OUT/'training.npz',**joined)
    dump(OUT/'training_trajectories.json',trajectories)
    route_summary={r:{'trajectories':sum(t['route']==r for t in trajectories),
           'decisions':sum(t['decisions'] for t in trajectories if t['route']==r),
           'rows':sum(t['candidate_rows'] for t in trajectories if t['route']==r)} for r in ROUTES}
    audit={**counts,'original_cases':len(files),'trajectories':len(trajectories),
           'labeled_decisions':counts['decisions'],'skipped_decisions':0,
           'unique_labeled_states':len(unique),'duplicate_state_occurrences':counts['decisions']-len(unique),
           'candidate_count_histogram':dict(sorted(hist.items())),'by_route':route_summary,
           'features':b.FEATURES,'training_future_samples':SAMPLES,
           'candidate_rollouts':counts['rows']*SAMPLES,
           'seconds':time.monotonic()-started,'all_occurrences_retained':True,
           'same_state_label_consistency':True,'future_benchmark_tail_used_as_label':False}
    assert sum(k*v for k,v in hist.items())==counts['rows']
    assert len(trajectories)==4*336 and sum(t['decisions'] for t in trajectories)==counts['decisions']
    dump(OUT/'training_audit.json',audit)
    with (OUT/'경로별_수집량.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['원본사례','진행방식','이동판단','평가한판단','후보별학습자료'])
        for t in trajectories:w.writerow([t['case'],t['route'],t['decisions'],t['labeled_decisions'],t['candidate_rows']])
    print('TRAINING_AUDIT',json.dumps(audit,ensure_ascii=False),flush=True)

def fit():
    d=np.load(OUT/'training.npz')
    params=json.loads((PROJECT/'results/model.json').read_text())['parameters']
    model=ExtraTreesRegressor(**params)
    model.fit(d['X'],d['Y'])
    joblib.dump(model,OUT/'move_value_model.joblib')
    dump(OUT/'model.json',{'model':'ExtraTreesRegressor','parameters':model.get_params(),
         'features':b.FEATURES,'training_rows':len(d['Y']),
         'target':'Mean cost under 12 probability-sampled greedy futures, centered within current decision',
         'feature_importances':dict(zip(b.FEATURES,model.feature_importances_.tolist()))})
    dump(OUT/'frozen_selection.json',{'primary':'learned','model_sha256':sha(OUT/'move_value_model.joblib'),
         'frozen_utc':datetime.now(timezone.utc).isoformat(),
         'extra_test_training':False,'reason':'Fixed user-requested collection revision; same model parameters; no tuning.',
         'evaluation_is_reused_benchmark':True})
    print('MODEL_FROZEN',sha(OUT/'move_value_model.joblib'),flush=True)

def audit_training():
    trajectories=json.loads((OUT/'training_trajectories.json').read_text())
    audit=json.loads((OUT/'training_audit.json').read_text())
    decisions=[json.loads(line) for line in (OUT/'training_decisions.jsonl').read_text().splitlines()]
    lookup={(d['case'],d['route'],d['number']):d for d in decisions}
    assert len(lookup)==len(decisions)
    data=np.load(OUT/'training.npz');covered=np.zeros(len(data['Y']),dtype=np.int8)
    for t in trajectories:
        state,h,order,p=b.load_case(PROJECT/'data/raw/instances'/f"{t['case']}.json")
        mu,pair=b.stats(p);k=m=0;rng=np.random.default_rng(t['behavior_seed'])
        for event in t['trace']:
            src=event['from'];assert event['target']==order[k] and state[src][-1]==event['container']
            if event['type']=='move':
                m+=1;d=lookup[(t['case'],t['route'],m)]
                cand,x=b.features(state,src,order[k],h,p,mu,pair)
                assert d['layout']==state and d['candidate_columns']==[j+1 for j in cand]
                assert event['to']==choose(t['route'],state,src,h,mu,pair,rng)
                start,stop=d['global_row_start'],d['global_row_stop'];covered[start:stop]+=1
                assert np.allclose(data['X'][start:stop],x,rtol=0,atol=1e-12)
                assert np.allclose(data['Y'][start:stop],d['relative_cost'],rtol=0,atol=1e-12)
                assert stop-start==len(cand) and event['to'] in cand and event['container']!=order[k]
                state[event['to']].append(state[src].pop());assert len(state[event['to']])<=h
            else:
                assert event['container']==order[k];state[src].pop();k+=1
            assert event['moves']==m
        assert k==len(order) and not any(state) and m==t['decisions']==t['labeled_decisions']
    assert np.all(covered==1) and len(decisions)==audit['decisions']
    # Recalculate deterministic labels on 16 scattered states, including both ends.
    for index in np.linspace(0,len(decisions)-1,16,dtype=int):
        d=decisions[index];_,h,_,p=b.load_case(PROJECT/'data/raw/instances'/f"{d['case']}.json")
        c,cost=b.action_costs(d['layout'],d['source_column']-1,d['target'],h,p,
                            np.random.default_rng(d['label_seed']),SAMPLES)
        assert c==[v-1 for v in d['candidate_columns']] and np.array_equal(cost,d['sample_mean_cost'])
    dump(OUT/'training_validation.json',{'all_passed':True,'trajectories_replayed':len(trajectories),
         'decisions_checked':len(decisions),'candidate_rows_checked':len(data['Y']),
         'every_occurrence_and_legal_destination_recorded':True,'recomputed_labels':16})
    print('TRAINING_REPLAY_PASSED',len(trajectories),len(decisions),len(data['Y']),flush=True)

def compare():
    summary=json.loads((OUT/'test_summary.json').read_text());by={v['policy']:v for v in summary}
    records=json.loads((OUT/'test_records.json').read_text())
    old=json.loads((PROJECT/'results/test_records.json').read_text());old_by={r['id']:r for r in old}
    rows=[];new_vs_old=Counter();mandatory=sum(r['containers'] for r in records)
    for r in records:
        prev=old_by[r['id']]
        assert r['initial']==prev['initial'] and r['order']==prev['order']
        for method in ['height','mean','risk','lookahead']:
            assert r['policies'][method]['relocations']==prev['policies'][method]['relocations']
        a=prev['policies']['learned']['relocations'];v=r['policies']['learned']['relocations']
        new_vs_old['better' if v<a else 'worse' if v>a else 'same']+=1
        rows.append({'case':r['id'],'old_ai':a,'new_ai':v,'difference_new_minus_old':v-a,
                     **{m:r['policies'][m]['relocations'] for m in ['height','mean','risk','lookahead']}})
    old_total=sum(r['old_ai'] for r in rows);new_total=by['learned']['total'];baseline=by['mean']['total']
    metrics={'cases':len(records),'containers':mandatory,'baseline':baseline,'ai':new_total,
       'relocation_saving_percent':100*(baseline-new_total)/baseline,
       'all_moves_saving_percent':100*(baseline-new_total)/(baseline+mandatory),
       'informed_risk_saving_percent':100*(by['risk']['total']-new_total)/by['risk']['total'],
       'naive_saving_percent':100*(by['height']['total']-new_total)/by['height']['total'],
       'lookahead_saved':baseline-by['lookahead']['total'],
       'ai_vs_search_relocation_difference':new_total-by['lookahead']['total'],
       **{k:by['learned'][k] for k in ['better','same','worse']},
       'old_ai':old_total,'new_vs_old_saving':old_total-new_total,'new_vs_old_cases':dict(new_vs_old),
       'old_collection_rows':10462,'new_collection_rows':json.loads((OUT/'training_audit.json').read_text())['rows'],
       'evaluation_is_previously_used_benchmark':True}
    dump(OUT/'key_metrics.json',metrics);dump(OUT/'old_new_comparison.json',rows)
    with (OUT/'수집방식변경_평가비교.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plan=json.loads((OUT/'study_plan.json').read_text())
    assert sha(PROJECT/'results/test_records.json')==plan['legacy_test_sha256']
    assert sha(PROJECT/'results/move_value_model.joblib')==plan['legacy_model_sha256']
    print('COMPARISON',json.dumps(metrics,ensure_ascii=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['all','train','fit','validation','test','audit','compare'])
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    initialize()
    if args.stage in ['all','train']:collect(args.workers)
    if args.stage in ['all','fit']:fit()
    if args.stage in ['all','validation']:b.evaluate('validation')
    if args.stage in ['all','test']:b.evaluate('test')
    if args.stage in ['all','audit']:audit_training();b.audit()
    if args.stage in ['all','compare']:compare()

if __name__=='__main__':main()
