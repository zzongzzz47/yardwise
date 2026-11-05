"""Restricted container relocation: isolated training, validation and holdout tests.

Only the currently requested ID is revealed to policies. Future benchmark requests
are held by the environment. Bottom-to-top stack convention; one homogeneous bay.
"""
from __future__ import annotations
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)

def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def load_case(path):
    data = json.loads(path.read_text())
    p = np.asarray(data['retrieval_probability'], dtype=float)
    assert p.shape[1] == 12 and np.isfinite(p).all() and (p >= 0).all()
    assert np.max(np.abs(p.sum(axis=1) - 1)) < .031
    p /= p.sum(axis=1, keepdims=True)
    state = [list(s) for s in data['initial_layout']]
    order = data['retrieval_sequence']
    assert sorted(sum(state, [])) == sorted(order) == list(range(len(p)))
    assert all(len(s) <= data['max_tier'] for s in state)
    return state, int(data['max_tier']), order, p

def paths(split):
    indices = {'train': range(6), 'validation': range(6, 8), 'test': range(8, 12)}[split]
    return sorted(p for p in (ROOT / 'data/raw/instances').glob('*.json')
                  if int(p.stem.split('-')[-1]) in indices)

def stats(p):
    mu = p @ np.arange(p.shape[1])
    # pair[i,j] = P(i retrieved before j), with equal probability for within-bin ties.
    future = 1 - np.cumsum(p, axis=1) + .5 * p
    pair = p @ future.T
    np.fill_diagonal(pair, .5)
    return mu, pair

def legal(state, src, h):
    return [j for j, s in enumerate(state) if j != src and len(s) < h]

def expected_rule(state, src, h, mu):
    b = state[src][-1]
    def score(j):
        minimum = min((mu[c] for c in state[j]), default=99.)
        if minimum > mu[b]:
            return (0, minimum - mu[b], len(state[j]), j)
        return (1, -minimum, len(state[j]), j)
    return min(legal(state, src, h), key=score)

def probability_rule(state, src, h, pair):
    b = state[src][-1]
    # A container b above a lower one c blocks c when c is requested first.
    def score(j):
        risks = [float(pair[c, b]) for c in state[j]]
        risk = 1 - math.prod(1 - v for v in risks)
        return risk, sum(risks), len(state[j]), j
    return min(legal(state, src, h), key=score)

def fixed_rule(state, src, h, rank):
    b = state[src][-1]
    safe, unsafe = [], []
    for j in legal(state, src, h):
        m = min((rank[c] for c in state[j]), default=999)
        if m > rank[b]:
            safe.append((m, len(state[j]), j))
        else:
            unsafe.append((-m, len(state[j]), j))
    return min(safe or unsafe)[-1]

def rollout(state, h, order, rank):
    state = [s.copy() for s in state]
    cost = 0
    for target in order:
        src = next(j for j, s in enumerate(state) if target in s)
        while state[src][-1] != target:
            j = fixed_rule(state, src, h, rank)
            state[j].append(state[src].pop())
            cost += 1
        state[src].pop()
    return cost

def sampled_futures(state, target, p, rng, samples):
    # No actual future sequence argument is accepted here.
    others = sorted(c for s in state for c in s if c != target)
    result = []
    cdf = np.cumsum(p[others], axis=1)
    for _ in range(samples):
        u = rng.random(len(others))
        bins = (u[:, None] > cdf).sum(axis=1)
        tie = rng.random(len(others))
        seq = [target] + [others[k] for k in np.lexsort((tie, bins))]
        rank = np.empty(len(p), dtype=int)
        rank.fill(999)
        rank[seq] = np.arange(len(seq))
        result.append((seq, rank))
    return result

def action_costs(state, src, target, h, p, rng, samples):
    futures = sampled_futures(state, target, p, rng, samples)
    candidates = legal(state, src, h)
    costs = []
    for dest in candidates:
        after = [s.copy() for s in state]
        after[dest].append(after[src].pop())
        costs.append(1 + np.mean([rollout(after, h, seq, rank) for seq, rank in futures]))
    return candidates, np.asarray(costs)

FEATURES = ['stacks','height','remaining','source_height','destination_height',
            'current_target_mean','blocker_mean','blocker_std','blocker_entropy',
            'destination_min_mean','destination_max_mean','destination_mean_mean',
            'destination_min_gap','destination_max_gap','risk_sum','risk_max','risk_any',
            'empty_stacks','free_slots','full_stacks','mean_stack_height','blocker_mean_rank',
            'destination_mean_rank','destination_risk_mean','best_available_risk',
            'state_inversion_sum','destination_spread','source_remaining_depth',
            'blocker_p0','blocker_p1','blocker_p2','blocker_p3','blocker_p4','blocker_p5',
            'blocker_p6','blocker_p7','blocker_p8','blocker_p9','blocker_p10','blocker_p11']

def features(state, src, target, h, p, mu, pair):
    b = state[src][-1]
    candidates = legal(state, src, h)
    remaining = sum(state, [])
    bm = float(mu[b])
    std = float(np.sqrt(p[b] @ (np.arange(12) - bm) ** 2))
    entropy = float(-(p[b] * np.log(np.maximum(p[b], 1e-12))).sum())
    inv = sum(float(pair[c, d]) for s in state for k, c in enumerate(s) for d in s[k+1:])
    risks_all = [1 - math.prod(1 - float(pair[c, b]) for c in state[j]) for j in candidates]
    common = [len(state), h, len(remaining), len(state[src])]
    rows = []
    for j in candidates:
        lower = state[j]
        ms = [float(mu[c]) for c in lower]
        risks = [float(pair[c, b]) for c in lower]
        mn, mx = min(ms, default=12.), max(ms, default=12.)
        row = common + [len(lower),float(mu[target]),bm,std,entropy,mn,mx,
                float(np.mean(ms)) if ms else 12.,mn-bm,mx-bm,
                sum(risks),max(risks,default=0.),1-math.prod(1-v for v in risks),
                sum(not s for s in state),len(state)*h-len(remaining),sum(len(s)==h for s in state),
                len(remaining)/len(state),sum(mu[c]<bm for c in remaining)/len(remaining),
                sum(mu[c]<mn for c in remaining)/len(remaining),
                float(np.mean(risks)) if risks else 0.,min(risks_all),inv,
                float(np.std(ms)) if ms else 0.,len(state[src])-1-state[src].index(target)] + p[b].tolist()
        assert len(row) == len(FEATURES)
        rows.append(row)
    return candidates, np.asarray(rows)

def run_policy(initial, h, order, p, policy, seed=0, model=None, samples=16, trace=False):
    state = [s.copy() for s in initial]
    mu, pair = stats(p)
    rng = np.random.default_rng(seed)
    moves, trail, elapsed = 0, [], 0.
    for target in order:
        src = next(j for j,s in enumerate(state) if target in s)
        while state[src][-1] != target:
            started = time.perf_counter()
            if policy == 'height':
                dest = min(legal(state,src,h),key=lambda j:(len(state[j]),abs(src-j),j))
            elif policy == 'mean':
                dest = expected_rule(state,src,h,mu)
            elif policy == 'risk':
                dest = probability_rule(state,src,h,pair)
            elif policy == 'lookahead':
                cand,cost = action_costs(state,src,target,h,p,rng,samples)
                dest = cand[min(range(len(cand)),key=lambda k:(cost[k],len(state[cand[k]]),cand[k]))]
            elif policy == 'learned':
                cand,x = features(state,src,target,h,p,mu,pair)
                scores = model.predict(x)
                dest = cand[min(range(len(cand)),key=lambda k:(scores[k],len(state[cand[k]]),cand[k]))]
            else:
                raise ValueError(policy)
            elapsed += time.perf_counter()-started
            c=state[src].pop();state[dest].append(c);moves+=1
            if trace: trail.append({'type':'move','container':c,'from':src,'to':dest,'target':target,'moves':moves})
        state[src].pop()
        if trace: trail.append({'type':'retrieve','container':target,'from':src,'to':-1,'target':target,'moves':moves})
    assert not any(state)
    return {'relocations':moves,'retrievals':len(order),'decisions_seconds':elapsed,'trace':trail}

def make_training():
    X,Y,G=[],[],[]
    count=0
    for idx,path in enumerate(paths('train')):
        state,h,order,p=load_case(path); mu,pair=stats(p)
        rng=np.random.default_rng(7000+idx)
        for target in order:
            src=next(j for j,s in enumerate(state) if target in s)
            while state[src][-1] != target:
                candidates,x=features(state,src,target,h,p,mu,pair)
                # Observe a mixed behavior policy. Training labels use simulated futures,
                # never the benchmark's unobserved actual tail.
                if count%2==0:
                    _,cost=action_costs(state,src,target,h,p,rng,12)
                    y=cost-cost.mean()
                    X.extend(x);Y.extend(y);G.extend([path.name]*len(y))
                count+=1
                if rng.random()<.25:
                    dest=int(rng.choice(candidates))
                elif rng.random()<.5:
                    dest=expected_rule(state,src,h,mu)
                else:
                    dest=probability_rule(state,src,h,pair)
                state[dest].append(state[src].pop())
            state[src].pop()
        if idx%40==0: print('training cases',idx,'action labels',len(Y),flush=True)
    np.savez_compressed(OUT/'training.npz',X=np.asarray(X),Y=np.asarray(Y),G=np.asarray(G))
    save('training_audit.json',{'cases':len(paths('train')),'labeled_actions':len(Y),
         'behavior_decisions':count,'label':'Centered mean cost of 12 probability-sampled greedy rollouts per legal action.',
         'future_benchmark_tail_used_as_label':False,'features':FEATURES})
    print('training data saved',len(Y),flush=True)

def fit():
    d=np.load(OUT/'training.npz')
    model=ExtraTreesRegressor(n_estimators=100,max_depth=16,min_samples_leaf=8,
                             max_features=.9,random_state=20261028,n_jobs=1)
    model.fit(d['X'],d['Y'])
    joblib.dump(model,OUT/'move_value_model.joblib')
    save('model.json',{'model':'ExtraTreesRegressor','parameters':model.get_params(),
         'features':FEATURES,'target':'simulated future relocation cost, centered within current decision',
         'source':'Own fitted model. Upstream retrieval probabilities supplied by the public research dataset.',
         'feature_importances':dict(zip(FEATURES,model.feature_importances_.tolist()))})

def aggregate(records):
    policies=list(records[0]['policies'])
    summary=[]
    for method in policies:
        counts=np.array([r['policies'][method]['relocations'] for r in records])
        baseline=np.array([r['policies']['mean']['relocations'] for r in records])
        diff=baseline-counts
        rng=np.random.default_rng(1985)
        boot=np.mean(rng.choice(diff,(4000,len(diff)),replace=True),axis=1)
        summary.append({'policy':method,'cases':len(records),'mean':float(counts.mean()),
            'total':int(counts.sum()),'mean_saving_vs_mean':float(diff.mean()),
            'saving_percent_vs_mean':float(100*diff.sum()/max(1,baseline.sum())),
            'better':int((diff>0).sum()),'same':int((diff==0).sum()),'worse':int((diff<0).sum()),
            'paired_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),
            'mean_decision_ms':1000*sum(r['policies'][method]['decisions_seconds'] for r in records)/max(1,counts.sum())})
    return summary

def evaluate(split):
    model=joblib.load(OUT/'move_value_model.joblib')
    policies=['height','mean','risk','learned','lookahead']
    rows=[]
    for idx,path in enumerate(paths(split)):
        state,h,order,p=load_case(path)
        row={'id':path.stem,'stacks':len(state),'height':h,'containers':len(order),
             'occupancy':len(order)/(len(state)*h),'initial':state,'order':order,'probability':p.tolist(),'policies':{}}
        for policy in policies:
            row['policies'][policy]=run_policy(state,h,order,p,policy,20000+idx,model,24,trace=True)
        rows.append(row)
        if idx%30==0: print(split,idx,'/',len(paths(split)),flush=True)
    save(split+'_records.json',rows)
    save(split+'_summary.json',aggregate(rows))
    print(json.dumps(aggregate(rows),ensure_ascii=False,indent=2),flush=True)

def exact_minimum(initial,h,order):
    rank={c:k for k,c in enumerate(order)}
    @lru_cache(maxsize=None)
    def solve(state,k):
        if k==len(order):return 0
        target=order[k]
        src=next(j for j,s in enumerate(state) if target in s)
        if state[src][-1]==target:
            nexts=list(state);nexts[src]=state[src][:-1]
            return solve(tuple(nexts),k+1)
        best=999
        for j in legal(state,src,h):
            nxt=list(state);nxt[j]=state[j]+(state[src][-1],);nxt[src]=state[src][:-1]
            best=min(best,1+solve(tuple(nxt),k))
        return best
    return solve(tuple(tuple(s) for s in initial),0)

def audit():
    splitsets={}
    for sp in ['train','validation','test']:
        splitsets[sp]=set()
        for path in paths(sp):
            s,h,o,p=load_case(path)
            splitsets[sp].add(hashlib.sha256(json.dumps([s,h]).encode()).hexdigest())
    intersections={f'{a}/{b}':len(splitsets[a]&splitsets[b]) for a,b in [('train','test'),('train','validation'),('validation','test')]}
    assert not any(intersections.values()),intersections
    records=json.loads((OUT/'test_records.json').read_text())
    checks=0;optimal=[]
    for row in records:
        for method,result in row['policies'].items():
            state=[s.copy() for s in row['initial']];k=0;m=0
            for e in result['trace']:
                assert e['target']==row['order'][k]
                assert state[e['from']][-1]==e['container']
                assert e['target'] in state[e['from']]
                c=state[e['from']].pop()
                if e['type']=='move':
                    assert c!=e['target'] and e['from']!=e['to']
                    assert len(state[e['to']])<row['height']
                    state[e['to']].append(c);m+=1
                else:
                    assert c==row['order'][k];k+=1
                assert e['moves']==m
            assert k==len(row['order']) and m==result['relocations'] and not any(state)
            checks+=1
        if row['stacks']==4 and row['height']==3:
            optimum=exact_minimum(row['initial'],row['height'],row['order'])
            assert all(r['relocations']>=optimum for r in row['policies'].values())
            optimal.append({'id':row['id'],'perfect_information_optimum':optimum})
    # First action must be invariant to replacing future requests, fixed seed.
    model=joblib.load(OUT/'move_value_model.joblib');invariance=0
    blocked_initial_requests=[r for r in records
        if any(r['order'][0] in s and s[-1]!=r['order'][0] for s in r['initial'])]
    for row in blocked_initial_requests[:12]:
        seq=row['order']; changed=[seq[0]]+list(reversed(seq[1:]))
        for method in row['policies']:
            a=run_policy(row['initial'],row['height'],seq,np.asarray(row['probability']),method,99,model,24,True)['trace'][0]
            b=run_policy(row['initial'],row['height'],changed,np.asarray(row['probability']),method,99,model,24,True)['trace'][0]
            assert a['type']=='move' and a==b;invariance+=1
    save('validation.json',{'all_passed':True,'trace_replays':checks,'disjoint_initial_layouts':intersections,
         'exact_small_cases':optimal,'future_tail_invariance_checks':invariance,
         'layout_convention':'lists treated bottom to top, last is crane-accessible',
         'limits':['Single bay, homogeneous boxes, one relocation at a time; no new arrivals.',
                   'Frozen published probabilities. No local terminal prediction model was trained.',
                   'No measured time, energy, emissions or field savings are claimed.']})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','fit','validation','test','audit'])
    arg=parser.parse_args().stage
    {'train':make_training,'fit':fit,'validation':lambda:evaluate('validation'),
     'test':lambda:evaluate('test'),'audit':audit}[arg]()
