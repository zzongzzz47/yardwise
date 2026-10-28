// Browser implementation of the fitted ExtraTrees policy. Future request order
// is held by the simulator; recommend() accepts only the current request.
export const clone=x=>JSON.parse(JSON.stringify(x));
const sum=a=>a.reduce((x,y)=>x+y,0);
const mean=a=>sum(a)/a.length;
const std=a=>{const m=mean(a);return Math.sqrt(mean(a.map(x=>(x-m)**2)));};
export function decodeModel(buffer){
  const d=new DataView(buffer);if(d.getUint32(0,true)!==0x59415244)throw Error('모형 파일 형식을 확인해 주세요.');
  const trees=d.getUint32(8,true),nodes=d.getUint32(12,true),base=16+trees*4;
  if(buffer.byteLength!==base+nodes*32)throw Error('모형 파일이 완전하지 않습니다.');
  const roots=Array.from({length:trees},(_,i)=>d.getUint32(16+4*i,true));
  return {predict(row){let total=0;for(const root of roots){let n=root;for(let depth=0;depth<100;depth++){
    const at=base+32*n,l=d.getInt32(at,true);if(l<0){total+=d.getFloat64(at+24,true);break;}
    const f=d.getInt32(at+8,true);n=Math.fround(row[f])<=d.getFloat64(at+16,true)?l:d.getInt32(at+4,true);
  }}return total/trees;}};
}
export function stats(p){
  const mu=p.map(row=>sum(row.map((v,i)=>v*i)));
  const future=p.map(row=>{let c=0;return row.map(v=>{c+=v;return 1-c+.5*v;});});
  const pair=p.map((row,i)=>p.map((_,j)=>i===j?.5:sum(row.map((v,k)=>v*future[j][k]))));
  return {mu,pair};
}
export function legal(state,src,h){return state.map((s,j)=>j!==src&&s.length<h?j:-1).filter(j=>j>=0);}
export function features(state,src,target,h,p,context=stats(p)){
  const {mu,pair}=context,b=state[src].at(-1),candidates=legal(state,src,h),remaining=state.flat(),bm=mu[b];
  const deviation=Math.sqrt(sum(p[b].map((v,i)=>v*(i-bm)**2)));
  const entropy=-sum(p[b].map(v=>v*Math.log(Math.max(v,1e-12))));
  let inv=0;for(const s of state)for(let i=0;i<s.length;i++)for(let k=i+1;k<s.length;k++)inv+=pair[s[i]][s[k]];
  const ra=candidates.map(j=>1-state[j].reduce((prod,c)=>prod*(1-pair[c][b]),1));
  const rows=candidates.map(j=>{
    const lower=state[j],ms=lower.map(c=>mu[c]),risks=lower.map(c=>pair[c][b]);
    const mn=ms.length?Math.min(...ms):12,mx=ms.length?Math.max(...ms):12;
    return [state.length,h,remaining.length,state[src].length,lower.length,mu[target],bm,deviation,entropy,mn,mx,
      ms.length?mean(ms):12,mn-bm,mx-bm,sum(risks),risks.length?Math.max(...risks):0,
      1-risks.reduce((prod,v)=>prod*(1-v),1),state.filter(s=>!s.length).length,state.length*h-remaining.length,
      state.filter(s=>s.length===h).length,remaining.length/state.length,
      remaining.filter(c=>mu[c]<bm).length/remaining.length,remaining.filter(c=>mu[c]<mn).length/remaining.length,
      risks.length?mean(risks):0,Math.min(...ra),inv,ms.length?std(ms):0,state[src].length-1-state[src].indexOf(target),...p[b]];
  });return {candidates,rows};
}
function lex(a,b){for(let i=0;i<a.length;i++){if(a[i]!==b[i])return a[i]-b[i];}return 0;}
export function recommend(state,target,h,p,model,policy='learned',context=stats(p)){
  const src=state.findIndex(s=>s.includes(target));
  if(src<0)throw Error('현재 배치에 없는 출고 요청입니다.');
  if(state[src].at(-1)===target)return {type:'retrieve',src,target,container:target,candidates:[],dest:-1};
  const b=state[src].at(-1),cand=legal(state,src,h);if(!cand.length)throw Error('옮길 수 있는 빈 자리가 없습니다.');
  const {mu,pair}=context;let ranks;
  if(policy==='learned'){
    const {rows}=features(state,src,target,h,p,context);
    ranks=cand.map((j,k)=>({j,score:model.predict(rows[k]),tuple:[]}));
    ranks.forEach(r=>r.tuple=[r.score,state[r.j].length,r.j]);
  }else ranks=cand.map(j=>{
    let tuple;if(policy==='height')tuple=[state[j].length,Math.abs(src-j),j];
    else if(policy==='risk'){const r=state[j].map(c=>pair[c][b]);tuple=[1-r.reduce((pr,v)=>pr*(1-v),1),sum(r),state[j].length,j];}
    else {const m=state[j].length?Math.min(...state[j].map(c=>mu[c])):99;tuple=m>mu[b]?[0,m-mu[b],state[j].length,j]:[1,-m,state[j].length,j];}
    return {j,score:tuple[0],tuple};
  });ranks.sort((a,b)=>lex(a.tuple,b.tuple));
  return {type:'move',src,target,container:b,candidates:ranks,dest:ranks[0].j};
}
export function applyAction(state,action,destination=action.dest){
  if(state[action.src].at(-1)!==action.container)throw Error('최상단 컨테이너만 이동할 수 있습니다.');
  if(action.type==='move'&&!action.candidates.some(r=>r.j===destination))throw Error('가능한 후보 위치를 선택해 주세요.');
  const box=state[action.src].pop();if(action.type==='move')state[destination].push(box);
}
export function simulate(c,model,policy='learned'){
  const state=clone(c.stacks),context=stats(c.p);let relocations=0,retrievals=0;const trace=[];
  for(const target of c.order){let guard=0;while(true){
    if(++guard>100)throw Error('작업 횟수가 허용 범위를 벗어났습니다.');
    const a=recommend(state,target,c.height,c.p,model,policy,context);applyAction(state,a);
    if(a.type==='move')relocations++;else retrievals++;
    trace.push({type:a.type,container:a.container,from:a.src,to:a.dest,target,moves:relocations});
    if(a.type==='retrieve')break;
  }}return {relocations,retrievals,trace};
}
export function validateCase(c){
  if(!Array.isArray(c.stacks)||c.stacks.length<4||c.stacks.length>10)throw Error('적재열은 4~10개로 구성해 주세요.');
  if(!Number.isInteger(c.height)||c.height<3||c.height>6)throw Error('최대 높이는 3~6단입니다.');
  const ids=c.stacks.flat(),n=ids.length;
  if(n<6||n>40)throw Error('컨테이너는 6~40개로 구성해 주세요.');
  if(new Set(ids).size!==n||ids.some(v=>!Number.isInteger(v)||v<0||v>=n))throw Error('컨테이너가 중복되거나 누락되었습니다.');
  if(c.stacks.some(s=>s.length>c.height))throw Error('최대 적재 높이를 초과한 열이 있습니다.');
  if(c.stacks.length*c.height-n<c.height-1)throw Error('재취급을 위해 최대 높이보다 1개 적은 수 이상의 빈 자리를 확보해 주세요.');
  if(!Array.isArray(c.order)||c.order.length!==n||new Set(c.order).size!==n||c.order.some(v=>!ids.includes(v)))throw Error('출고 순서에는 모든 컨테이너가 한 번씩 들어가야 합니다.');
  if(!Array.isArray(c.labels)||c.labels.length!==n||new Set(c.labels).size!==n||c.labels.some(s=>!/^[\p{L}\p{N}][\p{L}\p{N}_-]{0,23}$/u.test(s)))throw Error('ID는 문자나 숫자로 시작하는 24자 이내의 고유값이어야 합니다.');
  if(!Array.isArray(c.p)||c.p.length!==n)throw Error('컨테이너별 출고 확률을 입력해 주세요.');
  c.p=c.p.map(row=>{if(!Array.isArray(row)||row.length!==12||row.some(v=>typeof v!=='number'||!Number.isFinite(v)||v<0))throw Error('p0~p11은 0 이상의 유한한 숫자여야 합니다.');const s=sum(row);if(s<=0||Math.abs(s-1)>.05)throw Error('각 컨테이너의 출고 확률 합은 1이어야 합니다 (허용 오차 0.05).');return row.map(v=>v/s);});
  return c;
}
export function parseCSV(text,h){
  const rows=[];let row=[],value='',quote=false;
  text=text.replace(/^\uFEFF/,'');
  for(let i=0;i<text.length;i++){const ch=text[i];if(ch==='"'){if(quote&&text[i+1]==='"'){value+='"';i++;}else quote=!quote;}
    else if(ch===','&&!quote){row.push(value.trim());value='';}
    else if((ch==='\n'||ch==='\r')&&!quote){if(ch==='\r'&&text[i+1]==='\n')i++;row.push(value.trim());if(row.some(v=>v))rows.push(row);row=[];value='';}else value+=ch;}
  if(quote)throw Error('CSV의 따옴표를 확인해 주세요.');row.push(value.trim());if(row.some(v=>v))rows.push(row);
  const header=rows.shift()||[],keys=['id','stack','tier',...Array.from({length:12},(_,i)=>`p${i}`),'order'];
  if(keys.some(k=>!header.includes(k)))throw Error('양식의 id, stack, tier, p0~p11, order 열이 필요합니다.');
  if(rows.length<6||rows.length>40)throw Error('데이터 행은 6~40개여야 합니다.');
  const records=rows.map(r=>Object.fromEntries(header.map((k,i)=>[k,r[i]??''])));
  if(records.some(r=>keys.some(k=>r[k]==='')))throw Error('필수 값이 비어 있습니다.');
  const labels=records.map(r=>r.id),maxStack=Math.max(...records.map(r=>Number(r.stack))),count=header.includes('stack_count')?Number(records[0].stack_count):maxStack;
  if(header.includes('stack_count')&&records.some(r=>Number(r.stack_count)!==count))throw Error('stack_count는 모든 행에서 같은 전체 적재열 수여야 합니다.');
  if(maxStack>count)throw Error('stack이 전체 적재열 수를 초과했습니다.');
  if(!Number.isInteger(count)||count<4||count>10)throw Error('stack은 1~10의 정수이며 4개 이상의 열이 필요합니다.');
  const stacks=Array.from({length:count},()=>[]);
  records.forEach((r,i)=>{const s=Number(r.stack),t=Number(r.tier);if(!Number.isInteger(s)||s<1||!Number.isInteger(t)||t<1||t>h)throw Error('stack과 tier를 확인해 주세요.');if(stacks[s-1][t-1]!==undefined)throw Error('동일 위치에 컨테이너가 중복되었습니다.');stacks[s-1][t-1]=i;});
  for(const s of stacks)if(Array.from({length:s.length},(_,i)=>s[i]).includes(undefined))throw Error('각 열의 tier는 바닥 1단부터 빈칸 없이 입력해 주세요.');
  const orders=records.map(r=>Number(r.order));
  if(orders.some(v=>!Number.isInteger(v)||v<1||v>records.length)||new Set(orders).size!==records.length)throw Error('order는 1부터 컨테이너 수까지 중복 없이 입력해 주세요.');
  const order=records.map((_,i)=>i).sort((a,b)=>orders[a]-orders[b]);
  return validateCase({id:'사용자 CSV',custom:true,height:h,stacks,labels,order,p:records.map(r=>Array.from({length:12},(_,k)=>Number(r[`p${k}`])))});
}
export function toCSV(c){
  const rows=[['id','stack','tier',...Array.from({length:12},(_,i)=>`p${i}`),'order','stack_count']];
  c.stacks.forEach((s,j)=>s.forEach((v,t)=>rows.push([c.labels[v],j+1,t+1,...c.p[v],c.order.indexOf(v)+1,c.stacks.length])));
  return '\uFEFF'+rows.map(row=>row.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\r\n');
}
