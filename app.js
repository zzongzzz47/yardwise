import {clone,decodeModel,stats,recommend,applyAction,simulate,validateCase,parseCSV,toCSV} from './engine.js';
const $=id=>document.getElementById(id),esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const app={data:null,model:null,c:null,state:[],idx:0,relocations:0,log:[],undo:[],selected:-1,playing:false,busy:false,timer:null,action:null,inspected:null,token:0};
const num=n=>n.toLocaleString('ko-KR'),stack=j=>j<0?'출고':`S${String(j+1).padStart(2,'0')}`;
const iconArrow='<span aria-hidden="true"> → </span>';
function toast(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(app.toastTimer);app.toastTimer=setTimeout(()=>$('toast').hidden=true,3500);}
function download(name,text,type='text/csv;charset=utf-8'){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function pause(){app.playing=false;app.token++;clearTimeout(app.timer);$('playBtn').textContent='▶ 자동 시연';}
function changeView(view){pause();document.querySelectorAll('.view').forEach(e=>e.hidden=e.id!==view);document.querySelectorAll('.nav').forEach(e=>e.classList.toggle('active',e.dataset.view===view));history.replaceState(null,'',`#${view}`);window.scrollTo({top:0,behavior:'instant'});}
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>changeView(b.dataset.view)));
document.querySelector('.brand').addEventListener('click',e=>{e.preventDefault();changeView('workspace');});
function snapshot(){app.undo.push({c:clone(app.c),state:clone(app.state),idx:app.idx,relocations:app.relocations,log:clone(app.log),inspected:app.inspected});}
function loadCase(c){if(app.busy){toast('진행 중인 이동이 끝난 뒤 배치를 바꿔 주세요.');return;}pause();app.c=clone(c);app.state=clone(c.stacks);app.ctx=stats(c.p);app.idx=0;app.relocations=0;app.log=[];app.undo=[];app.inspected=c.order[0];app.selected=-1;app.busy=false;
  const customOption=$('caseSelect').querySelector('[data-custom]');if(customOption)customOption.remove();
  if(c.custom){const o=new Option(c.id,'custom');o.dataset.custom='true';$('caseSelect').add(o);$('caseSelect').value='custom';}else $('caseSelect').value=c.id;
  render();renderComparison();
}
function setControls(){
  const busy=app.busy,done=app.idx>=app.c.order.length;
  for(const id of ['caseSelect','targetSelect','editBtn','uploadBtn','resetBtn','templateBtn'])$(id).disabled=busy;
  $('targetSelect').disabled=busy||done;$('undoBtn').disabled=busy||!app.undo.length;
  $('approveBtn').disabled=busy||done;$('playBtn').disabled=done;
  $('exportLog').disabled=!app.log.length;$('playBtn').textContent=app.playing?'Ⅱ 시연 일시정지':'▶ 자동 시연';
  document.querySelectorAll('.candidate').forEach(b=>b.disabled=busy);
}
function render(){
  const c=app.c,done=app.idx>=c.order.length,target=c.order[app.idx];
  app.action=done?null:recommend(app.state,target,c.height,c.p,app.model,'learned',app.ctx);
  if(app.action?.type==='move'){if(!app.action.candidates.some(r=>r.j===app.selected))app.selected=app.action.dest;}else app.selected=-1;
  $('caseBadge').textContent=c.custom?'사용자 수정 시나리오':'공개 연구 자료';
  $('caseDesc').textContent=`${c.stacks.length}개 적재열 · 최대 ${c.height}단 · ${c.labels.length}개 컨테이너`;
  $('remaining').textContent=app.state.flat().length;$('retrieved').textContent=app.idx;$('totalBoxes').textContent=`/ ${c.labels.length}개`;$('relocated').textContent=app.relocations;
  $('statusText').textContent=done?'모든 출고 완료':app.playing?'자동 시연 중':app.action.type==='retrieve'?'바로 출고 가능':'이동 위치 검토';
  $('statusSub').textContent=done?'작업 이력을 저장할 수 있습니다.':`${stack(app.action.src)}에서 ${c.labels[target]} 출고 요청`;
  $('liveBadge').textContent=done?'완료':app.playing?'시연 중':'검토 중';
  $('targetSelect').replaceChildren(...(done?[new Option('모든 출고 완료','')]:c.order.slice(app.idx).map(id=>new Option(`${c.labels[id]} · ${stack(app.state.findIndex(s=>s.includes(id)))}`,id))));
  $('targetInfo').textContent=done?`${c.labels.length}개 컨테이너 출고 완료`:`${app.idx+1}번째 요청 · 선택하면 다음 출고 대상을 바꿀 수 있습니다.`;
  $('yardMeta').textContent=`${c.stacks.length}개 적재열 / 최대 ${c.height}단 · 컨테이너를 눌러 정보 확인`;
  $('progressFill').style.width=`${100*app.idx/c.labels.length}%`;$('progressText').textContent=`${app.idx} / ${c.labels.length} 출고`;
  $('candidates').replaceChildren();$('scoreNote').hidden=true;$('reasonLabel').hidden=true;
  if(done){$('actionTitle').textContent='작업을 모두 마쳤어요';$('actionDesc').textContent=`재취급 ${app.relocations}회, 필수 출고 ${app.idx}회가 기록되었습니다.`;$('approveBtn').textContent='출고 완료';}
  else if(app.action.type==='retrieve'){
    $('actionTitle').textContent='추가 이동 없이 출고';$('actionDesc').textContent=`${c.labels[target]}가 최상단에 있습니다. 다른 컨테이너를 옮기지 않고 바로 출고할 수 있습니다.`;
    $('approveBtn').textContent=`${c.labels[target]} 출고 승인`;
  }else{
    const a=app.action;$('actionTitle').textContent=`${stack(a.dest)} 이동을 추천해요`;
    $('actionDesc').textContent=`${c.labels[target]} 위에 있는 ${c.labels[a.container]}를 먼저 옮깁니다. 가능한 위치를 비교하고 선택하세요.`;
    a.candidates.forEach((r,i)=>{const b=document.createElement('button');b.type='button';b.className='candidate'+(app.selected===r.j?' selected':'');b.setAttribute('aria-pressed',app.selected===r.j);b.dataset.dest=r.j;
      b.innerHTML=`<span>${stack(r.j)}${i===0?'<em>AI 추천</em>':''}<br><small>현재 ${app.state[r.j].length} / ${c.height}단</small></span><strong>${r.score.toFixed(3)}</strong>`;
      b.addEventListener('click',()=>{if(app.busy)return;pause();app.selected=r.j;render();});$('candidates').append(b);});
    $('scoreNote').hidden=false;$('reasonLabel').hidden=false;$('approveBtn').textContent=`${c.labels[a.container]} → ${stack(app.selected)} 이동 승인`;
  }
  renderYard();renderInspector();renderHistory();setControls();
}
function layout(){const n=app.c.stacks.length,h=app.c.height;return {x0:58,dx:786/n,w:Math.min(100,786/n-15),base:331,bh:Math.min(48,238/h),gap:5};}
function position(j,t){const l=layout();return {x:l.x0+j*l.dx+(l.dx-l.w)/2,y:l.base-(t+1)*(l.bh+l.gap),w:l.w,h:l.bh};}
function renderYard(){
  const c=app.c,l=layout(),a=app.action;let s=`<defs><marker id="arrowHead" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 Z" fill="#138c7e"/></marker><pattern id="hatch" patternUnits="userSpaceOnUse" width="6" height="6"><path d="M0 6 L6 0" stroke="#aad1c5" stroke-width=".6"/></pattern></defs>`;
  s+='<path d="M35 45 H865 M35 50 H865" stroke="#cedde1" stroke-width="2"/><path d="M42 42 V341 M858 42 V341" stroke="#dbe6e9" stroke-width="8"/>';
  for(let tier=0;tier<c.height;tier++){const y=position(0,tier).y+l.bh;s+=`<line x1="53" x2="847" y1="${y}" y2="${y}" stroke="#dfe9eb" stroke-dasharray="3 5"/><text x="24" y="${y-14}" font-size="12" fill="#9aadb4">${tier+1}</text>`;}
  c.stacks.forEach((_,j)=>{
    const x=position(j,0).x;
    s+=`<rect x="${x-5}" y="340" width="${l.w+10}" height="7" rx="1" fill="#c9d8de"/><text x="${x+l.w/2}" y="372" text-anchor="middle" fill="#506c78" font-size="15" font-weight="600">${stack(j)}</text>`;
    app.state[j].forEach((id,t)=>{const p=position(j,t),target=a?.target===id,blocker=a?.type==='move'&&a.container===id,inspect=app.inspected===id;
      const mu=app.ctx.mu[id];let fill=['#7b9dae','#8aa9b7','#98b6bd','#a7c3c7'][Math.min(3,Math.floor(mu/3))];if(target)fill='#e8b34b';if(blocker)fill='#d38351';
      const label=c.labels[id],short=label.length>10?label.slice(0,9)+'…':label;
      s+=`<g class="box${app.busy&&a?.container===id?' moving':''}" data-id="${id}" role="button" tabindex="0" aria-label="${esc(label)}, ${stack(j)}, ${t+1}단, 출고 확률 확인"><title>${esc(label)} · ${stack(j)} ${t+1}단</title><path d="M${p.x},${p.y} l6,-5 h${p.w-6} v5Z" fill="${fill}" opacity=".7"/><rect x="${p.x}" y="${p.y}" width="${p.w}" height="${p.h}" rx="2" fill="${fill}" stroke="${inspect?'#314e5b':'#fff'}" stroke-width="${inspect?2.5:1.3}"/>`;
      for(let q=1;q<7;q++)s+=`<line x1="${p.x+p.w*q/7}" x2="${p.x+p.w*q/7}" y1="${p.y+5}" y2="${p.y+p.h-4}" stroke="#fff" opacity=".10"/>`;
      s+=`<text x="${p.x+p.w/2}" y="${p.y+p.h/2+5}" text-anchor="middle" fill="#fff" font-size="${c.stacks.length>8?12:14}" font-weight="700">${esc(short)}</text></g>`;
    });
  });
  if(a?.type==='move'){
    const from=position(a.src,app.state[a.src].length-1),to=position(app.selected,app.state[app.selected].length);
    s+=`<rect x="${to.x}" y="${to.y}" width="${to.w}" height="${to.h}" rx="3" fill="url(#hatch)" stroke="#138c7e" stroke-width="2" stroke-dasharray="6 3"/><text x="${to.x+to.w/2}" y="${to.y+to.h/2+5}" text-anchor="middle" font-size="12" fill="#0c7c6b" font-weight="700">선택 위치</text>`;
    const sx=from.x+from.w/2,tx=to.x+to.w/2;
    s+=`<path d="M${sx} ${from.y-8} V69 Q${sx} 59 ${sx+(tx>sx?10:-10)} 59 H${tx-(tx>sx?10:-10)} Q${tx} 59 ${tx} 69 V${to.y-10}" fill="none" stroke="#138c7e" stroke-width="2" marker-end="url(#arrowHead)" stroke-dasharray="5 3"/>`;
    s+=`<rect x="${(sx+tx)/2-41}" y="47" width="82" height="24" rx="12" fill="#e0f2eb"/><text x="${(sx+tx)/2}" y="63" text-anchor="middle" fill="#157b6a" font-size="11">이동 경로</text>`;
  }else if(a){const p=position(a.src,app.state[a.src].length-1);s+=`<text x="${p.x+p.w/2}" y="${p.y-17}" text-anchor="middle" fill="#ad802d" font-size="13" font-weight="700">출고 ↑</text>`;}
  if(!a)s+='<text x="450" y="180" text-anchor="middle" fill="#487c72" font-size="23" font-weight="600">모든 컨테이너의 출고를 마쳤습니다.</text>';
  $('yard').innerHTML=s;$('yard').setAttribute('aria-label',`적재 상태: ${app.state.flat().length}개 컨테이너. ${a?`출고 대상 ${c.labels[a.target]}`:'출고 완료'}`);
  $('yard').querySelectorAll('[data-id]').forEach(g=>{const select=()=>{app.inspected=Number(g.dataset.id);renderInspector();renderYard();};g.addEventListener('click',select);g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}});});
}
function renderInspector(){
  const c=app.c,id=app.inspected;if(id===null||!c.p[id])return;
  const j=app.state.findIndex(s=>s.includes(id)),p=c.p[id],peak=p.indexOf(Math.max(...p));
  $('inspector').innerHTML=`<span class="inspect-label">출고 확률</span><div class="inspect-info"><strong>${esc(c.labels[id])}</strong><small>${j<0?'출고 완료':`${stack(j)} · ${app.state[j].indexOf(id)+1}단`} / 최빈 구간 ${peak}<br>구간 0~11 · 출고 순번과 다릅니다.</small></div><div class="prob-bars" role="img" aria-label="${esc(c.labels[id])}의 구간별 출고 확률">${p.map((v,i)=>`<div title="구간 ${i}: ${(v*100).toFixed(1)}%"><i style="height:${Math.max(1,v*29)}px"></i><span>${i}</span></div>`).join('')}</div>`;
}
function renderHistory(){
  $('historyBody').innerHTML=app.log.length?app.log.slice().reverse().map((r,idx)=>`<tr><td>${app.log.length-idx}</td><td>${r.type==='move'?'재취급':r.type==='retrieve'?'출고':'요청 변경'}</td><td>${esc(r.label)}</td><td>${r.type==='request'?'다음 출고로 지정':stack(r.from)+' → '+stack(r.to)}</td><td title="${esc(r.note||'')}"><span class="${r.selection==='담당자'?'manual-tag':'ai-tag'}">${esc(r.selection)}</span></td></tr>`).join(''):'<tr><td colspan="5" class="empty">승인한 작업이 여기에 기록됩니다.</td></tr>';
}
function compareRow(label,value,max,muted=false,color=null){return `<div class="compare-row${muted?' muted':''}"><span>${esc(label)}</span><div class="compare-track"><i style="width:${max?value/max*100:0}%;${color?'background:'+color:''}"></i></div><strong>${num(value)}회</strong></div>`;}
function renderComparison(){
  const ai=simulate(app.c,app.model,'learned').relocations,base=simulate(app.c,app.model,'mean').relocations,diff=base-ai,max=Math.max(ai,base,1);
  $('scenarioCompare').innerHTML=compareRow('출고 우선순위',base,max,true)+compareRow('학습한 AI',ai,max)+`<div class="comparison-outcome">${diff>0?`AI에서 재취급 <b>${diff}회 감소</b>`:diff===0?'두 방법의 재취급 횟수가 같습니다.':`이 배치에서는 AI에서 재취급 <b>${-diff}회 증가</b>`}</div>`;
  $('comparisonNote').textContent=app.c.custom?'사용자가 입력한 배치와 출고 순서에서 계산한 시나리오 결과입니다.':'선택한 공개 연구 배치의 계산 결과입니다. 담당자의 다른 위치 선택은 위의 현재 작업 이력에 반영됩니다.';
}
async function animateAction(a,dest){
  if(matchMedia('(prefers-reduced-motion: reduce)').matches)return;
  const r=$('yard').getBoundingClientRect(),from=position(a.src,app.state[a.src].length-1),to=a.type==='move'?position(dest,app.state[dest].length):{x:850,y:0,w:40,h:20};
  const scale=r.width/900,box=document.createElement('div');box.className='flightbox';box.textContent=app.c.labels[a.container];box.style.width=`${from.w*scale}px`;box.style.height=`${from.h*scale}px`;
  $('motion').append(box);const sx=from.x*scale,sy=from.y*scale,tx=to.x*scale,ty=to.y*scale,top=30*scale;
  const animation=box.animate([{transform:`translate(${sx}px,${sy}px)`},{transform:`translate(${sx}px,${top}px)`,offset:.3},{transform:`translate(${tx}px,${top}px)`,offset:.7},{transform:`translate(${tx}px,${ty}px)`}],{duration:Math.min(650,Number($('speed').value)*.7),easing:'ease-in-out',fill:'forwards'});
  try{await animation.finished;}finally{box.remove();}
}
async function approve(auto=false){
  if(app.busy||!app.action)return;if(!auto)pause();const a=app.action,dest=auto?a.dest:app.selected,note=$('reason').value.trim();
  snapshot();app.busy=true;document.body.classList.add('animating');setControls();renderYard();
  try{await animateAction(a,dest);applyAction(app.state,a,dest);
    if(a.type==='move')app.relocations++;else app.idx++;
    app.log.push({type:a.type,label:app.c.labels[a.container],from:a.src,to:a.type==='retrieve'?-1:dest,
      selection:a.type==='retrieve'?'직접 출고':dest===a.dest?'AI 추천':'담당자',note,score:a.candidates.find(r=>r.j===dest)?.score??'',step:app.log.length+1});
    app.selected=-1;$('reason').value='';app.inspected=app.c.order[app.idx]??a.container;
  }catch(e){pause();toast(e.message);}finally{app.busy=false;document.body.classList.remove('animating');if(app.idx===app.c.order.length)pause();render();}
}
function play(){if(app.playing){pause();render();return;}app.playing=true;const token=++app.token;
  const tick=async()=>{if(!app.playing||token!==app.token)return;if(app.busy){app.timer=setTimeout(tick,150);return;}
    if(!app.action){pause();render();return;}await approve(true);if(app.playing&&token===app.token)app.timer=setTimeout(tick,Number($('speed').value));};render();tick();}
$('approveBtn').addEventListener('click',()=>approve());$('playBtn').addEventListener('click',play);
$('caseSelect').addEventListener('change',()=>{const c=app.data.cases.find(c=>c.id===$('caseSelect').value);if(c)loadCase(c);});
$('resetBtn').addEventListener('click',()=>{loadCase(app.c);toast('초기 배치로 돌아왔습니다.');});
$('undoBtn').addEventListener('click',()=>{if(app.busy||!app.undo.length)return;pause();const old=app.undo.pop();Object.assign(app,old);app.ctx=stats(app.c.p);app.selected=-1;render();renderComparison();});
$('targetSelect').addEventListener('change',()=>{if(app.busy)return;pause();const target=Number($('targetSelect').value);if(target===app.c.order[app.idx])return;snapshot();
  const remaining=app.c.order.slice(app.idx).filter(id=>id!==target);app.c.order=[...app.c.order.slice(0,app.idx),target,...remaining];app.c.custom=true;
  app.log.push({type:'request',label:app.c.labels[target],from:-1,to:-1,selection:'담당자',note:'다음 출고 요청 변경'});app.selected=-1;app.inspected=target;render();renderComparison();toast('다음 출고 요청을 변경했습니다.');});
function exportCase(){download('yardwise_input.csv',toCSV(app.c));}
$('templateBtn').addEventListener('click',exportCase);$('downloadTemplate').addEventListener('click',exportCase);
$('exportLog').addEventListener('click',()=>{const rows=[['순서','작업','컨테이너','출발','도착','선택','상대비용점수','메모']];app.log.forEach((r,i)=>rows.push([i+1,r.type==='move'?'재취급':r.type==='retrieve'?'출고':'요청변경',r.label,stack(r.from),stack(r.to),r.selection,r.score??'',r.note??'']));
  const csv='\uFEFF'+rows.map(r=>r.map(v=>{let s=String(v);if(/^[=+@\-]/.test(s.trimStart()))s="'"+s;return '"'+s.replaceAll('"','""')+'"';}).join(',')).join('\r\n');download('yardwise_decisions.csv',csv);toast('작업 이력을 저장했습니다.');});
document.querySelectorAll('.closeDialog').forEach(b=>b.addEventListener('click',()=>b.closest('dialog').close()));
$('editBtn').addEventListener('click',()=>{if(!app.c)return;pause();render();$('editHeight').value=app.c.height;$('layoutInput').value=app.c.stacks.map(s=>s.map(id=>app.c.labels[id]).join(', ')).join('\n');$('orderInput').value=app.c.order.map(id=>app.c.labels[id]).join(', ');$('editError').hidden=true;$('editDialog').showModal();});
$('editForm').addEventListener('submit',e=>{e.preventDefault();try{const c=clone(app.c),index=new Map(c.labels.map((id,i)=>[id,i])),parse=s=>s.trim()?s.split(/[\s,]+/).filter(Boolean).map(id=>{if(!index.has(id))throw Error(`알 수 없는 ID: ${id}`);return index.get(id);}):[];
    c.stacks=$('layoutInput').value.replaceAll('\r','').split('\n').map(parse);c.order=parse($('orderInput').value);c.height=Number($('editHeight').value);c.id='사용자 편집 배치';c.custom=true;validateCase(c);loadCase(c);$('editDialog').close();toast('수정한 초기 배치에서 추천을 계산했습니다.');
  }catch(err){$('editError').textContent=err.message;$('editError').hidden=false;}});
$('uploadBtn').addEventListener('click',()=>{if(!app.c||app.busy)return;pause();render();$('uploadHeight').value=app.c.height;$('uploadError').hidden=true;$('csvInput').value='';$('uploadDialog').showModal();});
$('csvInput').addEventListener('change',async()=>{const f=$('csvInput').files[0];if(!f)return;try{if(f.size>2*1024*1024)throw Error('2 MB 이하의 CSV 파일을 선택해 주세요.');const c=parseCSV(await f.text(),Number($('uploadHeight').value));loadCase(c);$('uploadDialog').close();toast('CSV 배치를 불러왔습니다.');}catch(err){$('uploadError').textContent=err.message;$('uploadError').hidden=false;$('csvInput').value='';}});
function renderBenchmarks(){
  const m=app.data.metrics,t=app.data.training,model=app.data.model;
  const ai=app.data.summary.find(r=>r.policy==='learned'),search=app.data.summary.find(r=>r.policy==='lookahead');
  $('modelVersion').textContent=`V2 · ${num(t.rows)}건 학습`;
  $('savingPercent').textContent=m.relocation_saving_percent.toFixed(1);
  $('allMovesPercent').textContent=m.all_moves_saving_percent.toFixed(1);
  $('mandatoryCount').textContent=`필수 출고 ${num(m.containers)}회 포함`;
  $('betterCases').textContent=num(m.better);$('caseTotal').textContent=` / ${num(m.cases)}`;
  $('caseOutcomes').textContent=`동일 ${num(m.same)}개 · 증가 ${num(m.worse)}개`;
  $('riskComparison').textContent=`같은 출고 확률을 활용하는 확률 위험 규칙과 비교해도 재취급이 ${m.informed_risk_saving_percent.toFixed(1)}% 줄었습니다.`;
  $('modelComparison').innerHTML=`<tr><td>학습한 AI</td><td>${num(ai.total)}회</td><td>${ai.mean_decision_ms.toFixed(2)}ms</td></tr><tr><td>미래 탐색 24개</td><td>${num(search.total)}회</td><td>${search.mean_decision_ms.toFixed(2)}ms</td></tr>`;
  const gap=ai.total-search.total;
  $('searchComparison').textContent=`${gap>0?`미래 탐색에서 재취급이 ${num(gap)}회 더 적었습니다.`:gap<0?`AI에서 재취급이 ${num(-gap)}회 더 적었습니다.`:'두 방법의 재취급 횟수는 같았습니다.'} AI의 계산시간은 미래 탐색의 약 ${(100*ai.mean_decision_ms/search.mean_decision_ms).toFixed(0)}%였습니다. 이 화면의 실시간 추천에는 새로 학습한 AI를 사용합니다.`;
  $('trainingDescription').textContent=`학습 원본 ${num(t.original_cases)}개 각각을 높이 우선, 출고 우선순위, 확률 위험, 무작위 이동의 네 방식으로 독립적으로 진행했습니다. 총 ${num(t.trajectories)}개 경로에서 발생한 이동 판단 ${num(t.decisions)}번을 건너뛰지 않고 평가하여 후보별 학습자료 ${num(t.rows)}건을 만들었습니다.`;
  const routeNames={height:'높이 우선',mean:'출고 우선순위',risk:'확률 위험',random:'무작위 이동'};
  $('trainingRoutes').innerHTML=Object.entries(t.by_route).map(([key,r])=>`<tr><td>${routeNames[key]}</td><td>${num(r.trajectories)}</td><td>${num(r.decisions)}</td><td>${num(r.rows)}</td></tr>`).join('');
  $('modelProvenance').textContent=`모형 갱신 ${model.updated_date} · ${model.trees}개 회귀나무 · 입력 변수 ${model.features}개`;
  const names={height:'높이 우선 규칙',mean:'출고 우선순위 규칙',risk:'확률 위험 규칙',learned:'학습한 AI',lookahead:'미래 탐색 (24개)'},max=Math.max(...app.data.summary.map(r=>r.total));
  $('benchmarkBars').innerHTML=app.data.summary.map(r=>compareRow(names[r.policy],r.total,max,r.policy!=='learned',r.policy==='lookahead'?'#b28748':null)).join('');
  renderCaseTable();
}
function renderCaseTable(){const filter=$('resultFilter').value;
  const rows=app.data.cases.filter(c=>{const d=c.results.mean-c.results.learned;return filter==='all'||(filter==='better'&&d>0)||(filter==='same'&&d===0)||(filter==='worse'&&d<0);});
  $('caseTable').innerHTML=rows.map(c=>{const d=c.results.mean-c.results.learned;return `<tr><td>${esc(c.id)}</td><td>${c.stacks.length}열 · ${c.height}단 · ${c.labels.length}개</td><td>${c.results.mean}회</td><td>${c.results.learned}회</td><td class="${d>=0?'ai-tag':'manual-tag'}">${d>0?d+'회 감소':d<0?-d+'회 증가':'동일'}</td><td><button data-case="${esc(c.id)}">이 배치 열기 ↗</button></td></tr>`;}).join('');
  $('caseTable').querySelectorAll('[data-case]').forEach(b=>b.addEventListener('click',()=>{loadCase(app.data.cases.find(c=>c.id===b.dataset.case));changeView('workspace');}));
}
$('resultFilter').addEventListener('change',()=>{if(app.data)renderCaseTable();});
// BEGIN_ASSET_LOADER
async function loadAssets(){
  const dr=await fetch('assets/data.json',{cache:'no-store'});
  if(!dr.ok)throw Error('분석 자료를 불러오지 못했습니다. 새로고침해 주세요.');
  const data=await dr.json();
  const chunks=await Promise.all(data.model.parts.map(async part=>{
    const response=await fetch('assets/'+part.file,{cache:'no-store'});
    if(!response.ok)throw Error('학습 모형을 불러오지 못했습니다. 새로고침해 주세요.');
    const bytes=new Uint8Array(await response.arrayBuffer());
    if(bytes.length!==part.bytes)throw Error('학습 모형의 파일 크기가 일치하지 않습니다. 새로고침해 주세요.');
    return bytes;
  }));
  const bytes=new Uint8Array(data.model.bytes);let offset=0;
  chunks.forEach(chunk=>{bytes.set(chunk,offset);offset+=chunk.length;});
  if(offset!==bytes.length)throw Error('학습 모형의 일부가 누락되었습니다.');
  return {data,buffer:bytes.buffer};
}
// END_ASSET_LOADER
async function init(){try{
  const assets=await loadAssets();app.data=assets.data;app.model=decodeModel(assets.buffer);
  const sorted=app.data.cases.slice().sort((a,b)=>a.stacks.length-b.stacks.length||a.height-b.height||a.labels.length-b.labels.length||a.id.localeCompare(b.id));
  $('caseSelect').replaceChildren(...sorted.map(c=>new Option(`${c.stacks.length}열 · ${c.height}단 · ${c.labels.length}개  /  ${c.id}`,c.id)));
  const chosen=new URLSearchParams(location.search).get('case');loadCase(app.data.cases.find(c=>c.id===chosen)||app.data.cases.find(c=>c.id==='6-4-16-8'));renderBenchmarks();
  const view=location.hash.slice(1);if(['results','guide'].includes(view))changeView(view);
}catch(e){$('loadError').textContent=e.message;$('loadError').hidden=false;$('statusText').textContent='불러오기 실패';$('actionTitle').textContent='자료를 확인해 주세요';$('actionDesc').textContent='네트워크 연결을 확인한 뒤 페이지를 새로고침해 주세요.';console.error(e);}}
init();
