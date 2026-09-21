import { compute, initialState, meta } from './core.mjs';
const form=document.querySelector('#workspace-form');
const resultPanel=document.querySelector('#results');
const target=document.querySelector('#result-content');
const reset=document.querySelector('#reset-button');
const fields=new Map(meta.fields.map(field=>[field.name,field]));
const escapeHtml=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function readInput(){const input={};for(const [name,field] of fields){const element=form.elements.namedItem(name);input[name]=field.type==='checkbox'?element.checked:field.type==='number'?Number(element.value):element.value}return input}
function renderTable(rows){if(!rows?.length)return'';const columns=[...new Set(rows.flatMap(row=>Object.keys(row)))];return '<div class="table-wrap"><table><thead><tr>'+columns.map(key=>'<th>'+escapeHtml(key.replace(/([A-Z])/g,' $1'))+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+columns.map(key=>'<td>'+escapeHtml(row[key]??'—')+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>'}
function render(result){target.innerHTML='<div class="status-line"><span class="status-dot"></span><h3>'+escapeHtml(result.status)+'</h3></div><p class="summary">'+escapeHtml(result.summary)+'</p><div class="metrics">'+result.metrics.map(metric=>'<div class="metric"><span>'+escapeHtml(metric.label)+'</span><strong>'+escapeHtml(metric.value)+'</strong></div>').join('')+'</div>'+renderTable(result.rows)+(result.detail?'<p class="detail">'+escapeHtml(result.detail)+'</p>':'');resultPanel.setAttribute('aria-busy','false')}
async function run(){resultPanel.setAttribute('aria-busy','true');target.innerHTML='<div class="loading">Computing scenario…</div>';try{const input=readInput();render(await compute(input))}catch(error){target.innerHTML='<div class="error"><strong>Unable to compute.</strong><br>'+escapeHtml(error.message)+'</div>';resultPanel.setAttribute('aria-busy','false')}}
form.addEventListener('submit',event=>{event.preventDefault();run()});
reset.addEventListener('click',()=>{for(const [name,field] of fields){const element=form.elements.namedItem(name);if(field.type==='checkbox')element.checked=Boolean(initialState[name]);else element.value=initialState[name]}run()});
await run();