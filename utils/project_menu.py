"""Browser-native project picker and independently editable menu title."""
import streamlit.components.v2 as components

_definition = dict(html='''
<div class="bar"><button class="hamburger" aria-label="Project menu" aria-expanded="false">☰</button>
<button class="name" title="Rename project"></button><input class="rename" aria-label="Project name" hidden></div>
<div class="menu" hidden><button data-action="new">New project</button><button data-action="open">Open Project</button>
<button data-action="save">Save project</button><button data-action="close">Close project</button></div>
<input class="picker" type="file" accept=".cozy" hidden><div class="error" role="alert"></div><hr>
''', css='''
:host {font-family: inherit; color: inherit;} .bar {display:flex;align-items:center;gap:8px;}
button,input {font:inherit;color:inherit;} button {cursor:pointer;background:transparent;border:0;padding:8px;}
button:hover {background:#80808018;} .name {font-size:1.1rem;text-align:left;}
.hamburger {font-size:1.3rem;} .menu {border:1px solid #8885;border-radius:6px;padding:4px;width:190px;}
.menu button {display:block;width:100%;text-align:left;} .rename {padding:7px;border:1px solid #8888;border-radius:4px;}
[hidden] {display:none!important;} .error {color:#d33;}
hr {margin:4px 0 0;border:0;border-top:1px solid #8884;}
''', js='''
export default function({parentElement: root, data, setTriggerValue}) {
 const hamburger=root.querySelector('.hamburger'), menu=root.querySelector('.menu'), name=root.querySelector('.name'),
 input=root.querySelector('.rename'), picker=root.querySelector('.picker'), error=root.querySelector('.error');
 name.textContent=data.name;
 const send=(event)=>setTriggerValue('event',{...event,id:crypto.randomUUID()});
 const collapse=()=>{menu.hidden=true;hamburger.setAttribute('aria-expanded','false');};
 hamburger.onclick=()=>{menu.hidden=!menu.hidden;hamburger.setAttribute('aria-expanded',String(!menu.hidden));};
 name.onclick=()=>{input.value=data.name;name.hidden=true;input.hidden=false;input.focus();input.select();};
 const finish=(commit)=>{if(input.hidden)return;const value=input.value.trim();input.hidden=true;name.hidden=false;
   if(commit&&value&&value!==data.name){name.textContent=value;send({action:'rename',name:value});}};
 input.onblur=()=>finish(true);input.onkeydown=(e)=>{if(e.key==='Enter'){e.preventDefault();finish(true);}if(e.key==='Escape')finish(false);};
 menu.querySelectorAll('button').forEach(button=>button.onclick=()=>{collapse();error.textContent='';
   if(button.dataset.action==='open'){picker.value='';picker.click();}
   else send({action:button.dataset.action});});
 picker.onchange=()=>{const file=picker.files[0];if(!file)return;
   if(!file.name.toLowerCase().endsWith('.cozy')||file.size>data.maxBytes){error.textContent='Choose a .cozy project under 512 MB.';return;}
   const reader=new FileReader();reader.onerror=()=>{error.textContent='Cannot read this file.';};
   reader.onload=()=>send({action:'open',content:reader.result.split(',')[1]});reader.readAsDataURL(file);};
 const outside=(e)=>{if(!e.composedPath().includes(hamburger)&&!e.composedPath().includes(menu))collapse();};
 document.addEventListener('click',outside);
 return ()=>document.removeEventListener('click',outside);
}
''')


def render_menu(name, max_bytes):
    menu = components.component('cozy_project_menu', **_definition)
    result = menu(key='project_menu', data={'name': name, 'maxBytes': max_bytes}, on_event_change=lambda: None)
    return result.event
