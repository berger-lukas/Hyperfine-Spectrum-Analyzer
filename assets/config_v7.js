(() => {
  if (window.configManagerInstalled) return;
  window.configManagerInstalled = true;
  let form, picker, model, selected = new Set(), browsePath = '', browseKind = 'spec';
  const el = (tag, text, cls) => { const n = document.createElement(tag); if (text) n.textContent = text; if (cls) n.className = cls; return n; };
  const button = (text, action) => { const b = el('button', text); b.type = 'button'; b.onclick = action; return b; };
  async function api(path, body) {
    const response = await fetch('/config-api/' + path, body ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {});
    const data = await response.json(); if (!response.ok) throw Error(data.error || 'Request failed'); return data;
  }
  function error(message) { form.querySelector('.cfg-error').textContent = message; }
  function dirty() {
    return form && (form.querySelector('#cfg-name').value !== model.name || form.querySelector('#cfg-spec').value !== model.csv_file || JSON.stringify(model.cat_files) !== model.originalCats);
  }
  function close() {
    if (dirty() && !confirm('Discard unsaved configuration changes?')) return;
    form.close(); form.remove(); form = null;
  }
  function rows() {
    const list = form.querySelector('#cfg-cats'); list.replaceChildren();
    model.cat_files.forEach((path, index) => {
      const row = el('div', null, 'cfg-cat-row'); const input = el('input'); input.value = path; input.setAttribute('aria-label', 'CAT path ' + (index+1));
      input.oninput = () => { model.cat_files[index] = input.value; };
      row.append(el('span', String(index+1)), input, button('Browse…', () => browse('cat', index)), button('Remove', () => {model.cat_files.splice(index,1); rows();})); list.append(row);
    });
    form.querySelector('#cfg-count').textContent = model.cat_files.length + ' CAT files • add as many as needed';
  }
  async function open(key) {
    if (form) return;
    try {
      model = key ? await api('config/' + key) : {name:'', csv_file:'', cat_files:[]};
      model.originalCats = JSON.stringify(model.cat_files);
      form = el('dialog', null, 'cfg-dialog');
      form.innerHTML = '<h2></h2><p>One spectrum configuration contains one experimental spectrum and any number of CAT files.</p><label for="cfg-name">1. Configuration name</label><input id="cfg-name" autocomplete="off"><label for="cfg-spec">2. Experimental spectrum</label><div class="cfg-spec-row"><input id="cfg-spec" placeholder="Select a spectrum CSV / TXT / DAT / TSV"></div><p class="cfg-help">First two columns: frequency (MHz), intensity. A header row is expected; delimiter is detected automatically for a new spectrum.</p><label>3. CAT files</label><div id="cfg-cats"></div><div id="cfg-add"></div><p id="cfg-count"></p><p class="cfg-help">Existing FID, uncertainty and quantum-number settings are retained. New configurations use FID 10 µs and base uncertainty 0.01 MHz.</p><p class="cfg-error" role="alert"></p><div class="cfg-actions"></div>';
      form.querySelector('h2').textContent = key ? 'Edit Configuration' : 'New Configuration';
      form.querySelector('#cfg-name').value = model.name;
      form.querySelector('#cfg-spec').value = model.csv_file;
      form.querySelector('.cfg-spec-row').append(button('Browse…', () => browse('spec')));
      form.querySelector('#cfg-add').append(button('+ Browse and Add CATs…', () => browse('cat')), button('+ Add Path Row', () => {model.cat_files.push(''); rows();}));
      form.querySelector('.cfg-actions').append(button('Cancel', close), button('Save and Open Configuration', save));
      form.addEventListener('cancel', e => {e.preventDefault(); close();});
      form.addEventListener('keydown', e => e.stopPropagation());
      document.body.append(form); rows(); form.showModal();
    } catch (exc) { alert(exc.message); }
  }
  async function save() {
    if (window.assignerEditorDirty) {error('Save or reload your Fitting Space drafts before changing configuration.'); return;}
    const buttons = form.querySelectorAll('button'); buttons.forEach(b => b.disabled=true);
    try {
      const result = await api('save', {...model, name:form.querySelector('#cfg-name').value, csv_file:form.querySelector('#cfg-spec').value});
      form.close(); form.remove(); form=null;
      window.location.assign(result.url);
    } catch (exc) {error(exc.message); buttons.forEach(b => b.disabled=false);}
  }
  async function browse(kind, replaceIndex=null) {
    browseKind = kind; selected = new Set();
    picker = el('dialog', null, 'cfg-dialog cfg-picker');
    picker.innerHTML = '<h2>Select files on this computer</h2><div class="cfg-browser-path"><select aria-label="Drive"></select><input aria-label="Directory path"></div><div class="cfg-files"></div><p class="cfg-picker-error" role="alert"></p><div class="cfg-actions"></div>';
    const pathInput = picker.querySelector('input');
    picker.querySelector('.cfg-browser-path').append(button('Go', () => load(pathInput.value)), button('Parent folder', () => load(picker.dataset.parent)));
    picker.querySelector('select').onchange = e => load(e.target.value);
    const dismiss = () => {picker.close(); picker.remove(); picker=null;};
    picker.querySelector('.cfg-actions').append(button('Cancel', dismiss), button(kind==='cat' ? 'Use Selected CATs' : 'Use Selected Spectrum', () => {
      const values = [...selected]; if (!values.length) {picker.querySelector('.cfg-picker-error').textContent='Select at least one file.'; return;}
      if (kind==='spec') form.querySelector('#cfg-spec').value=values[0];
      else if (replaceIndex!==null) {model.cat_files[replaceIndex]=values[0]; rows();}
      else {model.cat_files.push(...values.filter(x => !model.cat_files.includes(x))); rows();}
      dismiss();
    }));
    picker.addEventListener('cancel', e => {e.preventDefault(); dismiss();});
    picker.addEventListener('keydown', e => e.stopPropagation());
    document.body.append(picker); picker.showModal();
    const hint = kind==='spec' ? form.querySelector('#cfg-spec').value : (model.cat_files[replaceIndex??0] || form.querySelector('#cfg-spec').value);
    await load(hint ? hint.replace(/[\\/][^\\/]*$/, '') : browsePath);
    async function load(path) {
      try {
        const result = await api('browse?kind='+kind+'&path='+encodeURIComponent(path || ''));
        if (!picker) return;
        browsePath=result.path; pathInput.value=result.path; picker.dataset.parent=result.parent;
        const drives = picker.querySelector('select'); drives.replaceChildren();
        result.drives.forEach(d => {const o=el('option', d); o.value=d; o.selected=result.path.toLowerCase().startsWith(d.toLowerCase()); drives.append(o);});
        const list=picker.querySelector('.cfg-files'); list.replaceChildren();
        result.entries.forEach(entry => {
          if (entry.directory) {list.append(button('📁 '+entry.name, () => load(entry.path))); return;}
          const label=el('label'); const check=el('input'); check.type=kind==='spec'||replaceIndex!==null?'radio':'checkbox'; check.name='cfg-selected-file'; check.checked=selected.has(entry.path);
          check.onchange=()=>{if(check.type==='radio') selected.clear(); if(check.checked)selected.add(entry.path); else selected.delete(entry.path);};
          label.append(check, document.createTextNode(entry.name)); list.append(label);
        });
        if (!result.entries.length) list.append(el('p','No matching files in this folder.'));
        picker.querySelector('.cfg-picker-error').textContent=kind==='cat'?'Select multiple files, then click Use Selected CATs.':'';
      } catch(exc) {if(picker) picker.querySelector('.cfg-picker-error').textContent=exc.message;}
    }
  }
  document.addEventListener('click', e => {
    if(e.target.closest('#ws-new-config')) open();
    const edit=e.target.closest('#ws-edit-config, [data-edit-config]');
    if(edit) open(edit.dataset.editConfig || window.location.pathname.split('/')[2]);
  });
  window.addEventListener('beforeunload', e => {if (dirty()) {e.preventDefault(); e.returnValue='';}});
})();
