"""Local spectrum workspaces and Pickett tools for plotcomparison_2026_7.

Each JSON gets an isolated Dash application, preserving the original numerical
callbacks and their globals. The local WSGI router loads workspaces on demand.
"""
from pathlib import Path
import copy
import datetime
import hashlib
import html as html_escape
import json
import locale
import os
import re
import runpy
import shlex
import shutil
import subprocess
import threading
import time


EXTENSIONS = ('.par', '.var', '.int', '.lin', '.fit', '.out')
DIRECTORY_LOCKS = {}
DIRECTORY_LOCK_GUARD = threading.Lock()


def identity(path):
    return hashlib.sha256(os.path.normcase(str(Path(path).resolve())).encode()).hexdigest()[:16]


def discover(folder):
    result = []
    for path in sorted(Path(folder).glob('config*.json')):
        if 'backup' in path.name.lower():
            continue
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if not data.get('csv_file') or not (data.get('cat_files') or data.get('cat_file')):
                continue
            name = data.get('name') or (Path(data['csv_file']).stem if path.name == 'config.json' else path.stem[7:])
            result.append({'id': identity(path), 'path': str(path),
                           'label': name})
        except (ValueError, OSError):
            continue
    return result


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (ValueError, OSError):
        return default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temp, path)


def backup(path):
    path = Path(path)
    if not path.exists():
        return None
    dest = path.parent / '.assigner-history' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f') / path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    return dest


def resolve_executable(program, config, config_dir=None):
    """Resolve the existing JSON > environment > PATH precedence explicitly."""
    if program not in ('spfit', 'spcat'):
        raise ValueError('Only spfit and spcat are supported.')
    variable = program.upper() + '_PATH'
    configured = config.get(program + '_path')
    source = f'{program}_path in JSON' if configured else variable
    value = configured or os.environ.get(variable)
    if not value:
        value, source = shutil.which(program), 'PATH'
    if not value:
        raise ValueError(f'{program} not found. Set {variable} to the full executable path. See docs/INSTALL.md.')
    path = Path(os.path.expandvars(str(value))).expanduser()
    if configured and not path.is_absolute() and config_dir is not None:
        path = Path(config_dir) / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f'{program}: {source} points to a missing file: {path}. Update or remove this setting; lower-priority settings are not used.')
    if os.name != 'nt' and not os.access(path, os.X_OK):
        raise ValueError(f'{program} is not executable: {path}. Grant execute permission (chmod +x). See docs/INSTALL.md.')
    return str(path)


def file_snapshot(path):
    path = Path(path)
    if not path.is_file():
        return {'text': '', 'digest': None, 'encoding': 'utf-8', 'newline': os.linesep}
    raw = path.read_bytes()
    encoding = next((enc for enc in ('utf-8', locale.getpreferredencoding(False), 'cp1252')
                     if _decodes(raw, enc)), 'latin-1')
    return {'text': raw.decode(encoding).replace('\r\n', '\n'),
            'digest': hashlib.sha256(raw).hexdigest(), 'encoding': encoding,
            'newline': '\r\n' if b'\r\n' in raw else '\n'}


def _decodes(raw, encoding):
    try:
        raw.decode(encoding)
        return True
    except UnicodeDecodeError:
        return False


def save_document(path, text, original):
    if file_snapshot(path)['digest'] != original['digest']:
        raise ValueError('File changed on disk. Reload from Disk before saving; your draft is retained.')
    raw = text.replace('\r\n', '\n').replace('\n', original['newline']).encode(original['encoding'])
    backup(path)
    temp = Path(str(path) + '.assigner-tmp')
    temp.write_bytes(raw)
    os.replace(temp, path)
    return file_snapshot(path)


def directory_lock(directory):
    key = os.path.normcase(str(Path(directory).resolve()))
    with DIRECTORY_LOCK_GUARD:
        return DIRECTORY_LOCKS.setdefault(key, threading.Lock())


class Runner:
    def __init__(self):
        self.lock = threading.RLock()
        self.running = False
        self.process = None
        self.cancel = threading.Event()
        self.lines = []
        self.revision = 0
        self.directory = None
        self.log_path = None
        self.status = 'Ready.'

    def append(self, line):
        with self.lock:
            self.lines.append(line.rstrip('\r\n'))
            self.lines = self.lines[-5000:]
            if self.log_path:
                with self.log_path.open('a', encoding='utf-8') as log:
                    log.write(line.rstrip('\r\n') + '\n')

    def start(self, cat_path, commands, config, config_dir=None):
        directory = Path(cat_path).parent
        stem = Path(cat_path).stem
        jobs = []
        for program in commands:
            executable = resolve_executable(program, config, config_dir)
            jobs.append((program, executable, stem))
        with self.lock:
            if self.running:
                raise ValueError('A job is already running in this workspace.')
            guard = directory_lock(directory)
            if not guard.acquire(blocking=False):
                raise ValueError('This directory is busy in another workspace.')
            self.running = True
            self.directory = directory
            self.cancel.clear()
            self.lines = []
            self.log_path = None
            self.status = 'Preparing input backups…'
        threading.Thread(target=self._work, args=(directory, jobs, guard), daemon=True).start()

    def _work(self, directory, jobs, guard):
        try:
            folder = directory / '.assigner-history' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            folder.mkdir(parents=True, exist_ok=True)
            self.log_path = folder / 'run.log'
            stem = jobs[0][2]
            for ext in ('.par', '.var', '.int', '.lin', '.fit', '.cat', '.out', '.bak'):
                source = directory / (stem + ext)
                if source.is_file():
                    shutil.copy2(source, folder / source.name)
            self.append(f'Working directory: {directory}\nBackup / log: {folder}')
            for program, executable, stem in jobs:
                if self.cancel.is_set():
                    break
                required = ('.par', '.lin') if program == 'spfit' else ('.var', '.int')
                for ext in required:
                    if not (directory / (stem + ext)).is_file():
                        raise ValueError(f'Missing input: {stem}{ext}')
                if program == 'spfit':
                    count = sum(bool(line.strip()) for line in (directory / (stem + '.lin')).read_text(errors='replace').splitlines())
                    self.append(f'LIN nonblank lines: {count}')
                self.append(f'> {program} {stem}')
                self.status = f'Running {program} {stem}…'
                started = time.time_ns()
                with self.lock:
                    if self.cancel.is_set():
                        break
                    self.process = subprocess.Popen([executable, stem], cwd=str(directory),
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    process = self.process
                # Read chunks instead of readline: also show prompts without a newline.
                output_chunks = []
                while True:
                    chunk = os.read(process.stdout.fileno(), 4096)
                    if not chunk:
                        break
                    decoded = chunk.decode(locale.getpreferredencoding(False), errors='replace')
                    output_chunks.append(decoded)
                    self.append(decoded)
                code = process.wait()
                process.stdout.close()
                self.append(f'[{program}] Exit code: {code}')
                if self.cancel.is_set():
                    self.status = 'Stopped. Outputs may be incomplete.'
                    self.append('Stopped. Outputs may be incomplete; inputs are backed up above.')
                    break
                if code != 0:
                    self.status = f'Failed: {program} exited with code {code}. See console output.'
                    break
                ext = '.fit' if program == 'spfit' else '.cat'
                output = directory / (stem + ext)
                if not output.exists() or output.stat().st_mtime_ns < started - 2_000_000_000:
                    raise ValueError(f'{program} did not produce a fresh {ext} file.')
                if program == 'spfit':
                    report = output.read_text(errors='replace')
                    highlights = [line for line in report.splitlines()
                                  if re.search(r'LINES REQUESTED|Bad Line|reject|RMS|FIT COMPLETE', line, re.I)]
                    self.append('FIT summary:\n' + '\n'.join(highlights[-35:]))
                    combined = report + '\n' + ''.join(output_chunks)
                    bad_count = len(re.findall(r'Bad Line', report, re.I))
                    complete = bool(re.search(r'FIT COMPLETE', combined, re.I))
                    self.append(f'Bad Line occurrences in FIT: {bad_count}; FIT COMPLETE (FIT/console): {complete}')
                    if re.search(r'Bad Line', combined, re.I) or not complete:
                        self.status = 'SPFIT needs review; chained prediction was not run. See FIT / console.'
                        self.append('Inspect FIT before prediction: chained SPCAT has been stopped.')
                        break
                    self.status = 'SPFIT finished. Inspect FIT; run SPCAT to update predictions.'
                else:
                    self.status = 'SPCAT finished. Use Refresh CAT to apply the new predictions.'
                    self.append('Prediction written. Use Refresh CAT to apply it to the Assigner.')
        except Exception as exc:
            self.status = f'Failed: {exc}'
            if isinstance(exc, PermissionError):
                self.status += ' The server needs write permission in the catalog directory. No backup protection was bypassed.'
            # A failed log write must not hide the original error or leak the lock.
            try:
                self.append(f'ERROR: {exc}')
            except OSError:
                self.log_path = None
                self.append(f'ERROR: {exc}')
        finally:
            with self.lock:
                self.process = None
                self.running = False
                self.revision += 1
            guard.release()

    def stop(self):
        with self.lock:
            self.cancel.set()
            if self.process and self.process.poll() is None:
                self.process.terminate()


def remap_assignments(rows, catalog):
    """Match by all QNs, never by row number or nearest frequency."""
    if not rows:
        return [], 0
    fields = catalog['qn_order']
    index = {}
    for values in catalog['df'][fields + ['Freq', 'Intensity', 'Eu', 'SimUID']].itertuples(index=False, name=None):
        key = tuple(int(value) for value in values[:len(fields)])
        index.setdefault(key, []).append(dict(zip(['Freq', 'Intensity', 'Eu', 'SimUID'], values[len(fields):])))
    result, unmatched = [], 0
    for original in rows:
        row = copy.deepcopy(original)
        try:
            matches = index.get(tuple(int(row[field]) for field in fields), [])
        except (KeyError, ValueError, TypeError):
            matches = []
        if len(matches) == 1:
            hit = matches[0]
            row.update(sim=float(hit['Freq']), logI=float(__import__('numpy').log10(hit['Intensity'])),
                       Eu=float(hit['Eu']), SimUID=int(hit['SimUID']), CatalogStatus='matched')
        else:
            # Negative IDs cannot accidentally highlight a new CAT row.
            row.update(SimUID=-(len(result) + 1), sim=None, WeightedSim=None, Delta=None,
                       CatalogStatus='missing' if not matches else 'ambiguous')
            unmatched += 1
        result.append(row)
    return result, unmatched


def launch(script, port=8053):
    from werkzeug.serving import run_simple
    from werkzeug.wrappers import Response
    script = Path(script).resolve()
    registry, lock = {}, threading.RLock()
    from spectrum_config_v7 import config_api
    manager = config_api(script.parent, busy=lambda: any(entry['namespace']['workspace_runner'].running for entry in registry.values()))

    def application(environ, start_response):
        options = discover(script.parent)
        by_id = {item['id']: item for item in options}
        path = environ.get('PATH_INFO', '/')
        if path.startswith('/config-api/'):
            with lock:
                return manager(environ, start_response)
        match = re.match(r'^/spectrum/([a-f0-9]{16})/', path)
        if not match or match[1] not in by_id:
            links = ''.join(f'<li><a href="/spectrum/{item["id"]}/">{html_escape.escape(item["label"])}</a> <button data-edit-config="{item["id"]}">Edit</button></li>' for item in options)
            if path in ('/config-manager.js', '/config-manager.css', '/theme-v7.css'):
                asset = script.parent / 'assets' / ('zz_theme_v7.css' if path == '/theme-v7.css' else 'config_v7.js' if path.endswith('.js') else 'config_v7.css')
                return Response(asset.read_text(encoding='utf-8'), content_type='application/javascript' if path.endswith('.js') else 'text/css')(environ, start_response)
            response = Response('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Spectrum configurations · Assigner v7</title><link rel="stylesheet" href="/config-manager.css"><link rel="stylesheet" href="/theme-v7.css"></head><body class="v7-home"><main><div class="brand">Spectroscopy workspace · v7</div><h1>Hyperfine Interactive Spectrum Assigner</h1><p class="intro">Select a spectrum configuration to continue assigning and fitting transitions.</p><div class="home-toolbar"><h2>Spectrum configurations</h2><button id="ws-new-config">+ New Configuration</button></div><ul class="config-list">' + links + '</ul><p class="footnote">Each configuration connects one experimental spectrum with its catalog files.</p></main><script src="/config-manager.js"></script></body></html>', content_type='text/html; charset=utf-8')
            return response(environ, start_response)
        item = by_id[match[1]]
        try:
            with lock:
                revision = hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()
                if item['id'] not in registry or registry[item['id']]['revision'] != revision:
                    namespace = runpy.run_path(str(script), init_globals={
                        'WORKSPACE_CONFIG': item['path'], 'WORKSPACE_OPTIONS': options,
                        'WORKSPACE_PREFIX': f'/spectrum/{item["id"]}/'}, run_name='_spectrum_workspace_v7')
                    registry[item['id']] = {'namespace': namespace, 'revision': revision}
                server = registry[item['id']]['namespace']['app'].server
            return server(environ, start_response)
        except Exception as exc:
            response = Response('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="/theme-v7.css"><title>Configuration could not be loaded</title></head><body class="v7-home"><main><div class="brand">Spectroscopy workspace · v7</div><h1>Could not load spectrum configuration</h1><pre class="error-detail">' + html_escape.escape(str(exc)) + '</pre><a href="/">← Back to configurations</a></main></body></html>', status=500, content_type='text/html; charset=utf-8')
            return response(environ, start_response)

    print(f'Assigner v7: http://127.0.0.1:{port}/')
    run_simple('127.0.0.1', port, application, threaded=True, use_reloader=False)


def install(g):
    """Add the two spaces while keeping all original graph/store components mounted."""
    from dash import html, dcc, Input, Output, State, ctx, no_update
    from dash.exceptions import PreventUpdate
    app, catalogs = g['app'], g['catalogs']
    runner = Runner()
    g['workspace_runner'] = runner
    storage = Path(g['WORKSPACE_STATE_DIR'])
    storage.mkdir(parents=True, exist_ok=True)
    view = read_json(storage / 'view.json', {})
    initial_active = min(int(view.get('active', 0)), len(catalogs) - 1)
    original = app.layout.children
    for component in original:
        if getattr(component, 'id', '') == 'active-cat-idx':
            component.data = initial_active
        if getattr(component, 'id', '') == 'stored-zoom' and view.get('zoom'):
            component.data = view['zoom']
    button = lambda text, id: html.Button(text, id=id, n_clicks=0)
    app.layout = html.Div([
        html.Div([
            html.Div([html.Label('Spectrum configuration'), dcc.Dropdown(id='ws-profile',
                options=[{'label': x['label'], 'value': f'/spectrum/{x["id"]}/'} for x in g['WORKSPACE_OPTIONS']],
                value=g['WORKSPACE_PREFIX'], clearable=False)], style={'flex': '2'}),
            html.Div([html.Label('Active catalog'), dcc.Dropdown(id='ws-catalog',
                options=[{'label': c['name'], 'value': i} for i, c in enumerate(catalogs)],
                value=initial_active, clearable=False)], style={'flex': '1'}),
            button('FITTING SPACE →', 'ws-enter'), button('Refresh CAT', 'ws-refresh'),
            button('Refresh All CATs', 'ws-refresh-all'),
            button('New Configuration', 'ws-new-config'), button('Edit Configuration', 'ws-edit-config'),
        ], className='ws-toolbar'),
        html.Div(id='ws-context'), html.Div(id='ws-refresh-status', role='status'),
        html.Div(original, id='ws-assigner'),
        html.Div([
            button('← Back to Assigner Space', 'ws-back'),
            html.H2('Fitting Space'),
            html.Div([button('Write Assignments to Working LIN', 'ws-write-lin'),
                      button('Import Working LIN into Assigner', 'ws-import-lin')], className='ws-toolbar'),
            html.Div('Writing / importing LIN replaces the target after confirmation. Saves and runs keep timestamped backups.'),
            dcc.Tabs(id='ws-extension', value='.par', children=[dcc.Tab(label=ext[1:].upper(), value=ext) for ext in EXTENSIONS]),
            html.Div(id='ws-editor-path'),
            html.Div([html.Pre('1', id='ws-line-numbers', **{'aria-hidden': 'true'}),
                      dcc.Textarea(id='ws-editor', value='', spellCheck=False, className='ws-editor')], className='ws-editor-wrap'),
            html.Div([button('Reload from Disk', 'ws-reload'), button('Save', 'ws-save'),
                      button('Save All Drafts', 'ws-save-all'), html.Span(id='ws-dirty')], className='ws-toolbar'),
            html.Div(id='ws-file-status', role='status'),
            html.Div([button('Run SPFIT', 'ws-spfit'), button('Run SPCAT', 'ws-spcat'),
                      button('SPFIT → SPCAT', 'ws-both'), button('Stop', 'ws-stop')], className='ws-toolbar'),
            html.Div([dcc.Input(id='ws-command', placeholder='spfit or spcat [active file basename]', type='text', style={'width': '65%'}),
                      button('Run Command', 'ws-run-command')], className='ws-toolbar'),
            html.Div(id='ws-run-status', role='status'),
            html.Pre(id='ws-console', children='Ready.', className='ws-console'),
        ], id='ws-fitting', style={'display': 'none'}),
        dcc.Store(id='ws-page', data='assigner'), dcc.Store(id='ws-documents', data={}),
        dcc.Store(id='ws-open-document'), dcc.Store(id='ws-revision', data=0),
        dcc.Store(id='ws-job-revision', data=0), dcc.Store(id='ws-busy', data=False),
        dcc.Store(id='ws-view-saved'), dcc.Store(id='ws-navigation'),
        dcc.ConfirmDialog(id='ws-confirm-write', message='Replace the active working LIN with the assignment table? A timestamped backup will be kept.'),
        dcc.ConfirmDialog(id='ws-confirm-import', message='Replace the active assignment table with the working LIN? The previous assignments will be backed up.'),
        dcc.Interval(id='ws-poll', interval=600),
    ], className='ws-root')

    base_layout = app.layout

    def fresh_layout():
        # Re-entering a spectrum restores current disk state, not the startup snapshot.
        layout = copy.deepcopy(base_layout)
        state = read_json(storage / 'view.json', {})
        assignments = g['_load_assignment_autosave']()
        active = max(0, min(int(state.get('active', 0)), len(catalogs) - 1))
        def visit(component):
            name = getattr(component, 'id', None)
            if name == 'percat-assignments':
                component.data = assignments
            elif name == 'ws-profile':
                component.options = [{'label': x['label'], 'value': f'/spectrum/{x["id"]}/'} for x in discover(g['script_dir'])]
            elif name == 'active-cat-idx':
                component.data = active
            elif name == 'ws-catalog':
                component.value = active
            elif name == 'active-cat-label':
                component.children = catalogs[active]['name']
            elif name == 'assignment-table':
                component.data = g['_sanitize_for_table'](assignments.get(str(active), []))
                component.columns = g['build_assignment_columns'](catalogs[active]['qn_order'])
            elif name == 'stored-zoom' and state.get('zoom'):
                component.data = state['zoom']
            elif name == 'percat-scales':
                component.data = g['_load_scale_cache']()
            elif name == 'sim-scale':
                component.value = g['_load_scale_cache']().get(str(active), 1.0)
            children = getattr(component, 'children', None)
            for child in children if isinstance(children, list) else [children]:
                if hasattr(child, 'to_plotly_json'):
                    visit(child)
        visit(layout)
        return layout
    app.layout = fresh_layout

    # Serialize callbacks within a spectrum; runner subprocesses remain asynchronous.
    request_lock = threading.RLock()
    from flask import g as request_g
    @app.server.before_request
    def lock_callback():
        request_lock.acquire()
        request_g.workspace_locked = True

    @app.server.teardown_request
    def unlock_callback(_error):
        if getattr(request_g, 'workspace_locked', False):
            request_lock.release()

    @app.callback(Output('ws-assigner', 'style'), Output('ws-fitting', 'style'), Output('ws-page', 'data'),
                  Input('ws-enter', 'n_clicks'), Input('ws-back', 'n_clicks'), prevent_initial_call=True)
    def page(_enter, _back):
        fitting = ctx.triggered_id == 'ws-enter'
        return ({'display': 'none'} if fitting else {}, {} if fitting else {'display': 'none'}, 'fitting' if fitting else 'assigner')

    # A single circular callback keeps keyboard switching and the dropdown in sync.
    @app.callback(Output('active-cat-idx', 'data', allow_duplicate=True), Output('ws-catalog', 'value'),
                  Input('ws-catalog', 'value'), Input('active-cat-idx', 'data'), prevent_initial_call=True)
    def active(selected, current):
        if ctx.triggered_id == 'ws-catalog' and selected != current:
            return int(selected), no_update
        return no_update, int(current or 0)

    @app.callback(Output('ws-context', 'children'), Input('active-cat-idx', 'data'))
    def context(active_idx):
        path = Path(catalogs[int(active_idx or 0)]['path'])
        return f'Spectrum: {g["csv_file_path"]} | Working directory: {path.parent} | File basename: {path.stem}'

    @app.callback(Output('ws-view-saved', 'data'), Input('stored-zoom', 'data'), Input('active-cat-idx', 'data'))
    def save_view(zoom, active_idx):
        atomic_json(storage / 'view.json', {'zoom': zoom, 'active': active_idx})
        return time.time()

    app.clientside_callback('''function(profile) {
        if (profile && profile !== window.location.pathname) {
            window.location.assign(profile);
        }
        return window.dash_clientside.no_update;
    }''', Output('ws-navigation', 'data'), Input('ws-profile', 'value'), prevent_initial_call=True)

    def current_path(active_idx, ext):
        if ext not in EXTENSIONS:
            raise ValueError('Unsupported file extension')
        return Path(catalogs[int(active_idx or 0)]['path']).with_suffix(ext)

    @app.callback(Output('ws-documents', 'data'), Output('ws-open-document', 'data'),
        Output('ws-editor', 'value'), Output('ws-editor-path', 'children'), Output('ws-file-status', 'children'),
        Input('active-cat-idx', 'data'), Input('ws-extension', 'value'), Input('ws-save', 'n_clicks'),
        Input('ws-save-all', 'n_clicks'), Input('ws-reload', 'n_clicks'), Input('ws-job-revision', 'data'),
        Input('ws-editor', 'value'),
        State('ws-documents', 'data'), State('ws-open-document', 'data'))
    def editor(active_idx, ext, _save, _all, _reload, revision, text, documents, opened):
        documents = copy.deepcopy(documents or {})
        if opened in documents:
            documents[opened]['draft'] = text or ''
        path = current_path(active_idx, ext)
        key = str(path)
        if key not in documents:
            documents[key] = {**file_snapshot(path), 'draft': file_snapshot(path)['text']}
        status = ''
        trigger = ctx.triggered_id
        try:
            if trigger in ('ws-save', 'ws-save-all'):
                keys = list(documents) if trigger == 'ws-save-all' else [key]
                for name in keys:
                    doc = documents[name]
                    if doc['draft'] == doc['text']:
                        continue
                    guard = directory_lock(Path(name).parent)
                    if not guard.acquire(blocking=False):
                        raise ValueError('Directory is busy. Wait for the running job before saving.')
                    try:
                        snapshot = save_document(name, doc['draft'], doc)
                        documents[name] = {**snapshot, 'draft': snapshot['text']}
                    finally:
                        guard.release()
                status = 'Saved. Previous files are in .assigner-history.'
            elif trigger == 'ws-reload':
                # The browser capture handler confirms before discarding a draft.
                snapshot = file_snapshot(path)
                documents[key] = {**snapshot, 'draft': snapshot['text']}
                status = 'Reloaded from disk.'
            elif trigger == 'ws-job-revision':
                for name, doc in documents.items():
                    if doc['draft'] == doc['text']:
                        snapshot = file_snapshot(name)
                        documents[name] = {**snapshot, 'draft': snapshot['text']}
                status = 'Job ended; editor files checked against disk. Unsaved drafts retained.' if revision else ''
        except Exception as exc:
            status = str(exc)
        doc = documents[key]
        exists = '' if doc['digest'] is not None else ' (missing — Save creates this file)'
        # Avoid resetting the caret while typing.
        value = no_update if trigger == 'ws-editor' else doc['draft']
        return documents, key, value, key + exists, status

    app.clientside_callback('''function(text, docs, opened) {
        let dirty = Object.entries(docs || {}).some(([key,d]) => (key === opened ? (text || '') : d.draft) !== d.text);
        window.assignerEditorDirty = dirty;
        return dirty ? '● Unsaved draft(s)' : 'All changes saved';
    }''', Output('ws-dirty', 'children'), Input('ws-editor', 'value'), Input('ws-documents', 'data'), State('ws-open-document', 'data'))

    @app.callback(Output('ws-run-status', 'children'),
        Input('ws-spfit', 'n_clicks'), Input('ws-spcat', 'n_clicks'), Input('ws-both', 'n_clicks'),
        Input('ws-run-command', 'n_clicks'), Input('ws-stop', 'n_clicks'),
        State('active-cat-idx', 'data'), State('ws-command', 'value'), State('ws-documents', 'data'),
        State('ws-open-document', 'data'), State('ws-editor', 'value'), prevent_initial_call=True)
    def run(_fit, _cat, _both, _command, _stop, active_idx, command, documents, opened, text):
        if ctx.triggered_id == 'ws-stop':
            runner.stop()
            return 'Stop requested.'
        path = Path(catalogs[int(active_idx or 0)]['path'])
        try:
            for name, doc in (documents or {}).items():
                draft = text if name == opened else doc['draft']
                if Path(name).parent == path.parent and draft != doc['text']:
                    raise ValueError('Save or reload unsaved drafts in this working directory before running.')
            commands = {'ws-spfit': ['spfit'], 'ws-spcat': ['spcat'], 'ws-both': ['spfit', 'spcat']}.get(ctx.triggered_id)
            if commands is None:
                words = shlex.split(command or '', posix=False)
                words = [word.strip('"') for word in words]
                if not words or words[0].lower() not in ('spfit', 'spcat') or len(words) > 2:
                    raise ValueError('Use spfit or spcat, optionally followed by the active file basename.')
                if len(words) == 2 and words[1] != path.stem:
                    raise ValueError(f'This workspace runs the active basename: {path.stem}')
                commands = [words[0].lower()]
            runner.start(path, commands, g['config'], Path(g['WORKSPACE_CONFIG']).parent)
            return f'Running {" → ".join(commands)} {path.stem}; inputs backed up automatically.'
        except Exception as exc:
            return f'Cannot run: {exc}'

    @app.callback(Output('ws-console', 'children'), Output('ws-job-revision', 'data'), Output('ws-busy', 'data'),
                  Input('ws-poll', 'n_intervals'), State('ws-console', 'children'),
                  State('ws-job-revision', 'data'), State('ws-busy', 'data'))
    def poll(_, previous_text, previous_revision, previous_busy):
        with runner.lock:
            text = '\n'.join(runner.lines) or 'Ready. Commands run in the active catalog directory.'
            text = ('RUNNING\n' if runner.running else 'IDLE\n') + text
            return (text if text != previous_text else no_update,
                    runner.revision if runner.revision != previous_revision else no_update,
                    runner.running if runner.running != previous_busy else no_update)

    @app.callback([Output(name, 'disabled') for name in ('ws-profile', 'ws-catalog', 'ws-spfit', 'ws-spcat', 'ws-both', 'ws-run-command')],
                  Input('ws-busy', 'data'))
    def running_controls(busy):
        return [bool(busy)] * 6

    @app.callback(Output('ws-run-status', 'children', allow_duplicate=True),
                  Input('ws-job-revision', 'data'), prevent_initial_call=True)
    def finished_status(_revision):
        with runner.lock:
            return runner.status

    @app.callback(Output('ws-refresh-status', 'children'), Output('percat-assignments', 'data', allow_duplicate=True),
        Output('ws-revision', 'data'), Input('ws-refresh', 'n_clicks'), Input('ws-refresh-all', 'n_clicks'),
        State('active-cat-idx', 'data'), State('percat-assignments', 'data'), State('ws-revision', 'data'), prevent_initial_call=True)
    def refresh(_one, _all, active_idx, assignments, revision):
        indices = range(len(catalogs)) if ctx.triggered_id == 'ws-refresh-all' else [int(active_idx or 0)]
        updated, replacements, notes = copy.deepcopy(assignments or {}), {}, []
        guards = []
        try:
            for directory in sorted({str(Path(catalogs[i]['path']).parent) for i in indices}):
                guard = directory_lock(directory)
                if not guard.acquire(blocking=False):
                    raise ValueError('A selected directory is busy; wait for SPCAT/SPFIT to finish.')
                guards.append(guard)
            for i in indices:
                cat = g['load_catalog'](catalogs[i]['path'])
                rows, missing = remap_assignments(updated.get(str(i), []), cat)
                updated[str(i)] = g['recompute_peak_weights'](rows, recompute_weights=False)
                replacements[i] = cat
                notes.append(f'{cat["name"]}: {len(cat["df"])} lines, {missing} unmatched/ambiguous assignments retained')
            for i, cat in replacements.items():
                catalogs[i] = cat
            g['_save_assignment_autosave'](updated)
            return ' | '.join(notes), updated, int(revision or 0) + 1
        except Exception as exc:
            return f'Refresh failed; previous catalogs retained: {exc}', no_update, no_update
        finally:
            for guard in guards:
                guard.release()

    @app.callback(Output('ws-confirm-write', 'displayed'), Input('ws-write-lin', 'n_clicks'), prevent_initial_call=True)
    def ask_write(_):
        return True

    @app.callback(Output('ws-confirm-import', 'displayed'), Input('ws-import-lin', 'n_clicks'), prevent_initial_call=True)
    def ask_import(_):
        return True

    @app.callback(Output('ws-refresh-status', 'children', allow_duplicate=True),
        Output('percat-assignments', 'data', allow_duplicate=True),
        Input('ws-confirm-write', 'submit_n_clicks'), Input('ws-confirm-import', 'submit_n_clicks'),
        State('active-cat-idx', 'data'), State('percat-assignments', 'data'),
        State('ws-documents', 'data'), State('ws-open-document', 'data'), State('ws-editor', 'value'), prevent_initial_call=True)
    def lin(_write, _import, active_idx, assignments, docs, opened, text):
        i = int(active_idx or 0)
        path = current_path(i, '.lin')
        guard = directory_lock(path.parent)
        if not guard.acquire(blocking=False):
            return 'Working directory is busy.', no_update
        try:
            doc = (docs or {}).get(str(path))
            if doc and (text if opened == str(path) else doc['draft']) != doc['text']:
                raise ValueError('Save or reload the LIN draft first.')
            if ctx.triggered_id == 'ws-confirm-write':
                rows = (assignments or {}).get(str(i), [])
                if not rows:
                    raise ValueError('No assignments to write.')
                if any(r.get('CatalogStatus') in ('missing', 'ambiguous') for r in rows):
                    raise ValueError('Resolve unmatched/ambiguous assignments before writing LIN.')
                content = g['generate_lin_file'](rows, catalogs[i]['qn_order']) + '\n'
                save_document(path, content, file_snapshot(path))
                return f'Wrote {len(rows)} assignments to {path}. Reload the LIN editor to view it.', no_update
            loaded = []
            for number, raw in enumerate(path.read_text().splitlines(), 1):
                if not raw.strip():
                    continue
                qns, freq, unc, weight = g['parse_lin_line_flexible'](raw)
                fields = catalogs[i]['qn_order']
                if len(qns) != len(fields):
                    raise ValueError(f'LIN line {number}: expected {len(fields)} quantum numbers, got {len(qns)}.')
                loaded.append(dict(zip(fields, qns), obs=freq, Uncertainty=unc, Weight=weight))
            if not loaded:
                raise ValueError('LIN contains no assignments; current table retained.')
            loaded, missing = remap_assignments(loaded, catalogs[i])
            updated = copy.deepcopy(assignments or {})
            atomic_json(storage / ('before-lin-import-' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json'), updated)
            updated[str(i)] = g['recompute_peak_weights'](loaded, recompute_weights=False)
            g['_save_assignment_autosave'](updated)
            return f'Imported {len(loaded)} lines; {missing} unmatched/ambiguous. File uncertainties and weights retained.', updated
        except Exception as exc:
            return f'LIN operation failed: {exc}', no_update
        finally:
            guard.release()
