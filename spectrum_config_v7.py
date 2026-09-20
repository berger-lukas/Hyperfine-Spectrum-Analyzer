"""Local configuration manager and a browser-based local file picker."""
from pathlib import Path
import copy
import hashlib
import json
import os
import uuid

from flask import Flask, jsonify, request
from spectrum_workspace_v7 import discover, identity, atomic_json, backup, read_json


def config_api(folder, busy=lambda: False):
    folder = Path(folder).resolve()
    app = Flask(__name__)

    def locate(key):
        for item in discover(folder):
            if item['id'] == key:
                return Path(item['path'])
        raise ValueError('Configuration no longer exists. Refresh the list.')

    @app.errorhandler(ValueError)
    @app.errorhandler(OSError)
    def error(exc):
        return jsonify(error=str(exc)), 400

    @app.get('/config-api/list')
    def listing():
        return jsonify(discover(folder))

    @app.get('/config-api/config/<key>')
    def get_config(key):
        path = locate(key)
        raw = path.read_bytes()
        data = json.loads(raw)
        def absolute(value):
            value = Path(value).expanduser()
            return str(value if value.is_absolute() else (path.parent / value).resolve())
        return jsonify(id=key, name=next(x['label'] for x in discover(folder) if x['id'] == key),
                       csv_file=absolute(data['csv_file']),
                       cat_files=[absolute(x) for x in data.get('cat_files', [data.get('cat_file')])],
                       revision=hashlib.sha256(raw).hexdigest())

    @app.get('/config-api/browse')
    def browse():
        path = Path(request.args.get('path') or folder).expanduser().resolve()
        if not path.is_dir():
            raise ValueError('Choose an existing directory.')
        kind = request.args.get('kind', 'spec')
        extensions = {'.cat'} if kind == 'cat' else {'.csv', '.txt', '.dat', '.tsv'}
        entries = []
        with os.scandir(path) as children:
            for item in children:
                try:
                    is_dir = item.is_dir()
                    if is_dir or Path(item.name).suffix.lower() in extensions:
                        entries.append({'name': item.name, 'path': item.path, 'directory': is_dir})
                except OSError:
                    continue
        entries.sort(key=lambda x: (not x['directory'], x['name'].casefold()))
        drives = [f'{letter}:\\' for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if Path(f'{letter}:\\').is_dir()] if os.name == 'nt' else ['/']
        return jsonify(path=str(path), parent=str(path.parent), drives=drives, entries=entries)

    @app.post('/config-api/save')
    def save():
        if busy():
            raise ValueError('Wait for the running SPFIT/SPCAT job before changing configurations.')
        payload = request.get_json(force=True)
        name = str(payload.get('name', '')).strip()
        if not name:
            raise ValueError('Enter a configuration name.')
        key = payload.get('id')
        path = locate(key) if key else folder / f'config-{uuid.uuid4().hex[:12]}.json'
        previous = {}
        if key:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != payload.get('revision'):
                raise ValueError('Configuration changed on disk. Reopen Edit Configuration before saving.')
            previous = json.loads(raw)
        for item in discover(folder):
            if item['id'] != key and item['label'].casefold() == name.casefold():
                raise ValueError('This configuration name is already used. Choose another name.')
        def existing(value, extension=None):
            if not value:
                raise ValueError('Select the spectrum file and at least one CAT file.')
            p = Path(value).expanduser()
            p = (path.parent / p).resolve() if not p.is_absolute() else p.resolve()
            if not p.is_file():
                raise ValueError(f'File does not exist: {p}')
            if extension and p.suffix.lower() != extension:
                raise ValueError(f'Expected a {extension} file: {p}')
            return str(p)
        csv = existing(payload.get('csv_file'))
        cats = payload.get('cat_files', [])
        if not isinstance(cats, list):
            raise ValueError('CAT files must be a list.')
        cats = [existing(x, '.cat') for x in cats]
        if not cats:
            raise ValueError('Add at least one CAT file.')
        if len({os.path.normcase(x) for x in cats}) != len(cats):
            raise ValueError('The same CAT file is listed more than once.')
        # Validate spectrum structure before replacing a working configuration.
        import pandas as pd
        previous_csv = Path(previous.get('csv_file', '')).expanduser()
        if not previous_csv.is_absolute():
            previous_csv = path.parent / previous_csv
        same_spectrum = bool(previous) and previous_csv.resolve() == Path(csv)
        separator = previous.get('csv_separator', ';') if same_spectrum else 'auto'
        frame = pd.read_csv(csv, sep=None if separator == 'auto' else separator,
                            engine='python', nrows=20)
        if frame.shape[1] < 2 or frame.shape[0] == 0:
            raise ValueError('Spectrum must contain at least two columns: frequency and intensity.')
        try:
            frame.iloc[:, :2].astype(float)
        except (ValueError, TypeError):
            raise ValueError('The first two spectrum columns must be numeric.')
        updated = copy.deepcopy(previous)
        updated.update(name=name, csv_file=csv, cat_files=cats)
        if not same_spectrum:
            updated['csv_separator'] = 'auto'
        updated.pop('cat_file', None)
        # Existing scientific settings are never replaced by the form.
        updated.setdefault('uncertainty', {'fid_time_us': 10.0, 'base_sigma_instr_mhz': 0.01})
        backup(path)
        atomic_json(path, updated)
        # Index-based display settings must follow paths when the list is reordered.
        if previous:
            state = folder / 'autosave' / 'v7' / identity(path)
            old_cats = previous.get('cat_files', [previous.get('cat_file')])
            def canonical(value):
                p = Path(value).expanduser()
                return os.path.normcase(str((p if p.is_absolute() else folder / p).resolve()))
            mapping = {str(i): str(j) for i, old in enumerate(old_cats) for j, new in enumerate(cats)
                       if canonical(old) == canonical(new)}
            scales = read_json(state / 'scales.json', {})
            if scales:
                atomic_json(state / 'scales.json', {mapping[k]:v for k,v in scales.items() if k in mapping})
            view = read_json(state / 'view.json', {})
            if view:
                view['active'] = int(mapping.get(str(view.get('active',0)), '0'))
                if canonical(previous['csv_file']) != canonical(csv):
                    view.pop('zoom', None)
                atomic_json(state / 'view.json', view)
        return jsonify(id=identity(path), url=f'/spectrum/{identity(path)}/', name=name)

    return app
