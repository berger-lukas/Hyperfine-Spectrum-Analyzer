"""Regression checks; real Pickett runs use copies inside .v7-test-output only."""
from pathlib import Path
import copy
import hashlib
import json
import os
import runpy
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

from dash._callback_context import context_value
from dash._utils import AttributeDict
from spectrum_workspace_v7 import file_snapshot, save_document, remap_assignments, Runner

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / '.v7-test-output'
OUTPUT.mkdir(exist_ok=True)
WORK = Path(tempfile.mkdtemp(prefix='regression-', dir=OUTPUT)).resolve()
assert WORK.is_relative_to(OUTPUT.resolve())


class WorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Synthetic fixed-width records: no personal configuration or molecule needed.
        records = []
        for index in range(15):
            header = f'{5100 + index * 10:13.4f}{0.001:8.4f}{-4.0:8.4f}{3:2d}{1.0:10.4f}{3:3d}{12345:7d}{303:4d}'
            upper = f'{index+1:2d}{0:2d}{index+1:2d}' + ' ' * 6
            lower = f'{index:2d}{0:2d}{index:2d}' + ' ' * 6
            records.append(header + upper + lower + '\n')
        for name in ('first', 'second'):
            directory = WORK / name
            directory.mkdir()
            (directory / 'sample.cat').write_text(''.join(records))
        (WORK / 'spectrum.csv').write_text('freq;intensity\n' + '\n'.join(f'{5000+i};{1+(i%7)*0.01}' for i in range(1000)))
        cls.config = WORK / 'test.json'
        cls.config.write_text(json.dumps({'cat_files': [str(WORK / name / 'sample.cat') for name in ('first','second')],
            'csv_file': str(WORK / 'spectrum.csv'), 'uncertainty': {'fid_time_us': 25}}))
        cls.g = runpy.run_path(str(ROOT / 'plotcomparison_2026_7.py'), init_globals={'WORKSPACE_CONFIG': str(cls.config)})
        cls.app = cls.g['app']
        cls.client = cls.app.server.test_client()

    def callback(self, function_name, trigger, *args):
        fn = next(value['callback'].__wrapped__ for value in self.app.callback_map.values()
                  if 'callback' in value and getattr(value['callback'], '__name__', '') == function_name)
        token = context_value.set(AttributeDict(triggered_inputs=[{'prop_id': trigger, 'value': 1}]))
        try:
            return fn(*args)
        finally:
            context_value.reset(token)

    def test_01_layout_and_dependencies(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.get('/_dash-layout').status_code, 200)
        self.assertEqual(self.client.get('/_dash-dependencies').status_code, 200)
        self.assertEqual(self.g['DELTA_F_DEFAULT'], 0.04)

    def test_02_editor_roundtrip_conflict(self):
        path = WORK / 'first' / 'sample.par'
        path.write_bytes(b'title\r\n  123  0.010 / comments\r\n')
        original = file_snapshot(path)
        changed = save_document(path, original['text'].replace('123', '456'), original)
        self.assertIn(b'\r\n', path.read_bytes())
        self.assertTrue(list(path.parent.glob('.assigner-history/*/sample.par')))
        path.write_bytes(b'external change')
        with self.assertRaisesRegex(ValueError, 'changed on disk'):
            save_document(path, 'stale', changed)
        self.assertEqual(path.read_bytes(), b'external change')

    def test_03_editor_drafts_follow_catalog(self):
        first = self.callback('editor', 'active-cat-idx.data', 0, '.int', 0, 0, 0, 0, '', {}, None)
        second = self.callback('editor', 'active-cat-idx.data', 1, '.int', 0, 0, 0, 0, 'first draft', first[0], first[1])
        restored = self.callback('editor', 'active-cat-idx.data', 0, '.int', 0, 0, 0, 0, 'second draft', second[0], second[1])
        self.assertEqual(restored[2], 'first draft')
        self.assertEqual(restored[0][second[1]]['draft'], 'second draft')

    def test_04_refresh_remaps_and_rolls_back(self):
        g = self.g
        old = g['catalogs'][0]
        hit = old['df'].iloc[0]
        row = {field: int(hit[field]) for field in old['qn_order']}
        row.update(obs=float(hit['Freq']) + 0.1, sim=float(hit['Freq']), SimUID=0, Weight=1.0,
                   Uncertainty=0.0123, FitCtx={'example': True})
        path = Path(old['path'])
        records = path.read_text().splitlines()
        records[0] = f'{float(hit["Freq"])+0.05:13.4f}' + records[0][13:]
        path.write_text('\n'.join(reversed(records))+'\n')
        result = self.callback('refresh', 'ws-refresh.n_clicks', 1, 0, 0, {'0': [row]}, 0)
        saved = result[1]['0'][0]
        self.assertNotEqual(saved['SimUID'], 0)
        self.assertAlmostEqual(saved['Delta'], 0.05, places=3)
        self.assertEqual(saved['Uncertainty'], row['Uncertainty'])
        self.assertEqual(saved['FitCtx'], row['FitCtx'])
        path.write_text('invalid catalog\n')
        previous = g['catalogs'][0]
        result = self.callback('refresh', 'ws-refresh.n_clicks', 2, 0, 0, result[1], result[2])
        self.assertIn('failed', result[0])
        self.assertIs(g['catalogs'][0], previous)
        path.write_text('\n'.join(records)+'\n')

    def test_05_missing_transition_is_not_zero(self):
        cat = copy.deepcopy(self.g['catalogs'][0])
        row = {field: 9999 for field in cat['qn_order']}
        row.update(obs=10000, Weight=1, Uncertainty=0.01)
        mapped, missing = remap_assignments([row], cat)
        self.assertEqual(missing, 1)
        mapped = self.g['recompute_peak_weights'](mapped, False)
        clean = self.g['_sanitize_for_table'](mapped)[0]
        self.assertIsNone(clean['Delta'])
        self.assertIsNone(clean['WeightedSim'])

    def test_06_lin_roundtrip_and_isolated_state(self):
        cat = self.g['catalogs'][1]
        hit = cat['df'].iloc[0]
        row = {field: int(hit[field]) for field in cat['qn_order']}
        row.update(obs=float(hit['Freq'])+0.1, Weight=1, Uncertainty=0.0123)
        result = self.callback('lin', 'ws-confirm-write.submit_n_clicks', 1, 0, 1, {'1':[row]}, {}, None, '')
        self.assertIn('Wrote 1', result[0])
        result = self.callback('lin', 'ws-confirm-import.submit_n_clicks', 0, 1, 1, {}, {}, None, '')
        self.assertEqual(result[1]['1'][0]['Uncertainty'], 0.0123)
        layout = self.client.get('/_dash-layout').get_json()
        self.assertIn('0.0123', json.dumps(layout))
        other = WORK / 'other.json'
        other.write_text(self.config.read_text())
        second = runpy.run_path(str(ROOT/'plotcomparison_2026_7.py'), init_globals={'WORKSPACE_CONFIG': str(other)})
        self.assertNotEqual(second['WORKSPACE_STATE_DIR'], self.g['WORKSPACE_STATE_DIR'])
        self.assertEqual(second['_load_assignment_autosave'](), {})

    def test_07_real_pickett_in_copied_directory(self):
        source_value = os.environ.get('ASSIGNER_TEST_CAT')
        if not source_value:
            self.skipTest('Optional real Pickett test: set ASSIGNER_TEST_CAT to a CAT with matching inputs.')
        source_cat = Path(source_value).expanduser().resolve()
        directory = WORK / 'pickett-copy'
        directory.mkdir()
        originals = {}
        for ext in ('.par', '.var', '.int', '.lin'):
            source = source_cat.with_suffix(ext)
            originals[source] = hashlib.sha256(source.read_bytes()).hexdigest()
            shutil.copy2(source, directory / source.name)
        runner = Runner()
        runner.start(directory / source_cat.name, ['spfit', 'spcat'], {})
        deadline = time.monotonic() + 120
        while runner.running and time.monotonic() < deadline:
            time.sleep(0.1)
        if runner.running:
            runner.stop()
            self.fail('Pickett exceeded 120 seconds')
        log = '\n'.join(runner.lines)
        (WORK / 'pickett-result.log').write_text(log, encoding='utf-8')
        self.assertIn('[spfit] Exit code: 0', log)
        self.assertIn('[spcat] Exit code: 0', log)
        self.assertTrue((directory / source_cat.name).exists())
        for source, digest in originals.items():
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)

    def test_08_backup_permission_failure_reports_failed_without_launch(self):
        import threading
        runner = Runner()
        runner.running = True
        guard = threading.Lock()
        guard.acquire()
        with patch('spectrum_workspace_v7.Path.mkdir', side_effect=PermissionError('backup access denied')):
            with patch('spectrum_workspace_v7.subprocess.Popen') as process:
                runner._work(WORK, [('spfit', 'unused.exe', 'sample')], guard)
                process.assert_not_called()
        self.assertFalse(runner.running)
        self.assertFalse(guard.locked())
        self.assertEqual(runner.revision, 1)
        self.assertIn('Failed:', runner.status)
        self.assertIn('write permission', runner.status)

    def test_09_configuration_create_edit_browse_and_conflict(self):
        from spectrum_config_v7 import config_api
        directory = WORK / 'configs'
        directory.mkdir()
        client = config_api(directory).test_client()
        payload = {'name':'My spectrum', 'csv_file':str(WORK/'spectrum.csv'),
                   'cat_files':[str(WORK/'first/sample.cat'), str(WORK/'second/sample.cat')]}
        created = client.post('/config-api/save', json=payload)
        self.assertEqual(created.status_code, 200, created.get_json())
        key = created.get_json()['id']
        path = next(directory.glob('config*.json'))
        data = json.loads(path.read_text())
        data['uncertainty']['fid_time_us'] = 25
        data['qn_labels'] = {'UpperQ1':'F'}
        data['custom_option'] = 'keep me'
        path.write_text(json.dumps(data))
        edit = client.get('/config-api/config/'+key).get_json()
        edit.update(name='Renamed spectrum', cat_files=list(reversed(edit['cat_files'])))
        changed = client.post('/config-api/save', json=edit)
        self.assertEqual(changed.status_code, 200, changed.get_json())
        result = json.loads(path.read_text())
        self.assertEqual(result['uncertainty']['fid_time_us'],25)
        self.assertEqual(result['custom_option'],'keep me')
        self.assertEqual(result['qn_labels'],data['qn_labels'])
        self.assertEqual(client.get('/config-api/list').get_json()[0]['label'], 'Renamed spectrum')
        self.assertEqual(client.post('/config-api/save',json=edit).status_code,400)
        listing = client.get('/config-api/browse', query_string={'path':str(WORK/'first'),'kind':'cat'}).get_json()
        self.assertIn('sample.cat',[x['name'] for x in listing['entries']])
        self.assertNotIn('sample.par',[x['name'] for x in listing['entries']])
        payload['cat_files'] = []
        self.assertEqual(client.post('/config-api/save',json=payload).status_code,400)

    def test_10_catalog_reorder_restores_assignments_by_path(self):
        before = self.g['_load_assignment_autosave']()
        self.assertIn('1',before)
        self.g['catalogs'].reverse()
        try:
            after = self.g['_load_assignment_autosave']()
            self.assertEqual(after['0'][0]['obs'], before['1'][0]['obs'])
        finally:
            self.g['catalogs'].reverse()

    def test_11_original_intensity_plot_and_zoom_callback(self):
        import numpy as np
        args = (None, {}, None, {'x':[5000,5999], 'y':[-.1,1.1]}, 'zoom', 0, None, [], {}, 0, None, 0)
        normal = self.callback('update_plot', 'original-intensity.value', *args, [])
        raw = self.callback('update_plot', 'original-intensity.value', *args, ['raw'])
        factor = self.g['MEAS_INTENSITY_REFERENCE']
        normal_trace = next(t for t in normal[0].data if t.name == 'Measured')
        raw_trace = next(t for t in raw[0].data if t.name == 'Measured')
        np.testing.assert_allclose(raw_trace.y, np.array(normal_trace.y)*factor)
        self.assertEqual(normal[0].layout.yaxis.title.text,'Intensity')
        self.assertEqual(raw[1], normal[1])  # internal zoom stays normalized
        zoom = self.callback('handle_all_zoom_events', 'spectrum-plot.relayoutData',
            {'yaxis.range':[0,factor*.5]}, None, 0,0,0,0,0,0,5000,5999,
            {'x':[5000,5999],'y':None}, [], None,20,['raw'])
        np.testing.assert_allclose(zoom[0]['y'],[0,.5])


if __name__ == '__main__':
    print('Test artifacts:', WORK)
    unittest.main(verbosity=2)
