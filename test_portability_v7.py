"""Portable regressions; no installed Pickett binaries or personal files needed."""
import os
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from spectrum_workspace_v7 import resolve_executable, Runner


class PortabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='assigner space ')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.binary = self.folder / 'spfit'
        self.binary.write_bytes(b'test placeholder, never executed')
        self.binary.chmod(0o755)

    def test_json_precedes_environment_and_path(self):
        with patch.dict(os.environ, {'SPFIT_PATH': 'missing-environment-file'}):
            self.assertEqual(resolve_executable('spfit', {'spfit_path': 'spfit'}, self.folder), str(self.binary.resolve()))

    def test_environment_and_path_fallback(self):
        with patch.dict(os.environ, {'SPFIT_PATH': str(self.binary)}):
            self.assertEqual(resolve_executable('spfit', {}), str(self.binary.resolve()))
        with patch.dict(os.environ, {}, clear=True), patch('spectrum_workspace_v7.shutil.which', return_value=str(self.binary)):
            self.assertEqual(resolve_executable('spfit', {}), str(self.binary.resolve()))

    def test_stale_override_is_actionable_not_silent_fallback(self):
        with patch.dict(os.environ, {'SPFIT_PATH': str(self.binary)}):
            with self.assertRaisesRegex(ValueError, 'lower-priority'):
                resolve_executable('spfit', {'spfit_path': str(self.folder/'missing')})

    def test_home_expansion(self):
        with patch('spectrum_workspace_v7.Path.expanduser', return_value=self.binary):
            self.assertEqual(resolve_executable('spfit', {'spfit_path': '~/spfit'}), str(self.binary.resolve()))

    @unittest.skipIf(os.name == 'nt', 'POSIX executable permissions')
    def test_execute_permission(self):
        self.binary.chmod(0o644)
        with self.assertRaisesRegex(ValueError, 'chmod'):
            resolve_executable('spfit', {'spfit_path': str(self.binary)})

    def test_missing_binary_does_not_start_job(self):
        runner = Runner()
        with patch.dict(os.environ, {}, clear=True), patch('spectrum_workspace_v7.shutil.which', return_value=None):
            with self.assertRaisesRegex(ValueError, 'docs/INSTALL.md'):
                runner.start(self.folder/'sample.cat', ['spfit'], {})
        self.assertFalse(runner.running)

    def test_edit_preserves_delimiter_for_relative_spectrum(self):
        from spectrum_config_v7 import config_api
        from spectrum_workspace_v7 import identity
        (self.folder/'trace.txt').write_text('frequency|intensity\n5000|1\n5001|2\n')
        (self.folder/'sample.cat').write_text('placeholder')
        config = self.folder/'config-sample.json'
        config.write_text(json.dumps({'name':'sample', 'csv_file':'trace.txt',
            'cat_files':['sample.cat'], 'csv_separator':'|'}))
        client = config_api(self.folder).test_client()
        payload = client.get('/config-api/config/'+identity(config)).get_json()
        payload['name'] = 'renamed'
        response = client.post('/config-api/save', json=payload)
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(json.loads(config.read_text())['csv_separator'], '|')

    def test_log_write_failure_releases_job_lock(self):
        import threading
        runner = Runner()
        runner.running = True
        guard = threading.Lock()
        guard.acquire()
        with patch('spectrum_workspace_v7.Path.open', side_effect=PermissionError('log denied')):
            runner._work(self.folder, [('spfit', str(self.binary), 'missing')], guard)
        self.assertFalse(guard.locked())
        self.assertFalse(runner.running)
        self.assertIn('Failed:', runner.status)
        self.assertTrue(any('log denied' in line for line in runner.lines))


if __name__ == '__main__':
    unittest.main()
