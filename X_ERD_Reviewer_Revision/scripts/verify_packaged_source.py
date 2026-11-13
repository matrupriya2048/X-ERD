"""Check that the installed package matches the reviewed source manifest."""
from pathlib import Path
import hashlib, importlib.metadata, json
import xerd

ROOT = Path(__file__).resolve().parents[1]

def main():
    expected = json.loads((ROOT/'config/source_integrity.json').read_text())
    installed = Path(xerd.__file__).resolve().parent
    checked = []
    for name, sha in expected['sha256'].items():
        path = ROOT/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise AssertionError('Packaged source differs: '+name)
        if name.startswith('src/xerd/'):
            packaged = installed/Path(name).relative_to('src/xerd')
            if hashlib.sha256(packaged.read_bytes()).hexdigest() != sha:
                raise AssertionError('Installed module differs: '+name)
        checked.append(name)
    version = importlib.metadata.version('xerd-reviewer-revision')
    if version != '0.3.0':
        raise AssertionError('Incorrect installed version: '+version)
    result = {'status':'passed', 'files_checked':len(checked), 'files':checked,
              'installed_module_directory':str(installed), 'version':version,
              'scope':'Run in the final Docker image without source bind mounts.'}
    (ROOT/'results/packaged_source_verification.json').write_text(json.dumps(result,indent=2))
    print('PASS:',len(checked),'packaged files; installed version',version)

if __name__ == '__main__':
    main()
