"""Exercise the production shell tempfile recipe on the build host."""
import os
from pathlib import Path
import subprocess


def test_python_download_tempfiles_are_unique_and_owned(tmp_path):
    source = Path('panel/scripts/bundle-python.sh').read_text()
    assignment = next(line for line in source.splitlines()
                      if line.startswith('STANDALONE_TARBALL="$(mktemp '))
    env = {**os.environ, 'TMPDIR': str(tmp_path)}
    paths = []
    for _ in range(2):
        output = subprocess.check_output(
            ['bash', '-c', assignment + '\nprintf "%s" "$STANDALONE_TARBALL"'],
            env=env, text=True,
        )
        path = Path(output)
        assert path.parent == tmp_path
        assert path.is_file()
        paths.append(path)
    assert paths[0] != paths[1]
    assert sorted(tmp_path.iterdir()) == sorted(paths)
