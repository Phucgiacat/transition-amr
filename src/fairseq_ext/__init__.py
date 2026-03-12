"""Register fairseq extension modules.

In some editable-install environments, absolute imports like
``from fairseq_ext.data ...`` can fail during early package initialization.
We proactively load the local ``fairseq_ext.data`` package from disk to keep
task auto-registration stable.
"""

import importlib
import importlib.util
from pathlib import Path
import sys


def _ensure_local_data_package() -> None:
	"""Ensure fairseq_ext.data resolves to this repo's local package."""
	try:
		importlib.import_module('fairseq_ext.data')
		return
	except Exception:
		pass

	pkg_dir = Path(__file__).resolve().parent
	src_dir = pkg_dir.parent
	data_init = pkg_dir / 'data' / '__init__.py'

	if str(src_dir) not in sys.path:
		sys.path.insert(0, str(src_dir))

	if data_init.exists():
		spec = importlib.util.spec_from_file_location(
			'fairseq_ext.data',
			str(data_init),
			submodule_search_locations=[str(data_init.parent)],
		)
		if spec and spec.loader:
			module = importlib.util.module_from_spec(spec)
			sys.modules['fairseq_ext.data'] = module
			spec.loader.exec_module(module)


_ensure_local_data_package()

# Register all user-defined modules to fairseq.
import fairseq_ext.criterions  # noqa
import fairseq_ext.models  # noqa
import fairseq_ext.tasks  # noqa
