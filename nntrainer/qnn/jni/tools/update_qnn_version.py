import os
import re
import sys
import shutil

VENDOR_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vendor'))
VENDOR_STAMP = os.path.join(VENDOR_DIR, '.nntrainer-qnn-api36-compat-v2')
SAMPLEAPP_SRC_PREFIX = 'examples/QNN/SampleApp/SampleApp/src'
GENIE_SRC_PREFIX = 'examples/Genie/Genie/src'

def is_from_qnn(file_path):
	if not file_path.endswith(('.cpp', '.hpp', '.h')):
		return False
	with open(file_path, 'r') as f:
		return 'Qualcomm Technologies, Inc.' in f.read()

def recursive_overwrite(src, dest, ignore=None):
	if os.path.isdir(src):
		if not os.path.isdir(dest):
			os.makedirs(dest)
		files = os.listdir(src)
		ignored = ignore(src, files) if ignore else set()
		for f in files:
			if f not in ignored:
				recursive_overwrite(os.path.join(src, f), os.path.join(dest, f), ignore)
	else:
		shutil.copyfile(src, dest)

def find_replace_in_file(file_path, old_text, new_text):
	with open(file_path, 'r') as f:
		file_data = f.read()
	file_data = file_data.replace(old_text, new_text)
	with open(file_path, 'w') as f:
		f.write(file_data)

def replace_exactly(text, old_text, new_text, description, expected_count=1):
	actual_count = text.count(old_text)
	if actual_count != expected_count:
		raise RuntimeError(
			f'{description}: expected {expected_count} occurrence(s), found {actual_count}'
		)
	return text.replace(old_text, new_text, expected_count)

def replace_regex_exactly(text, pattern, replacement, description, expected_count=1):
	replaced, actual_count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
	if actual_count != expected_count:
		raise RuntimeError(
			f'{description}: expected {expected_count} occurrence(s), found {actual_count}'
		)
	return replaced

def require_exact_count(text, expected_text, description, expected_count=1):
	actual_count = text.count(expected_text)
	if actual_count != expected_count:
		raise RuntimeError(
			f'{description}: expected {expected_count} occurrence(s), found {actual_count}'
		)

def patch_backend_extensions():
	header_path = os.path.join(VENDOR_DIR, 'qnn-api/BackendExtensions.hpp')
	source_path = os.path.join(VENDOR_DIR, 'qnn-api/BackendExtensions.cpp')
	if not os.path.exists(header_path) or not os.path.exists(source_path):
		raise RuntimeError('QNN SDK is missing BackendExtensions sources')

	with open(header_path, 'r') as f:
		header_data = f.read()
	header_data = replace_exactly(
		header_data,
		'  ~BackendExtensions();\n'
		'  qnn::tools::netrun::IBackend* interface();',
		'  ~BackendExtensions() noexcept;\n'
		'  BackendExtensions(const BackendExtensions&) = delete;\n'
		'  BackendExtensions& operator=(const BackendExtensions&) = delete;\n'
		'  BackendExtensions(BackendExtensions&&) = delete;\n'
		'  BackendExtensions& operator=(BackendExtensions&&) = delete;\n'
		'\n'
		'  bool shutdown() noexcept;\n'
		'  qnn::tools::netrun::IBackend* interface();',
		'BackendExtensions ownership API',
	)
	header_data = replace_exactly(
		header_data,
		' private:\n'
		'  qnn::tools::netrun::IBackend* m_backendInterface;\n'
		'  qnn::tools::netrun::DestroyBackendInterfaceFnType_t m_destroyBackendInterfaceFn;',
		' private:\n'
		'  void* m_libraryHandle;\n'
		'  qnn::tools::netrun::IBackend* m_backendInterface;\n'
		'  qnn::tools::netrun::DestroyBackendInterfaceFnType_t m_destroyBackendInterfaceFn;\n'
		'  bool m_cleanupFailed;',
		'BackendExtensions ownership state',
	)

	with open(source_path, 'r') as f:
		source_data = f.read()
	source_data = replace_exactly(
		source_data,
		'    : m_backendInterface(nullptr), m_destroyBackendInterfaceFn(nullptr) {',
		'    : m_libraryHandle(nullptr),\n'
		'      m_backendInterface(nullptr),\n'
		'      m_destroyBackendInterfaceFn(nullptr),\n'
		'      m_cleanupFailed(false) {\n'
		'  if (m_resourceManager == nullptr) {\n'
		'    throw std::runtime_error("Backend extensions require a resource manager.");\n'
		'  }',
		'BackendExtensions member initialization',
	)
	source_data = replace_exactly(
		source_data,
		'  void* libHandle =\n'
		'      m_resourceManager->dlOpen(',
		'  m_libraryHandle =\n'
		'      m_resourceManager->dlOpen(',
		'BackendExtensions DSO ownership publication',
	)
	source_data = replace_regex_exactly(
		source_data,
		r'\blibHandle\b',
		'm_libraryHandle',
		'BackendExtensions DSO handle use',
		expected_count=3,
	)
	# Roll back only failures that the wrapper itself raises after a vendor call
	# returned normally. A catch-all must not close the extension DSO while a
	# DSO-defined exception object/typeinfo is still alive during rethrow.
	source_data = replace_regex_exactly(
		source_data,
		r'^([ \t]*)throw std::runtime_error\(',
		r'\1(void)shutdown();\n\1throw std::runtime_error(',
		'BackendExtensions deterministic construction rollback',
		expected_count=13,
	)
	source_data = replace_exactly(
		source_data,
		'BackendExtensions::~BackendExtensions() { '
		'm_destroyBackendInterfaceFn(m_backendInterface); }',
		'bool BackendExtensions::shutdown() noexcept {\n'
		'  if (m_cleanupFailed) {\n'
		'    return false;\n'
		'  }\n'
		'\n'
		'  if (m_backendInterface != nullptr) {\n'
		'    if (m_destroyBackendInterfaceFn == nullptr) {\n'
		'      m_cleanupFailed = true;\n'
		'      return false;\n'
		'    }\n'
		'    try {\n'
		'      m_destroyBackendInterfaceFn(m_backendInterface);\n'
		'    } catch (...) {\n'
		'      m_cleanupFailed = true;\n'
		'      return false;\n'
		'    }\n'
		'    m_backendInterface = nullptr;\n'
		'  }\n'
		'\n'
		'  if (m_libraryHandle != nullptr) {\n'
		'    try {\n'
		'      if (pal::dynamicloading::dlClose(m_libraryHandle) != 0) {\n'
		'        m_cleanupFailed = true;\n'
		'        return false;\n'
		'      }\n'
		'    } catch (...) {\n'
		'      m_cleanupFailed = true;\n'
		'      return false;\n'
		'    }\n'
		'    m_libraryHandle = nullptr;\n'
		'  }\n'
		'\n'
		'  m_destroyBackendInterfaceFn = nullptr;\n'
		'  return true;\n'
		'}\n'
		'\n'
		'BackendExtensions::~BackendExtensions() noexcept {\n'
		'  if (!shutdown()) {\n'
		'    QNN_ERROR("Backend extension cleanup failed; retaining its DSO");\n'
		'  }\n'
		'}',
		'BackendExtensions teardown state machine',
	)

	# Treat the generated API as an all-or-nothing contract. These checks catch
	# an accidentally broad transform as well as a future SDK source drift.
	require_exact_count(
		header_data,
		'  bool shutdown() noexcept;',
		'BackendExtensions shutdown declaration',
	)
	require_exact_count(
		header_data,
		'  void* m_libraryHandle;',
		'BackendExtensions DSO owner declaration',
	)
	require_exact_count(
		header_data,
		'  bool m_cleanupFailed;',
		'BackendExtensions sticky cleanup declaration',
	)
	require_exact_count(
		source_data,
		'bool BackendExtensions::shutdown() noexcept {',
		'BackendExtensions shutdown definition',
	)
	require_exact_count(
		source_data,
		'BackendExtensions::~BackendExtensions() noexcept {',
		'BackendExtensions noexcept destructor definition',
	)
	require_exact_count(
		source_data,
		'(void)shutdown();',
		'BackendExtensions deterministic rollback sites',
		expected_count=13,
	)
	require_exact_count(
		source_data,
		'BackendExtensions::~BackendExtensions() { '
		'm_destroyBackendInterfaceFn(m_backendInterface); }',
		'BackendExtensions legacy destructor',
		expected_count=0,
	)
	require_exact_count(
		source_data,
		'void* libHandle',
		'BackendExtensions unowned DSO handle',
		expected_count=0,
	)

	with open(header_path, 'w') as f:
		f.write(header_data)
	with open(source_path, 'w') as f:
		f.write(source_data)

def validate_sdk_layout(qnn_root):
	missing_paths = []
	for src_rel in target_src_dirs.values():
		src_path = os.path.join(qnn_root, src_rel)
		if not os.path.isdir(src_path):
			missing_paths.append(src_rel)
	for src_rel, _ in extra_files:
		src_path = os.path.join(qnn_root, src_rel)
		if not os.path.isfile(src_path):
			missing_paths.append(src_rel)
	if missing_paths:
		raise RuntimeError(
			'QNN SDK layout is missing required paths:\n  ' +
			'\n  '.join(missing_paths)
		)

def remove_path(path):
	if os.path.isdir(path):
		shutil.rmtree(path)
	elif os.path.exists(path):
		os.remove(path)

# ── 1. Copy from SDK ──────────────────────────────────────────────────────

# Maps target dirs (under VENDOR_DIR) to source dirs (under QNN_SDK_ROOT)
target_src_dirs = {
	'Log':            os.path.join(SAMPLEAPP_SRC_PREFIX, 'Log'),
	'PAL':            os.path.join(SAMPLEAPP_SRC_PREFIX, 'PAL'),
	'QNN':            'include/QNN',
	'Utils':          os.path.join(SAMPLEAPP_SRC_PREFIX, 'Utils'),
	'WrapperUtils':   os.path.join(SAMPLEAPP_SRC_PREFIX, 'WrapperUtils'),
	'qnn-api':        os.path.join(GENIE_SRC_PREFIX, 'qualla/engines/qnn-api'),
}

# Additional individual files to copy: (src_rel_path, target_rel_path)
extra_files = [
	(os.path.join(SAMPLEAPP_SRC_PREFIX, 'SampleApp.hpp'),
	 os.path.join(VENDOR_DIR, 'QNN.hpp')),
	(os.path.join(SAMPLEAPP_SRC_PREFIX, 'QnnTypeMacros.hpp'),
	 os.path.join(VENDOR_DIR, 'QnnTypeMacros.hpp')),
	(os.path.join(GENIE_SRC_PREFIX, 'resource-manager/include/ResourceManager.hpp'),
	 os.path.join(VENDOR_DIR, 'qnn-api/ResourceManager.hpp')),
	(os.path.join(GENIE_SRC_PREFIX, 'resource-manager/src/ResourceManager.cpp'),
	 os.path.join(VENDOR_DIR, 'qnn-api/ResourceManager.cpp')),
	(os.path.join(GENIE_SRC_PREFIX, 'qualla/include/qualla/detail/Log.hpp'),
	 os.path.join(VENDOR_DIR, 'qnn-api/qualla/detail/Log.hpp')),
	(os.path.join(GENIE_SRC_PREFIX, 'qualla/include/qualla/detail/dlOpenWrapper.hpp'),
	 os.path.join(VENDOR_DIR, 'qnn-api/qualla/detail/dlOpenWrapper.hpp')),
]

if __name__ == '__main__':
	# Resolve SDK root: --qnn-sdk-root=PATH flag takes priority, then env var.
	qnn_root = None
	for arg in sys.argv[1:]:
		if arg.startswith('--qnn-sdk-root='):
			qnn_root = arg[len('--qnn-sdk-root='):]
			break
	if not qnn_root:
		qnn_root = os.environ.get('QNN_SDK_ROOT', '')
	if not qnn_root:
		sys.exit(
			'Set --qnn-sdk-root=<path> or the QNN_SDK_ROOT env variable to your '
			'Qualcomm QNN SDK root (e.g. .../qairt/2.47.0.x)'
		)
	validate_sdk_layout(qnn_root)
	remove_path(VENDOR_STAMP)

	# Copy directories
	for target_name, src_rel in target_src_dirs.items():
		src_path = os.path.join(qnn_root, src_rel)
		target_path = os.path.join(VENDOR_DIR, target_name)
		if not os.path.exists(src_path):
			print(f'WARNING: {src_rel} does not exist in SDK, skipping')
			continue
		# Remove old Qualcomm files from target
		for root, dirs, files in os.walk(target_path):
			for f in files:
				cur = os.path.join(root, f)
				if is_from_qnn(cur):
					os.remove(cur)
		recursive_overwrite(src_path, target_path)

	# Copy extra files
	for src_rel, target_rel in extra_files:
		src_path = os.path.join(qnn_root, src_rel)
		if os.path.exists(src_path):
			os.makedirs(os.path.dirname(target_rel), exist_ok=True)
			shutil.copyfile(src_path, target_rel)

	# ── 1b. Pin QNN API version ───────────────────────────────────────────
	_qnn_common_h = os.path.join(VENDOR_DIR, 'QNN/QnnCommon.h')
	if not os.path.exists(_qnn_common_h):
		sys.exit(f'Expected {_qnn_common_h} to exist after SDK copy — check SDK layout.')
	_expected_minor = 36
	_found_minor = None
	with open(_qnn_common_h, 'r') as _fh:
		for _line in _fh:
			_m = re.match(r'\s*#define\s+QNN_API_VERSION_MINOR\s+(\d+)', _line)
			if _m:
				_found_minor = int(_m.group(1))
				break
	if _found_minor is None:
		sys.exit(f'{_qnn_common_h} does not define QNN_API_VERSION_MINOR — unexpected SDK layout.')
	if _found_minor != _expected_minor:
		sys.exit(
			f'Unexpected QNN API version: expected MINOR={_expected_minor} (QNN SDK 2.47.x), '
			f'found MINOR={_found_minor}. '
			f'Update the expected version in this script if you intend to bump the SDK.'
		)

	# ── 2. Remove unused files/directories ────────────────────────────────

	# Unused QNN backend subdirs (we only use HTP and System)
	for d in ['CPU', 'DSP', 'GPU', 'GPU.unused', 'HTA', 'Saver', 'IR',
	          'LPAI', 'TFLiteDelegate', 'GenAiTransformer',
	          'LoraAdapterBinUpdater', 'HTPQEMU']:
		remove_path(os.path.join(VENDOR_DIR, 'QNN', d))

	# HTP/core is internal headers not needed for our build
	remove_path(os.path.join(VENDOR_DIR, 'QNN', 'HTP', 'core'))

	# PAL windows (not needed for Android)
	remove_path(os.path.join(VENDOR_DIR, 'PAL/src/windows'))

	# Unused files from qnn-api with unresolvable dependencies
	for f in ['ClientBuffer.hpp', 'ClientBuffer.cpp',
	          'DmaBufAllocator.hpp', 'DmaBufAllocator.cpp', 'IBufferAlloc.hpp',
	          'IOTensor.hpp', 'IOTensor.cpp', 'qnn-utils.hpp', 'qnn-utils.cpp',
	          'QnnApi.hpp', 'QnnApi.cpp', 'QnnApiUtils.hpp', 'QnnApiUtils.cpp',
	          'RpcMem.hpp', 'RpcMem.cpp', 'QnnWrapperUtils.hpp',
	          'BufferUtils.hpp', 'BufferUtils.cpp',
	          'QnnTypeUtils.hpp', 'QnnTypeUtils.cpp',
	          'QnnTypeMacros.hpp']:
		remove_path(os.path.join(VENDOR_DIR, 'qnn-api', f))

	# Unused qnn-api subdirs with unresolvable dependencies
	for d in ['buffer', 'config', 'PAL']:
		remove_path(os.path.join(VENDOR_DIR, 'qnn-api', d))

	# Orphaned qualla/ at vendor root (duplicate of qnn-api/qualla/)
	remove_path(os.path.join(VENDOR_DIR, 'qualla'))

	# ── 3. Apply compatibility patches ─────────────────────────────────────

	# SDK uses SampleApp.hpp, our build uses QNN.hpp
	find_replace_in_file(os.path.join(VENDOR_DIR, 'Utils/DynamicLoadUtil.hpp'),
		'SampleApp.hpp', 'QNN.hpp')
	find_replace_in_file(os.path.join(VENDOR_DIR, 'Utils/QnnSampleAppUtils.hpp'),
		'SampleApp.hpp', 'QNN.hpp')

	# IOTensor.hpp: make members public for our usage
	find_replace_in_file(os.path.join(VENDOR_DIR, 'Utils/IOTensor.hpp'),
		'private', 'public')

	# dlOpenWrapper.hpp: fix const-correctness
	dl_open_wrapper = os.path.join(VENDOR_DIR, 'qnn-api/qualla/detail/dlOpenWrapper.hpp')
	find_replace_in_file(dl_open_wrapper, 'static const int s_anchor', 'static int s_anchor')
	find_replace_in_file(dl_open_wrapper, 'reinterpret_cast<const void*>', 'reinterpret_cast<void*>')

	# BackendExtensions owns a plugin DSO and a factory-created interface. Apply
	# this fail-closed transform only to the pinned SDK layout: a silent partial
	# match would restore the upstream leak or a throwing-destructor terminate.
	patch_backend_extensions()

	with open(VENDOR_STAMP, 'w') as f:
		f.write('QNN API 2.36 / nntrainer compatibility revision 2\n')

	print('QNN vendor files updated successfully.')
