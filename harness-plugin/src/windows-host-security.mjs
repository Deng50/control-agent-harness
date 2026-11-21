/** Query the actual Host token and directory rights, without launching a Host-token interpreter. */
import { createRequire } from 'node:module';
import { dirname } from 'node:path';

// The installed pinned Windows ACL backend already uses this exact platform FFI package.
const backend = createRequire(new URL('../../vendor/deepseek-harness/packages/sandbox/sandbox-windows-acl/package.json', import.meta.url));
const ffi = process.platform === 'win32' ? backend('koffi') : null;
const kernel = ffi?.load('kernel32.dll'), security = ffi?.load('advapi32.dll');
const call = (dll, name, returns, ...args) => dll?.func('__stdcall', name, returns, args);
const ptr = 'void *', wide = 'str16', u32 = 'uint32_t';
const lastError = call(kernel, 'GetLastError', u32);
const currentProcess = call(kernel, 'GetCurrentProcess', ptr);
const close = call(kernel, 'CloseHandle', 'int', ptr);
const free = call(kernel, 'LocalFree', ptr, ptr);
const openToken = call(security, 'OpenProcessToken', 'int', ptr, u32, ptr);
const tokenInfo = call(security, 'GetTokenInformation', 'int', ptr, 'int', ptr, u32, ptr);
const namedSecurity = call(security, 'GetNamedSecurityInfoW', u32, wide, 'int', u32, ptr, ptr, ptr, ptr, ptr);
const sidLength = call(security, 'GetLengthSid', u32, ptr);
const volumeRoot = call(kernel, 'GetVolumePathNameW', 'int', wide, ptr, u32);
const volumeInfo = call(kernel, 'GetVolumeInformationW', 'int', wide, ptr, u32, ptr, ptr, ptr, ptr, u32);
const open = call(kernel, 'CreateFileW', ptr, wide, u32, u32, ptr, u32, u32, ptr);
const fileInfo = call(kernel, 'GetFileInformationByHandleEx', 'int', ptr, 'int', ptr, u32);
const validHandle = handle => handle !== null && BigInt.asIntN(64, handle) !== -1n;

function check(success, operation, code) {
  if (success) return;
  const winerror = code ?? lastError();
  throw Object.assign(new Error(operation + ' failed (Win32 ' + winerror + ')'), {winerror, operation});
}

/** Pin each existing ancestor against rename/delete while Host operations use its pathname. */
export function pinDirectoryPath(path, allowMissing = false) {
  if (!ffi) return () => {};
  const chain = [], handles = [];
  for (let current = path; ; current = dirname(current)) {
    chain.push(current);
    if (dirname(current) === current) break;
  }
  const release = () => { while (handles.length) close(handles.pop()); };
  try {
    for (const directory of chain.reverse()) {
      const handle = open(directory, 0x80, 3, null, 3, 0x02200000, null);
      if (!validHandle(handle)) {
        const code = lastError();
        if (allowMissing && (code === 2 || code === 3)) break;
        check(false, 'pin_directory', code);
      }
      handles.push(handle);
      const entry = Buffer.alloc(8);
      check(fileInfo(handle, 9, entry, 8), 'inspect_pinned_directory');
      const flags = entry.readUInt32LE();
      if ((flags & 0x10) === 0 || (flags & 0x400) !== 0) throw new Error('Workspace path contains a file or reparse point: ' + directory);
    }
    return release;
  } catch (error) { release(); throw error; }
}

function sidAt(slot) {
  const address = ffi.decode(slot, ptr), length = sidLength(address);
  check(length >= 8 && length <= 68, 'GetLengthSid');
  const bytes = Buffer.from(ffi.view(address, length));
  const count = bytes[1];
  if (8 + count * 4 !== length) throw new Error('Invalid SID length from kernel');
  const parts = [bytes[0], bytes.readUIntBE(2, 6)];
  for (let index = 0; index < count; index++) parts.push(bytes.readUInt32LE(8 + index * 4));
  return 'S-' + parts.join('-');
}

function tokenSid(token, kind) {
  const size = Buffer.alloc(4);
  tokenInfo(token, kind, null, 0, size);
  const length = size.readUInt32LE();
  check(length >= 8 && length < 65536, 'GetTokenInformation(size)');
  const buffer = Buffer.alloc(length);
  check(tokenInfo(token, kind, buffer, length, size), 'GetTokenInformation');
  return sidAt(buffer);
}

/** The directory handles request WRITE_DAC/WRITE_OWNER individually and are closed without modifying anything. */
export function inspectHostSecurity(workspace) {
  const slot = Buffer.alloc(8), descriptor = Buffer.alloc(8);
  check(openToken(currentProcess(), 8, slot), 'OpenProcessToken');
  const token = ffi.decode(slot, ptr);
  let user, integrity, owner;
  try { user = tokenSid(token, 1); integrity = tokenSid(token, 25); } finally { close(token); }
  const result = namedSecurity(workspace, 1, 1, slot, null, null, null, descriptor);
  check(result === 0, 'GetNamedSecurityInfoW', result);
  try { owner = sidAt(slot); } finally { free(ffi.decode(descriptor, ptr)); }
  const root = Buffer.alloc(65536), filesystem = Buffer.alloc(128);
  check(volumeRoot(workspace, root, root.length / 2), 'GetVolumePathNameW');
  const string = buffer => buffer.toString('utf16le').split('\0', 1)[0];
  check(volumeInfo(string(root), null, 0, null, null, null, filesystem, filesystem.length / 2), 'GetVolumeInformationW');
  const effective_rights = {}, right_errors = {};
  for (const [right, mask] of Object.entries({WRITE_DAC:0x40000, WRITE_OWNER:0x80000})) {
    const handle = open(workspace, mask, 7, null, 3, 0x02200000, null);
    effective_rights[right] = validHandle(handle);
    if (effective_rights[right]) close(handle);
    else right_errors[right] = lastError();
  }
  return {workspace, user_sid:user, integrity_sid:integrity, owner_sid:owner, owner_is_current_user:user === owner,
    filesystem:string(filesystem), effective_rights, right_errors};
}
