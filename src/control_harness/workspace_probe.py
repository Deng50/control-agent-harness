"""工作区预检只使用标准库。诊断读取内核授权结果，读写探针只操作自己创建的随机子目录。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import uuid


class ProbeError(Exception):
    """保留最先失败的系统调用，即使后续清理也遇到文件占用。"""

    def __init__(self, operation, path, cause):
        super().__init__(str(cause))
        self.diagnostic = {
            "operation": operation, "path": str(path),
            "code": type(cause).__name__, "os_code": getattr(cause, "winerror", None) or getattr(cause, "errno", None),
            "os_domain": "Win32" if getattr(cause, "winerror", None) else ("errno" if getattr(cause, "errno", None) else None),
            "message": str(cause),
        }


def _windows_security(directory):
    """按当前进程令牌打开目录，分别请求两种权限。不会设置 DACL、所有者或完整性标签。"""
    import ctypes as c
    from ctypes import wintypes as w

    kernel = c.WinDLL("kernel32", use_last_error=True)
    security = c.WinDLL("advapi32", use_last_error=True)
    pointer = c.c_void_p
    out = c.POINTER(pointer)

    def api(library, name, result, *arguments):
        fn = getattr(library, name)
        fn.argtypes, fn.restype = arguments, result
        return fn

    close = api(kernel, "CloseHandle", w.BOOL, w.HANDLE)
    free = api(kernel, "LocalFree", pointer, pointer)
    sid_string = api(security, "ConvertSidToStringSidW", w.BOOL, pointer, out)

    def sid_text(sid):
        value = pointer()
        if not sid_string(sid, c.byref(value)):
            raise c.WinError(c.get_last_error())
        try:
            return c.wstring_at(value)
        finally:
            free(value)

    open_token = api(security, "OpenProcessToken", w.BOOL, w.HANDLE, w.DWORD, out)
    token_info = api(security, "GetTokenInformation", w.BOOL, w.HANDLE, w.DWORD, pointer, w.DWORD, c.POINTER(w.DWORD))
    token = pointer()
    if not open_token(api(kernel, "GetCurrentProcess", w.HANDLE)(), 8, c.byref(token)):  # TOKEN_QUERY
        raise c.WinError(c.get_last_error())
    try:
        def token_sid(kind):
            size = w.DWORD()
            token_info(token, kind, None, 0, c.byref(size))
            buffer = c.create_string_buffer(size.value)
            if not token_info(token, kind, buffer, size, c.byref(size)):
                raise c.WinError(c.get_last_error())
            return sid_text(pointer.from_buffer(buffer))

        user, integrity = token_sid(1), token_sid(25)
    finally:
        close(token)

    owner, descriptor = pointer(), pointer()
    get_security = api(security, "GetNamedSecurityInfoW", w.DWORD, w.LPCWSTR, w.DWORD, w.DWORD,
                       out, out, out, out, out)
    error = get_security(str(directory), 1, 1, c.byref(owner), None, None, None, c.byref(descriptor))
    if error:
        raise c.WinError(error)
    try:
        owner_sid = sid_text(owner)
    finally:
        free(descriptor)

    filesystem = c.create_unicode_buffer(64)
    mount = c.create_unicode_buffer(32768)
    volume_path = api(kernel, "GetVolumePathNameW", w.BOOL, w.LPCWSTR, w.LPWSTR, w.DWORD)
    volume_info = api(kernel, "GetVolumeInformationW", w.BOOL, w.LPCWSTR, w.LPWSTR, w.DWORD,
                      pointer, pointer, pointer, w.LPWSTR, w.DWORD)
    if not volume_path(str(directory), mount, len(mount)) or not volume_info(
        mount.value, None, 0, None, None, None, filesystem, len(filesystem)
    ):
        raise c.WinError(c.get_last_error())
    open_file = api(kernel, "CreateFileW", w.HANDLE, w.LPCWSTR, w.DWORD, w.DWORD, pointer,
                    w.DWORD, w.DWORD, w.HANDLE)
    rights, errors = {}, {}
    for name, mask in (("WRITE_DAC", 0x40000), ("WRITE_OWNER", 0x80000)):
        # OPEN_EXISTING + BACKUP_SEMANTICS + OPEN_REPARSE_POINT；与计算进程的文件写权限独立。
        handle = open_file(str(directory), mask, 7, None, 3, 0x02200000, None)
        success = handle != pointer(-1).value
        rights[name] = success
        if success:
            close(handle)
        else:
            errors[name] = c.get_last_error()
    return {"user_sid": user, "owner_sid": owner_sid, "owner_is_current_user": user == owner_sid,
            "integrity_sid": integrity, "filesystem": filesystem.value,
            "effective_rights": rights, "right_errors": errors}


def checked_directory(raw, create=False):
    """禁止通过 reparse point 或 '..' 把原本的工作区检查转到另一处。"""
    path = Path(raw)
    if not path.is_absolute() or ".." in path.parts or (os.name == "nt" and path.drive.startswith("\\\\")):
        raise ProbeError("validate_path", path, ValueError("Provide an absolute local workspace path without '..'"))
    for item in (*reversed(path.parents), path):
        try:
            try:
                entry = item.lstat()
            except FileNotFoundError:
                if not create:
                    raise
                # 仅显式向独立 CLI 提交任务时沿用新建工作区语义，逐个检查父目录。
                item.mkdir(exist_ok=True)
                entry = item.lstat()
        except OSError as error:
            raise ProbeError("stat_directory", item, error) from error
        if stat.S_ISLNK(entry.st_mode) or getattr(entry, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ProbeError("validate_path", item, ValueError("Workspace path traverses a link or reparse point"))
        if not stat.S_ISDIR(entry.st_mode):
            raise ProbeError("stat_directory", item, NotADirectoryError(str(item)))
    return path.resolve(strict=True)


def probe_io(directory):
    """在同一目录创建、读取、原子替换并删除文件，模拟检查点的发布。绝不递归删除。"""
    operation, target = "create_probe_directory", directory
    temporary = primary = None
    owned = set()
    try:
        # Windows 的 mkdtemp 使用 0700 创建独立 DACL，可能丢弃受限令牌的 capability ACE。
        # 使用已授权父目录的继承规则，随机名称通过 mkdir 排他创建。
        new_directory = directory / f".control-preflight-{uuid.uuid4().hex}"
        new_directory.mkdir(mode=0o777 if os.name == "nt" else 0o700)
        temporary = new_directory
        first, second = temporary / "current", temporary / "candidate"
        for target, value in ((first, b"old"), (second, b"new")):
            operation = "write_exclusive"
            with target.open("xb") as handle:
                owned.add(target)
                handle.write(value)
        operation, target = "atomic_replace", first
        os.replace(second, first)
        owned.remove(second)
        operation = "read_replacement"
        if first.read_bytes() != b"new":
            raise OSError("Replacement did not preserve expected content")
        operation = "unlink"
        first.unlink()
        owned.remove(first)
    except OSError as error:
        primary = ProbeError(operation, target, error)
    finally:
        if temporary:
            for leaf in (*owned, temporary):
                try:
                    if leaf == temporary:
                        leaf.rmdir()
                    else:
                        leaf.unlink(missing_ok=True)
                except OSError as error:
                    cleanup = ProbeError("cleanup", leaf, error)
                    if primary:
                        primary.diagnostic.setdefault("cleanup", cleanup.diagnostic)
                    else:
                        primary = cleanup
    if primary:
        raise primary


def replace_host_probe(directory, name):
    """在两种身份间验证 Host 发布的文件确实可读和可替换。只触及此轮随机探针。"""
    path = Path(name)
    if path.name != "current" or not path.parent.name.startswith(".control-preflight-") or path.parent.parent != directory:
        raise ProbeError("check_host_probe", path, ValueError("Unexpected Host probe location"))
    checked_directory(path.parent)
    candidate = path.with_name("worker-" + uuid.uuid4().hex)
    operation = "read_host_file"
    created = False
    try:
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_size != 3
                or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError("Unexpected Host probe file")
        if path.read_bytes() != b"new":
            raise ValueError("Unexpected Host probe contents")
        operation = "write_worker_candidate"
        with candidate.open("xb") as handle:
            created = True
            handle.write(b"worker")
        operation = "replace_host_file"
        os.replace(candidate, path)
    except (OSError, ValueError) as error:
        primary = ProbeError(operation, candidate if operation == "write_worker_candidate" else path, error)
        try:
            if created:
                candidate.unlink(missing_ok=True)
        except OSError as cleanup:
            primary.diagnostic["cleanup"] = ProbeError("cleanup", candidate, cleanup).diagnostic
        raise primary from error


def existing_business_directories(directory, operation):
    """所有已存在的固定目录都检查跳转。只有该操作会写的目录才做实际写入探针。"""
    inspections = {"inspect_workspace", "inspect_artifact", "read_artifact_file", "read_report_arrays", "inspect_model_source",
                   "inspect_table", "inspect_material", "read_case_plan", "recommend_sampling", "prepare_handoff",
                   "prepare_imitation_evaluation", "plan_workflow"}
    proposals = {"propose_case_plan", "propose_case_plans"}
    writes = set()
    if operation not in inspections and operation not in proposals and operation != "prepare_bundle_archive":
        writes.update({".staging", "artifacts", "failed"})
    if not operation or operation in proposals or operation == "run_workflow":
        writes.add(".case-plans")
    if not operation or operation == "prepare_bundle_archive":
        writes.add(".report-downloads")
    result = []
    for name in (".control-jobs", ".case-plans", ".staging", "artifacts", "failed", ".report-downloads"):
        destination = directory / name
        try:
            destination.lstat()
        except FileNotFoundError:
            continue
        actual = checked_directory(destination)
        if actual != destination:
            raise ProbeError("validate_reserved_directory", destination, ValueError("Reserved directory redirects"))
        if name in writes:
            result.append(actual)
    return result


def inspect_workspace(workspace, *, inspect_only=False, windows_acl=False, ambient_temp=None, create=False,
                      cross_file=None, operation=None):
    """可在 Host 只读诊断、受限 worker 或独立 CLI 中运行，结果显式区分实际进程。"""
    result = {"ok": False, "workspace": str(workspace), "pid": os.getpid(), "cwd": os.getcwd(),
              "python_executable": sys.executable}
    requested_operation = operation
    operation, target = "inspect_directory", workspace
    try:
        directory = checked_directory(workspace, create=create)
        result["workspace"] = str(directory)
        destinations = existing_business_directories(directory, requested_operation)
        if os.name == "nt":
            result.update(_windows_security(directory))
            if windows_acl and result["filesystem"].upper() != "NTFS":
                raise ProbeError("filesystem", directory, ValueError("Windows ACL backend requires local NTFS"))
            if windows_acl and inspect_only:
                for right, granted in result["effective_rights"].items():
                    if not granted:
                        code = result["right_errors"][right]
                        failure = ProbeError("check_" + right, directory, PermissionError(right))
                        failure.diagnostic["os_code"] = code
                        failure.diagnostic["os_domain"] = "Win32"
                        raise failure
        else:
            result.update(uid=os.geteuid(), owner_uid=directory.stat().st_uid)
        if not inspect_only:
            probe_io(directory)
            for destination in destinations:
                probe_io(destination)
            result["writable_business_directories"] = [str(path) for path in destinations]
            if cross_file:
                replace_host_probe(directory, cross_file)
            # 显式使用运行器提供的临时根。gettempdir() 会在失败时静默回退到当前目录。
            operation = "locate_temp"
            temp = next((os.environ[key] for key in ("TMPDIR", "TEMP", "TMP") if os.environ.get(key)), None)
            if os.name == "nt":
                temp = os.environ.get("TEMP")
            if not temp and (os.name == "nt" or ambient_temp):
                raise ProbeError(operation, directory, ValueError("Worker TEMP was not set"))
            target = Path(temp or tempfile.gettempdir()).resolve(strict=True)
            result["temp"] = str(target)
            if ambient_temp and (os.name == "nt" and (
                    result["integrity_sid"] != "S-1-16-4096" or target == Path(ambient_temp).resolve()
                    or Path(os.environ.get("TMP", "")).resolve() != target)):
                raise ProbeError("check_confined_identity_and_temp", target,
                                 ValueError("Expected the runner's low-integrity token and private TEMP/TMP"))
            probe_io(target)
        result["ok"] = True
    except ProbeError as error:
        result["error"] = error.diagnostic
    except (ValueError, OSError) as error:
        result["error"] = ProbeError(operation, target, error).diagnostic
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--windows-acl", action="store_true")
    parser.add_argument("--ambient-temp")
    parser.add_argument("--cross-file")
    parser.add_argument("--operation")
    args = parser.parse_args()
    report = inspect_workspace(args.workspace, inspect_only=args.inspect, windows_acl=args.windows_acl,
                               ambient_temp=args.ambient_temp, cross_file=args.cross_file, operation=args.operation)
    print(json.dumps(report, ensure_ascii=True))
    sys.exit(0 if report["ok"] else 1)
