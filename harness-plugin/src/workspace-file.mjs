import { lstat, open, realpath } from 'node:fs/promises';
import { basename, dirname, isAbsolute, join, relative } from 'node:path';

export const maxPythonBytes=50*1024*1024;

export function workspacePythonName(workspace, file) {
  if (typeof workspace!=='string' || !isAbsolute(workspace) || typeof file!=='string' || !isAbsolute(file) ||
      file.split(/[\\/]/).includes('..') || relative(workspace,dirname(file))!=='') return;
  const name=basename(file);
  if (name.length>3 && /\.py$/i.test(name) && !/[<>:"/\\|?*\u0000-\u001f]/u.test(name) &&
      !/^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])\./i.test(name)) return name;
}

// The caller pins the workspace. Read a bounded snapshot from one regular file handle.
export async function readWorkspaceBytes(workspace, name, maxBytes) {
  if (!name || name==='.' || name==='..' || name!==basename(name) || /[\\/:\0]/.test(name)) throw new Error('需要当前工作区内的单个文件名。');
  const path=join(workspace,name);
  const entry=await lstat(path,{bigint:true});
  if (!entry.isFile() || entry.isSymbolicLink() || entry.nlink!==1n || relative(path,await realpath(path))!=='') {
    throw new Error('需要当前工作区内的普通单链接文件。');
  }
  const handle=await open(path,'r');
  try {
    const stat=await handle.stat({bigint:true});
    if (stat.dev!==entry.dev || stat.ino!==entry.ino || !stat.isFile() || stat.nlink!==1n) throw new Error('工作区文件发生变化。');
    if (stat.size<1n || stat.size>BigInt(maxBytes)) throw new Error('文件为空或超过 '+maxBytes/(1024*1024)+' MiB。');
    const buffer=Buffer.alloc(Number(stat.size));
    let offset=0;
    while (offset<buffer.length) {
      const {bytesRead}=await handle.read(buffer,offset,buffer.length-offset,offset);
      if (!bytesRead) throw new Error('读取工作区文件时遇到意外文件结尾。');
      offset+=bytesRead;
    }
    const after=await handle.stat({bigint:true});
    if (after.size!==stat.size || after.mtimeNs!==stat.mtimeNs || after.nlink!==1n) throw new Error('工作区文件在读取中发生变化。');
    return buffer;
  } finally { await handle.close(); }
}
