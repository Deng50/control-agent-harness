/** Read-only admission check. Python also validates the exact bytes, training
 * contract and numerical state before restoring an optimizer. */
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile, realpath, stat } from 'node:fs/promises';
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';

const generation = /^checkpoint-[0-9a-f]{32}$/;
async function exactFile(path) {
  if (relative(path,await realpath(path)) || !(await stat(path)).isFile()) {
    throw new Error('Checkpoint file or metadata redirects outside its recorded path');
  }
  return path;
}
const metadata = async path => JSON.parse(await readFile(await exactFile(path),'utf8'));

export async function pinCheckpoint(record, selected, useRecoveredHead = false) {
  const path = resolve(record.workspace, selected), parts = relative(record.workspace,path).split(sep);
  if (parts[0] === '..' || isAbsolute(parts[0])) throw new Error('Checkpoint must be inside this workspace');
  if (parts.length === 3 && parts[0] === '.staging' && record.stageFailure?.stage_id === parts[1]) {
    const archive = join(record.workspace,'failed',parts[1]);
    let receipt;
    try { receipt = await metadata(join(archive,'interrupted-stage-recovery.json')); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    if (receipt?.stage_id === parts[1] && receipt.owner_verification === 'os_confirmed_process_exit') {
      return pinCheckpoint(record,join(archive,parts[2]),useRecoveredHead);
    }
  }
  if (parts.length !== 3 || !['artifacts','failed','failures'].includes(parts[0])) {
    throw new Error('请先用“校验未发布检查点”核验生产进程已退出，并归档已提交的代号，再选择续训。');
  }
  const root = dirname(path), stem = basename(path,'.pt');
  if (useRecoveredHead && parts[0] === 'failed') {
    let receipt;
    try { receipt = await metadata(join(root,'interrupted-stage-recovery.json')); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    if (receipt?.stage_id === parts[1] && typeof receipt.checkpoint === 'string' &&
        receipt.owner_verification === 'os_confirmed_process_exit') {
      const pinned = await pinCheckpoint(record,join(root,basename(receipt.checkpoint)));
      if (pinned.resume_checkpoint_sha256 !== receipt.checkpoint_sha256) throw new Error('Archived recovery selection changed');
      return pinned;
    }
  }
  const legacy = basename(path) === 'latest.pt';
  if (!legacy && (!generation.test(stem) || basename(path) !== stem + '.pt')) throw new Error('Invalid checkpoint generation');
  // A packaging failure can occur after its inner numerical model was
  // published. The recorded checkpoint then belongs to the inner stage.
  if (!record.checkpoint && record.stageFailure && parts[1] !== record.stageFailure.stage_id) throw new Error('Checkpoint is from a different failed stage');
  if (record.checkpoint && basename(dirname(resolve(record.workspace,record.checkpoint))) !== parts[1]) {
    throw new Error('Checkpoint is from a different attempt stage');
  }
  let head;
  try { head = await metadata(join(root,'checkpoint-head.json')); }
  catch (error) { if (error.code !== 'ENOENT' || !legacy) throw error; }
  if (head && (head.format_version !== 2 || !generation.test(head.current) ||
    (head.previous !== null && (!generation.test(head.previous) || head.current === head.previous)))) throw new Error('Invalid checkpoint commit pointer');
  if (!legacy && ![head.current,head.previous].includes(stem)) throw new Error('Uncommitted checkpoint generation');
  const meta = await metadata(join(root, legacy ? 'latest.checkpoint.json' : stem + '.json'));
  if (meta.format_version !== 1 || !/^[a-f0-9]{64}$/.test(meta.sha256) || (!legacy && meta.generation !== stem)
      || (legacy && meta.generation && !head)) {
    throw new Error('Invalid checkpoint sidecar');
  }
  if (legacy && head) {
    const current = await metadata(join(root,head.current+'.json'));
    if (current.generation !== head.current || current.sha256 !== meta.sha256) throw new Error('Stale latest.pt checkpoint snapshot');
    return pinCheckpoint(record,join(root,head.current+'.pt'));
  }
  const hash = createHash('sha256');
  for await (const bytes of createReadStream(await exactFile(path))) hash.update(bytes);
  const digest = hash.digest('hex');
  const known = record.checkpoints?.find(row => basename(row.path) === basename(path) && basename(dirname(row.path)) === parts[1]);
  if (digest !== meta.sha256 || (known?.sha256 && digest !== known.sha256)) throw new Error('Checkpoint SHA-256 differs from committed bytes');
  return {resume_checkpoint:path,resume_checkpoint_sha256:digest};
}
