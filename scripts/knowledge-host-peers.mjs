import { existsSync, readFileSync, realpathSync } from 'node:fs';
import { registerHooks } from 'node:module';
import { join, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

// dsh-knowledge 0.5.4 imports three host packages that it declares as development
// dependencies. Bind an absent peer to the running host without changing the profile.
export function registerKnowledgeHostPeers(projectRoot, profileDirectory) {
  const manifest = join(profileDirectory, 'node_modules', 'dsh-knowledge', 'package.json');
  if (!existsSync(manifest)) return;
  const pkg = JSON.parse(readFileSync(manifest, 'utf8'));
  if (pkg.name !== 'dsh-knowledge' || pkg.version !== '0.5.4') return;

  const pluginDirectory = pathToFileURL(realpathSync(resolve(manifest, '..')) + sep).href;
  const host = join(projectRoot, 'vendor', 'deepseek-harness');
  const providers = new Map([
    ['@deepseek-ai/cordis', 'vendor/cordis'],
    ['@deepseek-ai/dsh-storage-domain', 'packages/storage/storage-domain'],
    ['@deepseek-ai/dsh-tools', 'packages/core/tools'],
  ]);
  return registerHooks({
    resolve(specifier, context, nextResolve) {
      if (!providers.has(specifier) || !context.parentURL?.startsWith(pluginDirectory)) {
        return nextResolve(specifier, context);
      }
      try {
        return nextResolve(specifier, context);
      } catch (error) {
        if (error.code !== 'ERR_MODULE_NOT_FOUND'
            || !error.message.startsWith(`Cannot find package '${specifier}' imported from`)) throw error;
        const manifest = join(host, providers.get(specifier), 'package.json');
        return nextResolve(specifier, { ...context, parentURL: pathToFileURL(manifest).href });
      }
    },
  });
}

if (process.env.CONTROL_CODE_ROOT && process.env.DSH_HOME) {
  registerKnowledgeHostPeers(process.env.CONTROL_CODE_ROOT, join(process.env.DSH_HOME, 'profiles', 'web'));
}
