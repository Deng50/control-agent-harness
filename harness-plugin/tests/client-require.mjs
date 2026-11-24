import assert from 'node:assert/strict';
import { statSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';

/** Supply the native module loader's require.asset contract in built-client DOM tests. */
export const clientRequire = createRequire(new URL('../package.json', import.meta.url));
clientRequire.asset = reference => {
  const match = /^plugins\/@control-engineering\/harness-bundle\/assets\/([\w.-]+)$/.exec(reference);
  assert.ok(match, `Unexpected client asset reference: ${reference}`);
  const override = process.env.CONTROL_TEST_CLIENT_BUNDLE;
  const file = override
    ? join(dirname(override), 'assets', match[1])
    : clientRequire.resolve('@control-engineering/harness-bundle/assets/' + match[1]);
  const asset = statSync(file);
  assert.ok(asset.isFile() && asset.size > 0, `Missing built client asset: ${file}`);
  return '/' + reference;
};
