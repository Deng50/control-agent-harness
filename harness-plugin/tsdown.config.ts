import { clientBundle } from '../vendor/deepseek-harness/packages/client/tsdown.client.ts';

// The upstream browser factory format keeps React and Cordis shared with Web.
const bundle = clientBundle(import.meta.dirname, ['src/index.mjs']);
export default bundle;
