/** Save a user-supplied secret through Harness, with hidden terminal input only. */
import { connectHarness } from './harness-rpc.mjs';

if (!process.stdin.isTTY) throw new Error('Run in an interactive terminal; credentials are never command-line arguments');
const rpc = await connectHarness();
process.stdin.setRawMode(true);
process.stdin.resume();
console.log('Enter DeepSeek API credential (hidden input):');
let secret = '';
try {
  secret = await new Promise((resolve,reject) => {
    let value = '';
    const onData = data => {
      for (const ch of data.toString('utf8')) {
        if (ch === '\u0003') { process.stdin.off('data',onData); reject(new Error('Credential input cancelled')); return; }
        if (ch === '\r' || ch === '\n') { process.stdin.off('data',onData); resolve(value); value = ''; return; }
        if (ch === '\u007f' || ch === '\b') value = value.slice(0,-1);
        else value += ch;
      }
    };
    process.stdin.on('data',onData);
  });
  if (!secret) throw new Error('Credential is empty');
  await rpc('credentials/set', {ref:'DEEPSEEK_API_KEY',value:secret});
  const facts = await rpc('credentials/describe', {refs:['DEEPSEEK_API_KEY']});
  console.log(JSON.stringify({credential:facts.DEEPSEEK_API_KEY}));
} finally {
  secret = '';
  process.stdin.setRawMode(false);
  process.stdin.pause();
}
