// Capture stills of trixium.html at given times, for review: node stills.js 7.6 50 137.8
const puppeteer = require('puppeteer-core'), fs = require('fs'), path = require('path');
const CHROME = process.env.CHROME || '/usr/bin/google-chrome', OUT = path.join(__dirname, 'stills');
(async () => {
  fs.mkdirSync(OUT, {recursive: true});
  const b = await puppeteer.launch({executablePath: CHROME, headless: 'new', args: ['--no-sandbox', '--allow-file-access-from-files']});
  const p = await b.newPage(); await p.setViewport({width: 1920, height: 1080});
  p.on('pageerror', e => console.log('ERR', e.message));
  await p.goto('file://' + path.join(__dirname, 'trixium.html'), {waitUntil: 'networkidle0'});
  await p.evaluate(() => { document.body.classList.add('capture'); return window.__ready; });
  for (const t of process.argv.slice(2).map(Number)) {
    await p.evaluate(t => renderAt(t), t);
    await p.screenshot({path: path.join(OUT, `t_${t.toFixed(1).padStart(6, '0')}.png`), clip: {x: 0, y: 0, width: 1920, height: 1080}});
  }
  await b.close();
})();
