// Render trixium.html to numbered JPEG frames with headless Chrome (one browser per worker).
// Usage: npm i puppeteer-core && CHROME=/usr/bin/google-chrome WORKERS=8 node render.js
// Then:  ffmpeg -framerate 30 -i frames/f_%05d.jpg -i music.wav -c:v libx264 -preset slow -crf 17 \
//          -pix_fmt yuv420p -c:a aac -b:a 192k -shortest -movflags +faststart trixium.mp4
// Frames already present are skipped, so an interrupted render resumes.
const puppeteer = require('puppeteer-core'), fs = require('fs'), path = require('path');
const FPS = 30, DUR = 131, N = FPS * DUR, W = +process.env.WORKERS || 8;
const CHROME = process.env.CHROME || '/usr/bin/google-chrome';
const OUT = path.join(__dirname, 'frames');
async function worker(w) {
  const b = await puppeteer.launch({executablePath: CHROME, headless: 'new', protocolTimeout: 600000,
    args: ['--no-sandbox', '--allow-file-access-from-files', '--disable-background-timer-throttling',
           '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows']});
  const p = await b.newPage(); await p.setViewport({width: 1920, height: 1080});
  p.on('pageerror', e => console.log('ERR', e.message));
  await p.goto('file://' + path.join(__dirname, 'trixium.html'), {waitUntil: 'networkidle0'});
  await p.evaluate(() => { document.body.classList.add('capture'); return window.__ready; });
  for (let i = w; i < N; i += W) {
    const f = path.join(OUT, `f_${String(i).padStart(5, '0')}.jpg`);
    if (fs.existsSync(f) && fs.statSync(f).size > 0) continue;
    await p.evaluate(t => renderAt(t), i / FPS);
    await p.screenshot({path: f, type: 'jpeg', quality: 95, clip: {x: 0, y: 0, width: 1920, height: 1080}});
    if (i % 300 === 0) console.log('frame', i);
  }
  await b.close();
}
fs.mkdirSync(OUT, {recursive: true});
Promise.all([...Array(W)].map((_, w) => worker(w))).then(() => console.log('frames done'), e => { console.log('FAIL', e); process.exit(1); });
