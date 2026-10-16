# TriXium: the video

A 131-second, music-synced motion piece about this project (ternary LLMs on FPGA, open source
end to end with LiteX), made for Twitter/X: 1920x1080, 30 fps. At 2:11 it fits the standard X
limit (2:20).

Everything is generated from code: no video editor and no stock footage.

| file | what it is |
|---|---|
| `trixium.html` | The film: one Canvas 2D page with a deterministic `renderAt(t)`. Open it in a browser to play it with sound. Logos are embedded as data URIs. |
| `music.py` | Synthesizes the 120 BPM soundtrack (numpy/scipy), one continuous score; variants by name, `original` by default. Its bar grid is the film's cut grid. |
| `music.mp3` | The soundtrack the page plays (output of `music.py`). |
| `render.js` | Renders every frame with headless Chrome, for encoding to MP4. |
| `stills.js` | Captures stills at chosen times, for review. |

## Preview

Serve the folder (the page loads `music.mp3` next to it) and press play, or click the canvas:

    python3 -m http.server -d doc/illustration 8000   # then open http://localhost:8000/trixium.html

## Render the MP4

    cd doc/illustration
    python3 music.py music.wav
    npm i puppeteer-core
    WORKERS=6 node render.js                     # 3,930 frames into frames/, ~10 min
    ffmpeg -framerate 30 -i frames/f_%05d.jpg -i music.wav -c:v libx264 -preset slow -crf 17 \
      -pix_fmt yuv420p -profile:v high -level 4.1 -g 60 -c:a aac -b:a 192k -shortest \
      -movflags +faststart trixium.mp4

The full-quality MP4 is not committed (about 200 MB, most of it film grain). The README links
a 720p copy, `doc/images/trixium.mp4` (12 MB), encoded from the rendered frames with:

    ffmpeg -framerate 30 -i frames/f_%05d.jpg -i music.mp3 -vf "scale=1280:720,hqdn3d=4:3:6:4" \
      -c:v libx264 -preset slow -crf 26 -pix_fmt yuv420p -profile:v high -movflags +faststart \
      -c:a aac -b:a 128k -shortest ../images/trixium.mp4

The stills in `doc/images/video-*.jpg` are single frames (`frames/f_<t*30>.jpg`) at 1280x720,
JPEG quality 82; the poster `video-poster.jpg` is one at 1920x1080:

    convert frames/f_02220.jpg -resize 1280x720 -quality 82 ../images/video-27b-builds.jpg

| still | time (s) | frame |
|---|---|---|
| `video-intro-boards.jpg` | 4.8 | the three second-hand cards, their first lives and prices, the tagline |
| `video-poster.jpg` | 8.8 | the first " Paris" on the FK33 (1920x1080) |
| `video-ternary.jpg` | 21.0 | a ternary dot product: select, skip, negate, add, no per-weight multipliers |
| `video-fk33-soc.jpg` | 31.0 | FK33: the SoC, module by module, up to the vector unit |
| `video-exact.jpg` | 34.5 | the verification ladder |
| `video-climb.jpg` | 51.0 | the 1.7B climb, 2 to 101.9 tok/s |
| `video-upstream.jpg` | 59.0 | limits removed, upstream: the LiteX and LiteDRAM PRs |
| `video-models.jpg` | 64.0 | then 8B (30.3 tok/s), then 27B (4.03 tok/s) |
| `video-xcu1525-memory.jpg` | 68.5 | XCU1525: four LiteDRAM DDR4 controllers, 16 ports |
| `video-27b-builds.jpg` | 74.0 | the 27B bitstream by bitstream |
| `video-third-card.jpg` | 79.0 | a third card: the YPCB-00338, 1.7B 35.2 / 8B 8.11 tok/s |
| `video-another-computer.jpg` | 83.0 | against another computer: RTX 3080, M4 Pro GPU, YPCB-00338, per GB/s |
| `video-memory-wall.jpg` | 86.5 | the wall: the memory; faster tokens = a faster LiteDRAM |
| `video-memory-speeds.jpg` | 91.0 | memory speeds today: DDR4-1500, DDR3-1066, HBM2, the native DDR4 PHY |
| `video-portability.jpg` | 95.5 | one design, every LiteX vendor; any LiteX-Boards card with PCIe and enough memory |
| `video-smaller-homes.jpg` | 98.5 | same blocks, smaller homes: three measured boards, two projected |
| `video-everywhere.jpg` | 104.0 | intelligence everywhere: from the data centre to a sensor node |
| `video-whats-next.jpg` | 112.0 | what's next: reliable DDR3, faster DDR3/DDR4, all of the FK33's bandwidth, openXC7, FPGAs not ASICs, no PC needed |
| `video-open-source.jpg` | 118.0 | open source: BSD-2-Clause, the repository, custom work |
| `video-credits.jpg` | 124.0 | thanks |
| `video-end-card.jpg` | 129.0 | the end card: the repository, the contact, made in France |

## The glow

One constant at the top of the script, `GLOW` (1 = the first cut's heavy bloom, 0 = none),
scales the bloom pass, the coloured halos behind text and boxes, the light sprites and the film
grain. It is 0.4: under half the first cut's bloom and grain, so the text stays crisp.

## Music

The soundtrack is fully synthesized by `music.py`: one continuous 131 s score at 120 BPM, A minor,
with its sections on the film's chapters (a breakdown under the verification chapter, a drop
into the climb, half-time under the memory chapter, the final hit into the thanks at 120 s, a
quiet bed under the end card). The film uses the `original` score (synthwave: saw pads, a pulse arpeggio, a lead in the drops),
the default; `epic`, `warm`, `techno`, `cinematic` and `chiptune` share
its grid, hits and risers. To compare them in the page, render each into `music_variants/<name>.mp3`
(not committed) and open `trixium.html?music=<name>`.

## Structure

| time (s) | chapter |
|---|---|
| 0-6 | Opening: the three second-hand cards (the board photograph), their first lives and prices, "Ternary LLMs on FPGA, open source end to end with LiteX." |
| 6-14 | Intro: first " Paris" on the FPGA, "up to 27 billion parameters", "One FPGA. No GPU.", wordmark |
| 14-22 | 01 Ternary: why ternary fits an FPGA, the adder tree |
| 22-32 | 02 FK33: the SoC, a module-by-module tour |
| 32-38 | 03 Exact: the verification ladder, "Correct first", "Now make it fast" |
| 38-52 | 04 The climb: 1.7B from 2 to 101.9 tok/s, "A large increase." |
| 52-60 | 05 Upstream: the limits removed in LiteX and LiteDRAM, with the PRs |
| 60-66 | 06 Then 8B (30.3 tok/s), then the Ternary-Bonsai-2 27B (4.03 tok/s) |
| 66-76 | 07 XCU1525: one core, a second target (four LiteDRAM DDR4 controllers), the 27B bitstream by bitstream |
| 76-84 | 08 A third card: the YPCB-00338 (1.7B 35.2, 8B 8.11 tok/s), then against another computer (RTX 3080, M4 Pro GPU; per GB/s of memory bandwidth) |
| 84-92 | 09 The memory: the wall, "Faster tokens? A faster LiteDRAM.", memory speeds today |
| 92-96 | 10 Every vendor: one design, the LiteX vendors (only AMD measured), the easy-port line |
| 96-100 | 11 Smaller homes: three measured boards, two projected |
| 100-106 | 12 Intelligence everywhere: from the data centre to a sensor node |
| 106-114 | 13 What's next |
| 114-120 | 14 Open source (BSD-2-Clause, github.com/enjoy-digital/trixium), custom work, "Fork it. Port it. Make it yours." |
| 120-126 | Thanks (the final hit) |
| 126-131 | End card: the repository, "Questions, ports, custom work: florent@enjoy-digital.fr", made in France |

The film is the first cut re-edited (2026-10-16): the old scenes keep their own time base
(`SCENES` maps film time to each scene's), under one continuous score.

## Where the numbers come from

Every figure on screen is from this repository, and projections are labelled as projections:
the root `README.md`, `doc/timeline.md` (dates, per-build measurements, the 27B table), `doc/results.md`,
`doc/boards.md`, `doc/performance-next-steps.md` and `doc/simulation.md`; the `trixium_smi`
panel is the README's sample of the XCU1525 decoding the 27B. The first cut (2026-09-29) also
drew on per-build notes, board reports and positioning documents that the release cleanup
folded into `doc/timeline.md` and `doc/method.md` (the originals are in git history).
Two outside sources are also used: LiteDRAM PR #406, the native UltraScale+ DDR4 PHY (open at
the time of writing), and LiteNVMe's published throughput. The first cut was reviewed twice
for technical accuracy against these sources before it was committed; the 2026-10-09 update
brought the 8B, the 27B and the DDR-rate figures to the release state.

When a number changes, update it in `trixium.html`. The data arrays are `LADDER` (1.7B
climb), `V27` (27B bitstreams), `HOMES` (boards), `CASES` (use cases) and `CREDITS`.

## Third-party logos

The logos are used unaltered, on a light tile where needed, for attribution in the credits.
They remain the property of their owners:

- LiteX and Enjoy-Digital: `doc/images/litex-logo.png`, `doc/images/enjoy-digital-logo.png` (embedded
  in the page)
- SpinalHDL: `demo/public/spinalhdl-logo.svg` of an earlier demo (removed in the release cleanup, in
  git history), see its `demo/public/SPINALHDL-LOGO-SOURCE.txt`
- PrismML: the Hugging Face `prism-ml` organization avatar, which hosts the Ternary-Bonsai models
- Qwen (QwenLM), M-Labs (Migen), Verilator, ggml, AMD (Xilinx) and Anthropic: their GitHub
  organization avatars, fetched 2026-09-29

The film itself was written with Claude, under Enjoy-Digital's direction.
