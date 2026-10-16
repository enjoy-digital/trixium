# Diagrams

The SVGs are directly editable in Inkscape or as XML. Text remains editable and the existing
LiteX and Enjoy-Digital logos use the presentation's dark-background variants and are embedded,
so the diagrams display without external assets.

| SVG | Content |
| --- | --- |
| [architecture.svg](architecture.svg) | Common LiteX system and TriXium core |
| [board-fk33.svg](board-fk33.svg) | FK33: direct HBM2 engine ports |
| [board-xcu1525.svg](board-xcu1525.svg) | XCU1525: four LiteDRAM DDR4 controllers |
| [board-ypcb.svg](board-ypcb.svg) | YPCB: two LiteDRAM DDR3 controllers |
| [experimentation.svg](experimentation.svg) | Simulation, measurement and upstream feedback |
| [why-not-tpu.svg](why-not-tpu.svg) | Weight reuse versus streaming ternary computation |

The style follows the supplied [LiteX SoC](https://user-images.githubusercontent.com/1450143/103343266-f8cc9a00-4a8b-11eb-9444-f02e1522a490.png)
and [HBM2](https://user-images.githubusercontent.com/1450143/124902018-d4c1b680-dfe2-11eb-89c4-8b498605c34d.png)
references: clipped-corner blocks, nested system boundaries and orthogonal, labelled arrows.
Colours follow the [video/presentation](../illustration/trixium.html): dark green backgrounds,
teal TriXium blocks, blue LiteX components and gold memory. Teal arrows show data movement;
muted arrows show control and CPU access. The three board views share a layout and show the
current hardware defaults; compare them with the corresponding targets.

The glow (soft halos behind the teal, blue and gold blocks, a light bloom on the data arrows and
titles, the video's `GLOW` look) is added by [glow.py](glow.py): re-run `python3
doc/images/glow.py` after editing a diagram (it removes the previous pass first; `GLOW` scales
it, 0.5 as in the video).

When editing, retain the matching board layouts, presentation palette, embedded logos, titles
and descriptions. Check legibility and arrow routing at approximately 900 px wide. Arrow tips
meet block edges; use a junction dot where routes branch, without an arrowhead at the junction.
