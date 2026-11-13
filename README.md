![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg)

# Hardware-Native Analog Language Model (`tt_um_alm`)

130nm Mixed-Signal / Analog LLM Compute Tile for [Tiny Tapeout](https://tinytapeout.com) (`1x1` tile, `161.0 um x 111.52 um`):

- **Differential 6-Bit Conductance Crossbar (`G+ / G-`) + Direct KCL Virtual-Ground Skip (`R_skip = 10 kOhm`)**
- **8-Bit Current-Steering Auto-Zero Trim DACs (`±60 uA`)** for single-pass offset & mismatch nulling
- **Switched-Capacitor `2.0 pF` Sample-and-Hold (S&H) KV-Cache & 1-Pole RC Leaky Integrator Bank**
- **Four-Quadrant Gilbert Multiplier Array, Emitter-Coupled Exponential Softmax & On-Tile 8-Bit SAR ADC**

See [docs/info.md](docs/info.md) for the full hardware datasheet and pinout.
