<!---
This file is used to generate your project datasheet on tinytapeout.com.
--->

## How it works

`tt_um_alm` is a hardware-native mixed-signal / analog Language Model compute tile designed in 130nm technology (`1x1` Tiny Tapeout tile, `161.0 um x 111.52 um`). Instead of converting analog currents back and forth to digital after every matrix multiplication, this core implements the complete Hardware-Native Analog LLM signal path (`R3_GQA1_RCAttnBypass_BiCtx112`) directly in continuous voltages and currents inside a single `1x1` tile, bridged to the digital bus via on-tile wordline DACs and an 8-bit SAR readout ADC:

1. **Differential 6-Bit Programmable Conductance Crossbar (`G+ / G-`) + Direct KCL Virtual-Ground Skip (`R_skip = 10 kOhm`)**:
   - Computes signed current-mode matrix-vector products (`I_col = sum_i V_in[i] * (G+[i,j] - G-[i,j]) + V_skip[j] / R_skip + I_trim[j]`) directly into the virtual-ground node (`v_minus`) of folded-cascode CMOS transimpedance amplifiers (TIAs).
   - Each column includes an **8-Bit Current-Steering Auto-Zero Trim DAC** (`±60 uA`) that nulls out static op-amp offset (`V_os`) and device mismatch in a single pass when `ui_in[3]` (`cal_auto_zero_en`) is strobed with `V_in = 0 V`.
2. **Switched-Capacitor `2.0 pF` Sample-and-Hold (S&H) KV-Cache & 1-Pole RC Leaky Integrator Bank**:
   - Implements the 4:1 Grouped-Query Attention (GQA) sample-and-hold capacitor slots alongside a **1-Pole Switched-Capacitor RC Leaky Integrator** (`V_rc(t) = (1 - alpha) * V_rc(t-1) + alpha * V_in(t)`), which physically low-pass filters `300K` thermal noise while providing multi-timescale contextual memory across tokens using `3.53x` fewer capacitors than a standard Transformer.
3. **Four-Quadrant Gilbert Multiplier Array, Emitter-Coupled Softmax & On-Tile 8-Bit SAR ADC**:
   - Computes dynamic attention dot-products and **Multi-Timescale RC-BiCtx-Gilbert Bilinear FFN gating** (`V_lin * tanh(V_gate / V_lin) * tanh(V_ctx / V_lin)`), normalized by an emitter-coupled exponential Softmax current-steering array and digitized by the on-tile 8-bit SAR ADC onto `uo_out[7:2]` and `uio_out[1:0]`.

## How to test

1. **Auto-Zero Calibration**: Drive `ui_in[3] = 1` (`cal_auto_zero_en`) for 16 clock cycles at startup with inputs zeroed to null all column TIA offsets via the on-chip 8-bit Trim DACs.
2. **Streaming TinyStories Inference from the Onboard RP2040**: Run `firmware/rp2040_tinystories_driver.py` on the Tiny Tapeout demo board's RP2040 microcontroller to stream the calibrated 2-layer TinyStories checkpoint (`alm_calibrated_die_v2.npz`) through the physical analog tile at `1 MHz` and print generated stories over USB serial.

## External hardware

Uses only the standard Tiny Tapeout Demo Board RP2040 microcontroller over the included digital SPI and control lines (`ui_in[7:0]`, `uo_out[7:0]`, `uio[7:0]`) — zero external components or extra analog pins required.
