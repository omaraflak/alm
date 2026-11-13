/*
 * Copyright (c) 2026 Omar Aflak
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

// Tiny Tapeout 1x1 Macro Wrapper for `tt_um_alm`
// Hardware-Native Analog SPICE-KCL LLM Tile (4:1 GQA S&H + 32-Ch RC-Gilbert Bypass + BiCtx-Gilbert FFN)
module tt_um_alm (
    input  wire       VGND,
    input  wire       VDPWR,    // 1.8v power supply
    input  wire [7:0] ui_in,    // Dedicated digital inputs (SPI SCLK, MOSI, CS_N, CAL_EN, TILE_OP[3:0])
    output wire [7:0] uo_out,   // Dedicated digital outputs (SPI MISO, SAR_ADC_DONE, TOKEN_MSB[5:0])
    input  wire [7:0] uio_in,   // Bidirectional IOs: Input path (Wordline DAC / Column address bus)
    output wire [7:0] uio_out,  // Bidirectional IOs: Output path (8-bit SAR ADC readout bus)
    output wire [7:0] uio_oe,   // Bidirectional IOs: Enable path (1=output, 0=input)
    input  wire       ena,      // High when design is selected
    input  wire       clk,      // 1 MHz token & switched-capacitor S&H clock
    input  wire       rst_n     // Active-low capacitor & state reset
);

endmodule
