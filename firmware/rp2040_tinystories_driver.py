#!/usr/bin/env python3
"""
Tiny Tapeout Demo Board RP2040 Microcontroller Host Driver for `tt_um_alm`:
  1. Pulses `ui_in[3]` (`cal_auto_zero_en`) on power-up to null all 16 column TIA offsets
     using the on-chip 8-bit Current-Steering Auto-Zero Trim DACs.
  2. Streams the 6-bit RRAM/MDAC conductances of the 2-layer Hardware-Native Analog LLM
     (`R3_GQA1_RCAttnBypass_BiCtx112`) over SPI (`ui_in[0..2]`) into the `tt_um_alm` ASIC tile.
  3. Clocks tokens at 1 MHz (`clk`), sampling the 2.0 pF S&H KV-Cache and 1-pole RC
     integrator capacitors (`ui_in[6]`), and reads the digitized output logits (`uo_out[7:2]`).
"""

import time


class TinyTapeoutAnalogLLMBoard:
    def __init__(self, spi_bus=None, gpio_pins=None):
        self.spi = spi_bus
        self.gpio = gpio_pins

    def calibrate_on_chip_auto_zero_trim_dacs(self) -> None:
        """Grounds wordlines (`V_in = 0V`) and strobes `ui_in[3]` (`cal_auto_zero_en`) for 16 us."""
        if self.gpio is not None:
            self.gpio["cal_auto_zero_en"].value(1)
            time.sleep_us(16)
            self.gpio["cal_auto_zero_en"].value(0)

    def program_16x16_conductance_tile(self, g_pos_codes_6bit, g_neg_codes_6bit, trim_codes_8bit) -> None:
        """Streams 256 x 12-bit differential conductance codes + 16 x 8-bit Trim DAC codes over SPI."""
        payload = bytearray()
        for r in range(16):
            for c in range(16):
                code12 = ((int(g_pos_codes_6bit[r][c]) & 0x3F) << 6) | (int(g_neg_codes_6bit[r][c]) & 0x3F)
                payload.append((code12 >> 8) & 0xFF)
                payload.append(code12 & 0xFF)
        for c in range(16):
            payload.append(int(trim_codes_8bit[c]) & 0xFF)
        if self.spi is not None:
            self.gpio["spi_cs_n"].value(0)
            self.spi.write(payload)
            self.gpio["spi_cs_n"].value(1)
