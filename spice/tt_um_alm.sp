* ==============================================================================
* LVS-MATCHED SPICE NETLIST FOR TINY TAPEOUT MACRO: tt_um_alm
* Tile Size: 161.000 um x 111.520 um (2x2 Sky130 Mixed-Signal Tile)
* ==============================================================================
.subckt ALM_DIFF_SYNAPSE_PAIR vin_p vin_n v_minus r_p_val=35714.28 r_n_val=35714.28 r_wire=0.40
R_GPOS vin_n n_mid_p {r_p_val}
R_WPOS n_mid_p v_minus {r_wire}
R_GNEG vin_p n_mid_n {r_n_val}
R_WNEG n_mid_n v_minus {r_wire}
.ends ALM_DIFF_SYNAPSE_PAIR

.subckt ALM_COLUMN_TIA_TRIM_SKIP v_minus v_skip_in v_out i_trim=0.0 vos=0.0 r_skip=10000.0
R_SKIP v_skip_in v_minus {r_skip}
I_TRIM v_minus 0 {i_trim}
R_FB v_minus v_out 10000.0
V_OS v_plus 0 {vos}
E_AMP v_out 0 v_plus v_minus 2000.0
.ends ALM_COLUMN_TIA_TRIM_SKIP

.subckt tt_um_alm VPWR VGND clk rst_n ena ua_0 ua_1 ua_2 ua_3 ua_4 ua_5
XSYN_R0_C0 vin_p_0 vin_n_0 v_minus_c0 ALM_DIFF_SYNAPSE_PAIR r_p_val=37234.0426 r_n_val=34313.7255 r_wire=3.4000
XSYN_R1_C0 vin_p_1 vin_n_1 v_minus_c0 ALM_DIFF_SYNAPSE_PAIR r_p_val=19516.7286 r_n_val=210000.0000 r_wire=3.2000
XSYN_R2_C0 vin_p_2 vin_n_2 v_minus_c0 ALM_DIFF_SYNAPSE_PAIR r_p_val=71917.8082 r_n_val=23755.6561 r_wire=3.0000
XSYN_R3_C0 vin_p_3 vin_n_3 v_minus_c0 ALM_DIFF_SYNAPSE_PAIR r_p_val=99056.6038 r_n_val=21784.2324 r_wire=2.8000
XSYN_R4_C0 vin_p_4 vin_n_4 v_minus_c0 ALM_DIFF_SYNAPSE_PAIR r_p_val=40697.6744 r_n_val=31818.1818 r_wire=2.6000
XSYN_R5_C0 vin_p_5 vin_n_5 v_minus_c0 ALM_DIFF_SYNAPSE_PAIR r_p_val=28378.3784 r_n_val=48165.1376 r_wire=2.4000
XSYN_R6_C0 vin_p_6 vin_n_6 v_minus_c0 ALM_DIFF_SYNAPSE_PAIR r_p_val=20114.9425 r_n_val=159090.9091 r_wire=2.2000
XSYN_R7_C0 vin_p_7 vin_n_7 v_minus_c0 ALM_DIFF_SYNAPSE_PAIR r_p_val=61764.7059 r_n_val=25119.6172 r_wire=2.0000
XSYN_R8_C0 vin_p_8 vin_n_8 v_minus_c0 ALM_DIFF_SYNAPSE_PAIR r_p_val=29005.5249 r_n_val=46460.1770 r_wire=1.8000
XSYN_R9_C0 vin_p_9 vin_n_9 v_minus_c0 ALM_DIFF_SYNAPSE_PAIR r_p_val=22532.1888 r_n_val=86065.5738 r_wire=1.6000
XSYN_R10_C0 vin_p_10 vin_n_10 v_minus_c0 ALM_DIFF_SYNAPSE_PAIR r_p_val=42000.0000 r_n_val=31065.0888 r_wire=1.4000
XSYN_R11_C0 vin_p_11 vin_n_11 v_minus_c0 ALM_DIFF_SYNAPSE_PAIR r_p_val=31818.1818 r_n_val=40697.6744 r_wire=1.2000
XSYN_R12_C0 vin_p_12 vin_n_12 v_minus_c0 ALM_DIFF_SYNAPSE_PAIR r_p_val=25609.7561 r_n_val=58988.7640 r_wire=1.0000
XSYN_R13_C0 vin_p_13 vin_n_13 v_minus_c0 ALM_DIFF_SYNAPSE_PAIR r_p_val=116666.6667 r_n_val=21084.3373 r_wire=0.8000
XSYN_R14_C0 vin_p_14 vin_n_14 v_minus_c0 ALM_DIFF_SYNAPSE_PAIR r_p_val=61764.7059 r_n_val=25119.6172 r_wire=0.6000
XSYN_R15_C0 vin_p_15 vin_n_15 v_minus_c0 ALM_DIFF_SYNAPSE_PAIR r_p_val=68181.8182 r_n_val=24193.5484 r_wire=0.4000
XTIA_C0 v_minus_c0 ua_0 ua_1 ALM_COLUMN_TIA_TRIM_SKIP i_trim=0.0 vos=0.0 r_skip=10000.0
R_RC_INT ua_1 ua_3 25000.0
C_MIM_2PF ua_3 VGND 2.0e-12
.ends tt_um_alm

* --- Standalone ngspice Verification Testbench for Tiny Tapeout Tile ---
.option temp=26.85 tnom=26.85 gmin=1e-18 abstol=1e-18 vntol=1e-12
V_PWR VPWR 0 1.8
V_GND VGND 0 0.0
V_SKIP ua_0 0 -0.18
X_DUT VPWR VGND 0 1 1 ua_0 ua_1 ua_2 ua_3 ua_4 ua_5 tt_um_alm
V_INP_0 X_DUT.vin_p_0 0 -3.00000000e-01
V_INN_0 X_DUT.vin_n_0 0 3.00000000e-01
V_INP_1 X_DUT.vin_p_1 0 -2.60000000e-01
V_INN_1 X_DUT.vin_n_1 0 2.60000000e-01
V_INP_2 X_DUT.vin_p_2 0 -2.20000000e-01
V_INN_2 X_DUT.vin_n_2 0 2.20000000e-01
V_INP_3 X_DUT.vin_p_3 0 -1.80000000e-01
V_INN_3 X_DUT.vin_n_3 0 1.80000000e-01
V_INP_4 X_DUT.vin_p_4 0 -1.40000000e-01
V_INN_4 X_DUT.vin_n_4 0 1.40000000e-01
V_INP_5 X_DUT.vin_p_5 0 -1.00000000e-01
V_INN_5 X_DUT.vin_n_5 0 1.00000000e-01
V_INP_6 X_DUT.vin_p_6 0 -6.00000000e-02
V_INN_6 X_DUT.vin_n_6 0 6.00000000e-02
V_INP_7 X_DUT.vin_p_7 0 -2.00000000e-02
V_INN_7 X_DUT.vin_n_7 0 2.00000000e-02
V_INP_8 X_DUT.vin_p_8 0 2.00000000e-02
V_INN_8 X_DUT.vin_n_8 0 -2.00000000e-02
V_INP_9 X_DUT.vin_p_9 0 6.00000000e-02
V_INN_9 X_DUT.vin_n_9 0 -6.00000000e-02
V_INP_10 X_DUT.vin_p_10 0 1.00000000e-01
V_INN_10 X_DUT.vin_n_10 0 -1.00000000e-01
V_INP_11 X_DUT.vin_p_11 0 1.40000000e-01
V_INN_11 X_DUT.vin_n_11 0 -1.40000000e-01
V_INP_12 X_DUT.vin_p_12 0 1.80000000e-01
V_INN_12 X_DUT.vin_n_12 0 -1.80000000e-01
V_INP_13 X_DUT.vin_p_13 0 2.20000000e-01
V_INN_13 X_DUT.vin_n_13 0 -2.20000000e-01
V_INP_14 X_DUT.vin_p_14 0 2.60000000e-01
V_INN_14 X_DUT.vin_n_14 0 -2.60000000e-01
V_INP_15 X_DUT.vin_p_15 0 3.00000000e-01
V_INN_15 X_DUT.vin_n_15 0 -3.00000000e-01
.control
op
print v(ua_1) v(ua_3)
quit
.endc
.end
