#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause
#
# Area by cell-name prefix on a routed Vivado checkpoint. Migen emits one flat module, so Vivado's
# hierarchical report sees no units; the cell names keep the module names (main_<module>_...),
# which is what this groups by. LUT counts are LUT primitives (above the "Slice LUTs" figure by
# ~9%: a slice LUT holding two LUT5s counts twice here); FFs, RAMB36/18 and DSP48 as such.
#
#   vivado -mode batch -source software/lab/area_by_prefix.tcl -tclargs build/<name>/gateware/<top>_route.dcp area.txt
#
set dcp [lindex $argv 0]
set out [lindex $argv 1]
open_checkpoint $dcp
set prefixes {
  main_matvecengine_dnstream main_matvecengine_attentionstream
  main_matvecengine_pipeisland main_matvecengine_readerisland main_matvecengine_reader main_matvecengine_pipe
  main_matvecengine main_softmaxscratchpad main_siluscratchpad main_normscratchpad main_hadamardscratchpad
  main_convscratchpad main_activationquantize main_quantize main_projectionrestore main_restore
  main_scratchpad main_sequencer main_loader main_vectorunit main_engine_csr main_trixium
  main_add_ddr main_k7ddrphy main_litedramaxi2native main_axicrossing main_axiwritemerge main_axireadmerge
  main_axiresetshim main_engport main_s7pciephy main_pcie main_trixiumypcbsoc main_uart
}
set fh [open $out w]
puts $fh [format "%-40s %9s %9s %7s %7s %6s" prefix LUTs FFs RAMB36 RAMB18 DSPs]
foreach p $prefixes {
  set cells [get_cells -hierarchical -quiet -filter "NAME =~ ${p}* && IS_PRIMITIVE"]
  if {[llength $cells] == 0} { continue }
  set luts [llength [filter $cells {PRIMITIVE_GROUP == LUT || PRIMITIVE_SUBGROUP == LUT}]]
  set ffs  [llength [filter $cells {PRIMITIVE_GROUP == FLOP_LATCH || PRIMITIVE_GROUP == REGISTER}]]
  set b36  [llength [filter $cells {REF_NAME =~ RAMB36*}]]
  set b18  [llength [filter $cells {REF_NAME =~ RAMB18*}]]
  set dsp  [llength [filter $cells {REF_NAME =~ DSP48*}]]
  puts $fh [format "%-40s %9d %9d %7d %7d %6d" $p $luts $ffs $b36 $b18 $dsp]
}
set all [get_cells -hierarchical -quiet -filter {IS_PRIMITIVE}]
puts $fh [format "%-40s %9d %9d %7d %7d %6d" TOTAL \
  [llength [filter $all {PRIMITIVE_GROUP == LUT || PRIMITIVE_SUBGROUP == LUT}]] \
  [llength [filter $all {PRIMITIVE_GROUP == FLOP_LATCH || PRIMITIVE_GROUP == REGISTER}]] \
  [llength [filter $all {REF_NAME =~ RAMB36*}]] [llength [filter $all {REF_NAME =~ RAMB18*}]] \
  [llength [filter $all {REF_NAME =~ DSP48*}]]]
close $fh
