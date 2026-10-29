# Router cluster feasibility, CPU only

Source: router_cluster_compile_only.py SHA256 018a96eca3f833280b1df05d02cf5ffab57b5233496ab10b20b33d3a45556c10.
Pinned existing CPU-only Modal compile_only function; zero GPU allocations, no judge checks.

Extracted existing expert_cluster_epilogue.py reduction into an offline SM90 compiler probe. Original auto-layout prototype did not compile. Explicit separate source/output ranges fix conflicting auto encodings; multi-axis expand_dims must be sequential; constructing nested-CGA layout constants outside the JIT avoids a nested-list layout equality AttributeError in this pinned compiler. Original GPU probe is unchanged and remains unvalidated.

Final raw evidence: router_cluster_compile_globals.out. Two-CTA R32/T64 FP32 partial reduction:
- Generic shared store: compile PASS, 8192 shared bytes, 0 st.async, 4 static barrier.cluster instructions, 0 shfl.
- Register-source async store: compile PASS, 8200 shared bytes, 8 static st.async, 4 static barrier.cluster instructions, 0 shfl.

This establishes availability of the intended Gluon cluster lowering on SM90 for this isolated epilogue. It is not a complete router, numerical proof, timing proof, or eight-CTA proof. No basis to assert Triton is capped and switch to C++. Next source work must extend the layout/sum contract to actual cluster size4/8, then integrate the BF16-hi/lo FP32-accumulating mainloop with BT64 while preserving full precision, shared lifetime and fresh-input rules. Existing cluster reduction performance losses cannot be dismissed from this compiler success.

Canonical cp18 and all checkpoints unchanged. No pending judge jobs.

## Cluster sizes 4/8 compiler extension
router_cluster_sizes_compile_only.py and .out extend the isolated epilogue to2/4/8CTAs. Allsix generic/async variants compile onSM90: shared8192/8200B respectively,0shfl,4staticclusterbarrier instructions, async8staticst.async. NVMMADistributedLayout ininstalledTriton3.8exposes cga_layout. StillnotGPU numerical/timing orfullrouter proof. Nextboundedsourcework: router_cluster_bt64.py MMA2 hi/lo FP32acc mainloop plus localcp.asyncring andsameclusterepilogue; no newC++ orGPUallocation.
