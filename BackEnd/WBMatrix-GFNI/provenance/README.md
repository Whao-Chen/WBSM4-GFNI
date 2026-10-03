# Release provenance

The library core is the validated **2026-08-21 slim snapshot** of WBMatrix 2.0.
It is published here as WBMatrix-GFNI. All 11 files under include/ and src/ are
unchanged from that snapshot; source-manifest.tsv records the exact byte lengths
and SHA-256 values.

## Reference scope

| Item | Original WBMatrix | WBMatrix-GFNI core |
|---|---:|---:|
| All library source bytes | 171088 | 361790 |
| GCC 9.4.0 -O2 static archive bytes | 92220 | 163218 |
| Original public API symbols | 227 | 227 |

The reference optimized archive SHA-256 is
2ded23c4f4dfc757c4189f6efc618949339a8fd84a06c6d0a091d1b0be0dc7a4.
The archive's name is libWBMatrixOpt.a.
The source totals include public and internal headers, not only C translation units.
Repository scaffolding, tests, licenses, manifests and documentation are outside
the 361790-byte library-core measurement.

The size check uses GCC 9.4.0, -O2 -DNDEBUG and the explicit ISA flags in
CMakeLists.txt. No LTO or PGO is enabled. Archive identity can depend on compiler
and binutils versions; source hashes are the authoritative snapshot identity.
expected-defined-symbols.txt preserves the reference archive's nm output,
including its non-public implementation symbols.

## Upstream and modifications

Base: https://github.com/Nexus-TYF/WBMatrix
Commit: 0043eb53e4ee1d7641c06c0155b5f2357bce18e9
License: Apache-2.0 (LICENSE and NOTICE).

- include/WBMatrix/WBMatrix.h, random.h and structure.h retain the upstream bytes.
- src/WBMatrix/WBMatrix.c retains unchanged public routines and data while selected
  definitions have moved to the dedicated optimized implementation files.
- src/WBMatrix/random.c preserves the compatible generator with release-specific
  changes; it is not a new cryptographic entropy source.
- matrix_mul.c, matrix_vector.c and affine.c contain the selected GFNI/SIMD routes.
- hbup.c implements the HBUP invertible-pair generator.
- inverse_m8.c implements fixed-exponent inversion for invertible M8 inputs.
- internal.h declares the internal kernels and prepared representations.
  Internal historical identifiers are preserved to keep the validated bytes.

The tests were copied from the same release's build-support directory:
smoke.c, primitive_driver.c, test_release_hbup.c and whole_archive_main.c.
The standalone build system, example, source-verification tool, citation metadata
and repository documentation were added for this public package.

Legacy printing helpers retain the snapshot's format strings and produce GCC
format warnings on this platform. They are preserved with the reference bytes;
the arithmetic validation and archive-identity checks are reported separately.

## Reproduction

Run the README build and CTest commands on supported hardware, then:

    python3 tools/verify_snapshot.py --archive build/libWBMatrixOpt.a

This verifies the immutable core and the reference archive size. The upstream
baseline, white-box applications and unpublished manuscripts are not required
for the standalone library build.
