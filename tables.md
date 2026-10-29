# Appendix: benchmark tables

Stanford-ORB scores on 8 test views per object; lower is better for chamfer, normal, depth and LPIPS; `--` marks a missing run.

## A.1 Objects

| short name | DTC object |
|---|---|
| Birdhouse | `Birdhouse_Lighting001` |
| Cup | `Cup_B08TWHJ33Q_Tan_Lighting001` |
| Figurine | `Figurine_B08FYFNYP4_LionKing_Lighting001` |
| Gargoyle | `Gargoyle_B08SQMBDXY_HandsOnKnees_Lighting001` |
| Mallard | `Mallard_B082D168CK_MintGreen_Lighting001` |
| Mouse | `Mouse_B0CHNVBBLF_Honeycomb_1_Lighting001` |

## A.2 Baselines: all metrics, averaged over the objects

| method | objects | chamfer (x2e-3) | normal (1-cos) | depth SI-MSE | view PSNR-H | view PSNR-L | view SSIM | view LPIPS | light PSNR-H | light PSNR-L | light SSIM | light LPIPS | albedo PSNR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nvdiffrec (multi-view) | 6 | 0.0359 | 0.1798 | 0.0011 | 30.82 | 36.80 | 0.9775 | 0.0234 | 31.06 | 36.96 | 0.9776 | 0.0235 | 30.40 |
| RecGen (v1) | 6 | 0.1576 | 0.2992 | 0.0062 | 18.24 | 22.51 | 0.9165 | 0.0963 | 18.96 | 23.02 | 0.9274 | 0.0915 | 24.44 |
| RecGen (v2) | 6 | 0.0128 | 0.1610 | 0.0014 | 20.74 | 25.97 | 0.9444 | 0.0641 | 21.62 | 26.82 | 0.9454 | 0.0648 | 29.36 |
| SAM3D (v1) | 6 | 0.0525 | 0.2156 | 0.0020 | 19.98 | 24.18 | 0.9327 | 0.0678 | 20.38 | 24.43 | 0.9350 | 0.0664 | 26.01 |
| SAM3D (v2) | 6 | 0.0387 | 0.2199 | 0.0017 | 19.44 | 24.28 | 0.9370 | 0.0671 | 21.11 | 25.65 | 0.9412 | 0.0651 | 25.96 |
| TRELLIS (v1) | 6 | 0.0351 | 0.2184 | 0.0029 | 19.36 | 24.15 | 0.9370 | 0.0726 | 20.24 | 24.82 | 0.9386 | 0.0711 | 26.16 |
| TRELLIS (v2) | 6 | 0.0266 | 0.2036 | 0.0017 | 19.64 | 24.09 | 0.9392 | 0.0666 | 20.43 | 24.70 | 0.9394 | 0.0658 | 25.62 |
| TRELLIS.2 (v1) | 6 | 0.1052 | 0.2952 | 0.0048 | 19.46 | 24.22 | 0.9311 | 0.0736 | 19.98 | 24.75 | 0.9317 | 0.0729 | 26.35 |
| TRELLIS.2 (v2) | 6 | 0.0201 | 0.1763 | 0.0008 | 20.27 | 25.48 | 0.9417 | 0.0554 | 20.96 | 26.05 | 0.9419 | 0.0551 | 27.78 |

## A.3 Baselines, per object

**chamfer (x2e-3)**

| method | Birdhouse | Cup | Figurine | Gargoyle | Mallard | Mouse | mean |
|---|---|---|---|---|---|---|---|
| nvdiffrec (multi-view) | 0.0818 | 0.0518 | 0.0400 | 0.0297 | 0.0021 | 0.0101 | 0.0359 |
| RecGen (v1) | 0.5692 | 0.1458 | 0.0447 | 0.1211 | 0.0432 | 0.0217 | 0.1576 |
| RecGen (v2) | 0.0236 | 0.0047 | 0.0104 | 0.0223 | 0.0073 | 0.0084 | 0.0128 |
| SAM3D (v1) | 0.1161 | 0.0094 | 0.0320 | 0.1172 | 0.0250 | 0.0152 | 0.0525 |
| SAM3D (v2) | 0.0382 | 0.0057 | 0.0199 | 0.0661 | 0.0929 | 0.0096 | 0.0387 |
| TRELLIS (v1) | 0.0222 | 0.0652 | 0.0214 | 0.0441 | 0.0446 | 0.0129 | 0.0351 |
| TRELLIS (v2) | 0.0230 | 0.0073 | 0.0180 | 0.0468 | 0.0558 | 0.0088 | 0.0266 |
| TRELLIS.2 (v1) | 0.0285 | 0.5303 | 0.0187 | 0.0194 | 0.0252 | 0.0089 | 0.1052 |
| TRELLIS.2 (v2) | 0.0244 | 0.0029 | 0.0303 | 0.0147 | 0.0383 | 0.0098 | 0.0201 |

**normal (1-cos)**

| method | Birdhouse | Cup | Figurine | Gargoyle | Mallard | Mouse | mean |
|---|---|---|---|---|---|---|---|
| nvdiffrec (multi-view) | 0.1958 | 0.0682 | 0.2826 | 0.2522 | 0.0482 | 0.2321 | 0.1798 |
| RecGen (v1) | 0.4098 | 0.2933 | 0.3022 | 0.3009 | 0.2183 | 0.2710 | 0.2992 |
| RecGen (v2) | 0.1896 | 0.0611 | 0.2040 | 0.2095 | 0.0880 | 0.2141 | 0.1610 |
| SAM3D (v1) | 0.1864 | 0.0768 | 0.2833 | 0.2848 | 0.1605 | 0.3019 | 0.2156 |
| SAM3D (v2) | 0.1579 | 0.0393 | 0.2316 | 0.2658 | 0.3373 | 0.2878 | 0.2199 |
| TRELLIS (v1) | 0.1959 | 0.1797 | 0.2633 | 0.2404 | 0.1758 | 0.2553 | 0.2184 |
| TRELLIS (v2) | 0.1701 | 0.0884 | 0.2384 | 0.2419 | 0.2116 | 0.2709 | 0.2036 |
| TRELLIS.2 (v1) | 0.1870 | 0.6573 | 0.2618 | 0.2162 | 0.1551 | 0.2938 | 0.2952 |
| TRELLIS.2 (v2) | 0.1400 | 0.0245 | 0.2176 | 0.2006 | 0.1941 | 0.2810 | 0.1763 |

**depth SI-MSE**

| method | Birdhouse | Cup | Figurine | Gargoyle | Mallard | Mouse | mean |
|---|---|---|---|---|---|---|---|
| nvdiffrec (multi-view) | 0.0023 | 0.0000 | 0.0011 | 0.0029 | 0.0002 | 0.0000 | 0.0011 |
| RecGen (v1) | 0.0257 | 0.0049 | 0.0023 | 0.0011 | 0.0021 | 0.0011 | 0.0062 |
| RecGen (v2) | 0.0058 | 0.0003 | 0.0004 | 0.0013 | 0.0005 | 0.0001 | 0.0014 |
| SAM3D (v1) | 0.0029 | 0.0009 | 0.0024 | 0.0035 | 0.0013 | 0.0012 | 0.0020 |
| SAM3D (v2) | 0.0014 | 0.0004 | 0.0007 | 0.0029 | 0.0049 | 0.0001 | 0.0017 |
| TRELLIS (v1) | 0.0075 | 0.0039 | 0.0016 | 0.0021 | 0.0019 | 0.0002 | 0.0029 |
| TRELLIS (v2) | 0.0028 | 0.0008 | 0.0007 | 0.0021 | 0.0034 | 0.0005 | 0.0017 |
| TRELLIS.2 (v1) | 0.0059 | 0.0191 | 0.0009 | 0.0011 | 0.0017 | 0.0003 | 0.0048 |
| TRELLIS.2 (v2) | 0.0002 | 0.0001 | 0.0003 | 0.0008 | 0.0029 | 0.0003 | 0.0008 |

**view PSNR-L**

| method | Birdhouse | Cup | Figurine | Gargoyle | Mallard | Mouse | mean |
|---|---|---|---|---|---|---|---|
| nvdiffrec (multi-view) | 28.50 | 38.69 | 35.10 | 38.34 | 42.51 | 37.63 | 36.80 |
| RecGen (v1) | 14.14 | 23.49 | 24.26 | 26.04 | 24.25 | 22.88 | 22.51 |
| RecGen (v2) | 16.01 | 26.52 | 25.86 | 26.04 | 29.27 | 32.15 | 25.97 |
| SAM3D (v1) | 17.51 | 26.33 | 24.26 | 26.04 | 25.24 | 25.67 | 24.18 |
| SAM3D (v2) | 17.44 | 24.87 | 24.26 | 26.04 | 24.26 | 28.81 | 24.28 |
| TRELLIS (v1) | 15.06 | 23.49 | 24.46 | 26.12 | 25.07 | 30.71 | 24.15 |
| TRELLIS (v2) | 17.58 | 24.81 | 24.36 | 26.18 | 24.55 | 27.08 | 24.09 |
| TRELLIS.2 (v1) | 15.46 | 23.49 | 25.22 | 26.84 | 24.36 | 29.96 | 24.22 |
| TRELLIS.2 (v2) | 17.32 | 27.83 | 25.57 | 27.54 | 25.02 | 29.58 | 25.48 |

**view LPIPS**

| method | Birdhouse | Cup | Figurine | Gargoyle | Mallard | Mouse | mean |
|---|---|---|---|---|---|---|---|
| nvdiffrec (multi-view) | 0.0713 | 0.0198 | 0.0170 | 0.0193 | 0.0048 | 0.0083 | 0.0234 |
| RecGen (v1) | 0.2752 | 0.0839 | 0.0590 | 0.0823 | 0.0505 | 0.0270 | 0.0963 |
| RecGen (v2) | 0.1973 | 0.0384 | 0.0421 | 0.0657 | 0.0209 | 0.0204 | 0.0641 |
| SAM3D (v1) | 0.1900 | 0.0440 | 0.0523 | 0.0703 | 0.0278 | 0.0223 | 0.0678 |
| SAM3D (v2) | 0.1876 | 0.0382 | 0.0497 | 0.0705 | 0.0408 | 0.0156 | 0.0671 |
| TRELLIS (v1) | 0.2129 | 0.0595 | 0.0465 | 0.0668 | 0.0323 | 0.0174 | 0.0726 |
| TRELLIS (v2) | 0.1878 | 0.0486 | 0.0461 | 0.0672 | 0.0314 | 0.0187 | 0.0666 |
| TRELLIS.2 (v1) | 0.2076 | 0.0866 | 0.0439 | 0.0525 | 0.0334 | 0.0175 | 0.0736 |
| TRELLIS.2 (v2) | 0.1544 | 0.0343 | 0.0392 | 0.0519 | 0.0287 | 0.0241 | 0.0554 |

**albedo PSNR**

| method | Birdhouse | Cup | Figurine | Gargoyle | Mallard | Mouse | mean |
|---|---|---|---|---|---|---|---|
| nvdiffrec (multi-view) | 19.69 | 30.92 | 33.13 | 37.96 | 29.83 | 30.85 | 30.40 |
| RecGen (v1) | 14.14 | 23.63 | 28.68 | 27.13 | 27.04 | 26.01 | 24.44 |
| RecGen (v2) | 16.37 | 29.76 | 32.44 | 34.51 | 32.90 | 30.20 | 29.36 |
| SAM3D (v1) | 16.45 | 27.30 | 29.06 | 29.02 | 27.46 | 26.76 | 26.01 |
| SAM3D (v2) | 16.77 | 25.47 | 29.51 | 28.96 | 26.86 | 28.19 | 25.96 |
| TRELLIS (v1) | 15.34 | 25.16 | 29.28 | 29.75 | 28.23 | 29.19 | 26.16 |
| TRELLIS (v2) | 16.40 | 24.91 | 29.28 | 30.20 | 26.93 | 26.00 | 25.62 |
| TRELLIS.2 (v1) | 15.35 | 23.63 | 31.75 | 30.26 | 28.90 | 28.20 | 26.35 |
| TRELLIS.2 (v2) | 17.50 | 30.36 | 32.43 | 29.28 | 27.52 | 29.62 | 27.78 |

## A.4 Refinement: each source with its four optimized variables (means)

| method | chamfer (x2e-3) | normal (1-cos) | depth SI-MSE | albedo PSNR | view PSNR-L | view LPIPS | objects |
|---|---|---|---|---|---|---|---|
| **RecGen (v2)** | 0.0128 | 0.1610 | 0.0014 | 29.36 | 25.97 | 0.0641 | 6 |
| &nbsp;&nbsp;+ SLAT (T1) | 0.0096 | 0.0859 | 0.0007 | 35.52 | 27.00 | 0.0433 | 6 |
| &nbsp;&nbsp;+ cubefeats (T1) | 0.0132 | 0.1390 | 0.0009 | 32.60 | 26.79 | 0.0453 | 6 |
| &nbsp;&nbsp;+ SLAT (T2) | 0.0118 | 0.1265 | 0.0013 | 32.33 | 26.24 | 0.0606 | 5 |
| &nbsp;&nbsp;+ o-voxel (T2) | 0.0127 | 0.1640 | 0.0013 | 31.07 | 26.22 | 0.0580 | 6 |
| **SAM3D (v2)** | 0.0387 | 0.2199 | 0.0017 | 25.96 | 24.28 | 0.0671 | 6 |
| &nbsp;&nbsp;+ SLAT (T1) | 0.0262 | 0.1207 | 0.0007 | 33.31 | 26.58 | 0.0454 | 6 |
| &nbsp;&nbsp;+ cubefeats (T1) | 0.0335 | 0.1846 | 0.0012 | 28.97 | 25.52 | 0.0491 | 6 |
| &nbsp;&nbsp;+ SLAT (T2) | 0.0348 | 0.1813 | 0.0019 | 29.97 | 25.21 | 0.0690 | 6 |
| &nbsp;&nbsp;+ o-voxel (T2) | 0.0407 | 0.2131 | 0.0016 | 28.98 | 25.39 | 0.0611 | 6 |
| **TRELLIS (v2)** | 0.0266 | 0.2036 | 0.0017 | 25.62 | 24.09 | 0.0666 | 6 |
| &nbsp;&nbsp;+ SLAT (T1) | 0.0201 | 0.1383 | 0.0011 | 31.87 | 25.77 | 0.0476 | 6 |
| &nbsp;&nbsp;+ cubefeats (T1) | 0.0225 | 0.1651 | 0.0012 | 29.63 | 25.70 | 0.0472 | 6 |
| &nbsp;&nbsp;+ SLAT (T2) | 0.0226 | 0.1852 | 0.0020 | 29.25 | 24.51 | 0.0711 | 6 |
| &nbsp;&nbsp;+ o-voxel (T2) | 0.0269 | 0.2044 | 0.0016 | 28.78 | 24.85 | 0.0604 | 6 |
| **TRELLIS.2 (v2)** | 0.0201 | 0.1763 | 0.0008 | 27.78 | 25.48 | 0.0554 | 6 |
| &nbsp;&nbsp;+ SLAT (T1) | 0.0145 | 0.0805 | 0.0002 | 35.07 | 27.02 | 0.0402 | 6 |
| &nbsp;&nbsp;+ cubefeats (T1) | 0.0182 | 0.1141 | 0.0003 | 33.18 | 27.13 | 0.0398 | 6 |
| &nbsp;&nbsp;+ SLAT (T2) | 0.0164 | 0.1069 | 0.0007 | 33.15 | 26.00 | 0.0446 | 6 |
| &nbsp;&nbsp;+ o-voxel (T2) | 0.0199 | 0.1790 | 0.0008 | 30.87 | 25.79 | 0.0526 | 6 |
| *nvdiffrec (reference)* | 0.0359 | 0.1798 | 0.0011 | 30.40 | 36.80 | 0.0234 | 6 |

`RecGen + SLAT (T2)` is missing Cup (out of GPU memory); baseline on the same 5 objects:

| method | chamfer (x2e-3) | normal (1-cos) | depth SI-MSE | albedo PSNR |
|---|---|---|---|---|
| RecGen (v2), same 5 objects | 0.0144 | 0.1810 | 0.0016 | 29.28 |
| RecGen + SLAT (T2) | 0.0118 | 0.1265 | 0.0013 | 32.33 |

## A.5 Refinement, per object

**chamfer (x2e-3)**

| method | Birdhouse | Cup | Figurine | Gargoyle | Mallard | Mouse | mean |
|---|---|---|---|---|---|---|---|
| nvdiffrec (multi-view) | 0.0818 | 0.0518 | 0.0400 | 0.0297 | 0.0021 | 0.0101 | 0.0359 |
| RecGen (v2) | 0.0236 | 0.0047 | 0.0104 | 0.0223 | 0.0073 | 0.0084 | 0.0128 |
| SAM3D (v2) | 0.0382 | 0.0057 | 0.0199 | 0.0661 | 0.0929 | 0.0096 | 0.0387 |
| TRELLIS (v2) | 0.0230 | 0.0073 | 0.0180 | 0.0468 | 0.0558 | 0.0088 | 0.0266 |
| TRELLIS.2 (v2) | 0.0244 | 0.0029 | 0.0303 | 0.0147 | 0.0383 | 0.0098 | 0.0201 |
| RecGen + SLAT (T1) | 0.0154 | 0.0045 | 0.0099 | 0.0165 | 0.0043 | 0.0071 | 0.0096 |
| RecGen + cubefeats (T1) | 0.0233 | 0.0054 | 0.0123 | 0.0225 | 0.0078 | 0.0080 | 0.0132 |
| RecGen + SLAT (T2) | 0.0191 | -- | 0.0093 | 0.0187 | 0.0045 | 0.0076 | 0.0118 |
| RecGen + o-voxel (T2) | 0.0229 | 0.0046 | 0.0107 | 0.0225 | 0.0073 | 0.0083 | 0.0127 |
| SAM3D + SLAT (T1) | 0.0269 | 0.0059 | 0.0163 | 0.0568 | 0.0414 | 0.0099 | 0.0262 |
| SAM3D + cubefeats (T1) | 0.0380 | 0.0062 | 0.0200 | 0.0667 | 0.0587 | 0.0115 | 0.0335 |
| SAM3D + SLAT (T2) | 0.0323 | 0.0050 | 0.0176 | 0.0627 | 0.0829 | 0.0085 | 0.0348 |
| SAM3D + o-voxel (T2) | 0.0386 | 0.0056 | 0.0200 | 0.0703 | 0.1003 | 0.0095 | 0.0407 |
| TRELLIS + SLAT (T1) | 0.0151 | 0.0067 | 0.0128 | 0.0376 | 0.0421 | 0.0061 | 0.0201 |
| TRELLIS + cubefeats (T1) | 0.0221 | 0.0085 | 0.0169 | 0.0427 | 0.0363 | 0.0082 | 0.0225 |
| TRELLIS + SLAT (T2) | 0.0193 | 0.0066 | 0.0146 | 0.0399 | 0.0477 | 0.0073 | 0.0226 |
| TRELLIS + o-voxel (T2) | 0.0227 | 0.0071 | 0.0179 | 0.0482 | 0.0566 | 0.0086 | 0.0269 |
| TRELLIS.2 + SLAT (T1) | 0.0271 | 0.0030 | 0.0245 | 0.0085 | 0.0153 | 0.0088 | 0.0145 |
| TRELLIS.2 + cubefeats (T1) | 0.0299 | 0.0044 | 0.0232 | 0.0138 | 0.0273 | 0.0109 | 0.0182 |
| TRELLIS.2 + SLAT (T2) | 0.0215 | 0.0026 | 0.0218 | 0.0114 | 0.0327 | 0.0086 | 0.0164 |
| TRELLIS.2 + o-voxel (T2) | 0.0247 | 0.0029 | 0.0263 | 0.0152 | 0.0405 | 0.0101 | 0.0199 |

**normal (1-cos)**

| method | Birdhouse | Cup | Figurine | Gargoyle | Mallard | Mouse | mean |
|---|---|---|---|---|---|---|---|
| nvdiffrec (multi-view) | 0.1958 | 0.0682 | 0.2826 | 0.2522 | 0.0482 | 0.2321 | 0.1798 |
| RecGen (v2) | 0.1896 | 0.0611 | 0.2040 | 0.2095 | 0.0880 | 0.2141 | 0.1610 |
| SAM3D (v2) | 0.1579 | 0.0393 | 0.2316 | 0.2658 | 0.3373 | 0.2878 | 0.2199 |
| TRELLIS (v2) | 0.1701 | 0.0884 | 0.2384 | 0.2419 | 0.2116 | 0.2709 | 0.2036 |
| TRELLIS.2 (v2) | 0.1400 | 0.0245 | 0.2176 | 0.2006 | 0.1941 | 0.2810 | 0.1763 |
| RecGen + SLAT (T1) | 0.1003 | 0.0400 | 0.1217 | 0.0974 | 0.0568 | 0.0990 | 0.0859 |
| RecGen + cubefeats (T1) | 0.1663 | 0.0550 | 0.1815 | 0.1801 | 0.0745 | 0.1768 | 0.1390 |
| RecGen + SLAT (T2) | 0.1447 | -- | 0.1272 | 0.1496 | 0.0449 | 0.1659 | 0.1265 |
| RecGen + o-voxel (T2) | 0.1844 | 0.0636 | 0.2093 | 0.2132 | 0.0877 | 0.2257 | 0.1640 |
| SAM3D + SLAT (T1) | 0.0820 | 0.0193 | 0.1670 | 0.1537 | 0.1698 | 0.1325 | 0.1207 |
| SAM3D + cubefeats (T1) | 0.1400 | 0.0341 | 0.2076 | 0.2543 | 0.2113 | 0.2605 | 0.1846 |
| SAM3D + SLAT (T2) | 0.1375 | 0.0325 | 0.2252 | 0.2075 | 0.2918 | 0.1931 | 0.1813 |
| SAM3D + o-voxel (T2) | 0.1547 | 0.0378 | 0.2373 | 0.2676 | 0.2956 | 0.2860 | 0.2131 |
| TRELLIS + SLAT (T1) | 0.0862 | 0.0710 | 0.1706 | 0.1635 | 0.1997 | 0.1389 | 0.1383 |
| TRELLIS + cubefeats (T1) | 0.1474 | 0.0804 | 0.2139 | 0.2073 | 0.1157 | 0.2258 | 0.1651 |
| TRELLIS + SLAT (T2) | 0.1531 | 0.0693 | 0.2392 | 0.2055 | 0.2216 | 0.2226 | 0.1852 |
| TRELLIS + o-voxel (T2) | 0.1646 | 0.0843 | 0.2426 | 0.2447 | 0.2146 | 0.2755 | 0.2044 |
| TRELLIS.2 + SLAT (T1) | 0.0482 | 0.0171 | 0.1208 | 0.1027 | 0.0865 | 0.1077 | 0.0805 |
| TRELLIS.2 + cubefeats (T1) | 0.0871 | 0.0180 | 0.1748 | 0.1529 | 0.0759 | 0.1758 | 0.1141 |
| TRELLIS.2 + SLAT (T2) | 0.0858 | 0.0117 | 0.1341 | 0.1097 | 0.1636 | 0.1362 | 0.1069 |
| TRELLIS.2 + o-voxel (T2) | 0.1333 | 0.0237 | 0.2263 | 0.2065 | 0.2018 | 0.2825 | 0.1790 |

**depth SI-MSE**

| method | Birdhouse | Cup | Figurine | Gargoyle | Mallard | Mouse | mean |
|---|---|---|---|---|---|---|---|
| nvdiffrec (multi-view) | 0.0023 | 0.0000 | 0.0011 | 0.0029 | 0.0002 | 0.0000 | 0.0011 |
| RecGen (v2) | 0.0058 | 0.0003 | 0.0004 | 0.0013 | 0.0005 | 0.0001 | 0.0014 |
| SAM3D (v2) | 0.0014 | 0.0004 | 0.0007 | 0.0029 | 0.0049 | 0.0001 | 0.0017 |
| TRELLIS (v2) | 0.0028 | 0.0008 | 0.0007 | 0.0021 | 0.0034 | 0.0005 | 0.0017 |
| TRELLIS.2 (v2) | 0.0002 | 0.0001 | 0.0003 | 0.0008 | 0.0029 | 0.0003 | 0.0008 |
| RecGen + SLAT (T1) | 0.0022 | 0.0001 | 0.0005 | 0.0010 | 0.0004 | 0.0000 | 0.0007 |
| RecGen + cubefeats (T1) | 0.0026 | 0.0003 | 0.0006 | 0.0013 | 0.0005 | 0.0001 | 0.0009 |
| RecGen + SLAT (T2) | 0.0045 | -- | 0.0004 | 0.0011 | 0.0002 | 0.0003 | 0.0013 |
| RecGen + o-voxel (T2) | 0.0055 | 0.0002 | 0.0004 | 0.0012 | 0.0005 | 0.0001 | 0.0013 |
| SAM3D + SLAT (T1) | 0.0005 | 0.0002 | 0.0006 | 0.0018 | 0.0010 | 0.0001 | 0.0007 |
| SAM3D + cubefeats (T1) | 0.0013 | 0.0003 | 0.0007 | 0.0027 | 0.0019 | 0.0001 | 0.0012 |
| SAM3D + SLAT (T2) | 0.0030 | 0.0003 | 0.0017 | 0.0023 | 0.0041 | 0.0002 | 0.0019 |
| SAM3D + o-voxel (T2) | 0.0013 | 0.0004 | 0.0007 | 0.0028 | 0.0043 | 0.0001 | 0.0016 |
| TRELLIS + SLAT (T1) | 0.0008 | 0.0006 | 0.0005 | 0.0015 | 0.0031 | 0.0001 | 0.0011 |
| TRELLIS + cubefeats (T1) | 0.0022 | 0.0008 | 0.0007 | 0.0018 | 0.0009 | 0.0005 | 0.0012 |
| TRELLIS + SLAT (T2) | 0.0033 | 0.0007 | 0.0017 | 0.0020 | 0.0033 | 0.0010 | 0.0020 |
| TRELLIS + o-voxel (T2) | 0.0026 | 0.0008 | 0.0007 | 0.0019 | 0.0034 | 0.0005 | 0.0016 |
| TRELLIS.2 + SLAT (T1) | 0.0002 | 0.0001 | 0.0004 | 0.0004 | 0.0002 | 0.0002 | 0.0002 |
| TRELLIS.2 + cubefeats (T1) | 0.0002 | 0.0001 | 0.0005 | 0.0008 | 0.0002 | 0.0002 | 0.0003 |
| TRELLIS.2 + SLAT (T2) | 0.0002 | 0.0001 | 0.0003 | 0.0005 | 0.0026 | 0.0003 | 0.0007 |
| TRELLIS.2 + o-voxel (T2) | 0.0002 | 0.0001 | 0.0003 | 0.0008 | 0.0030 | 0.0003 | 0.0008 |

**albedo PSNR**

| method | Birdhouse | Cup | Figurine | Gargoyle | Mallard | Mouse | mean |
|---|---|---|---|---|---|---|---|
| nvdiffrec (multi-view) | 19.69 | 30.92 | 33.13 | 37.96 | 29.83 | 30.85 | 30.40 |
| RecGen (v2) | 16.37 | 29.76 | 32.44 | 34.51 | 32.90 | 30.20 | 29.36 |
| SAM3D (v2) | 16.77 | 25.47 | 29.51 | 28.96 | 26.86 | 28.19 | 25.96 |
| TRELLIS (v2) | 16.40 | 24.91 | 29.28 | 30.20 | 26.93 | 26.00 | 25.62 |
| TRELLIS.2 (v2) | 17.50 | 30.36 | 32.43 | 29.28 | 27.52 | 29.62 | 27.78 |
| RecGen + SLAT (T1) | 21.40 | 39.36 | 36.43 | 38.68 | 41.93 | 35.30 | 35.52 |
| RecGen + cubefeats (T1) | 19.26 | 35.75 | 35.90 | 36.76 | 36.01 | 31.93 | 32.60 |
| RecGen + SLAT (T2) | 18.96 | -- | 36.24 | 37.18 | 38.01 | 31.23 | 32.33 |
| RecGen + o-voxel (T2) | 18.70 | 34.04 | 34.17 | 34.32 | 34.85 | 30.34 | 31.07 |
| SAM3D + SLAT (T1) | 23.15 | 39.55 | 35.81 | 36.30 | 31.81 | 33.25 | 33.31 |
| SAM3D + cubefeats (T1) | 20.10 | 27.88 | 33.57 | 32.43 | 29.06 | 30.80 | 28.97 |
| SAM3D + SLAT (T2) | 19.29 | 33.17 | 33.83 | 35.21 | 27.19 | 31.12 | 29.97 |
| SAM3D + o-voxel (T2) | 20.65 | 31.22 | 33.14 | 32.45 | 27.20 | 29.24 | 28.98 |
| TRELLIS + SLAT (T1) | 23.01 | 35.16 | 35.76 | 36.75 | 28.66 | 31.87 | 31.87 |
| TRELLIS + cubefeats (T1) | 19.92 | 28.53 | 34.89 | 35.47 | 31.34 | 27.61 | 29.63 |
| TRELLIS + SLAT (T2) | 19.06 | 31.96 | 33.54 | 35.29 | 28.13 | 27.49 | 29.25 |
| TRELLIS + o-voxel (T2) | 19.75 | 30.25 | 33.54 | 33.71 | 27.92 | 27.53 | 28.78 |
| TRELLIS.2 + SLAT (T1) | 25.70 | 42.09 | 36.50 | 39.17 | 33.90 | 33.10 | 35.07 |
| TRELLIS.2 + cubefeats (T1) | 22.93 | 37.42 | 35.80 | 37.30 | 33.83 | 31.79 | 33.18 |
| TRELLIS.2 + SLAT (T2) | 23.35 | 40.66 | 35.99 | 38.25 | 29.10 | 31.58 | 33.15 |
| TRELLIS.2 + o-voxel (T2) | 22.49 | 35.71 | 34.06 | 34.78 | 28.45 | 29.75 | 30.87 |

**view PSNR-L**

| method | Birdhouse | Cup | Figurine | Gargoyle | Mallard | Mouse | mean |
|---|---|---|---|---|---|---|---|
| nvdiffrec (multi-view) | 28.50 | 38.69 | 35.10 | 38.34 | 42.51 | 37.63 | 36.80 |
| RecGen (v2) | 16.01 | 26.52 | 25.86 | 26.04 | 29.27 | 32.15 | 25.97 |
| SAM3D (v2) | 17.44 | 24.87 | 24.26 | 26.04 | 24.26 | 28.81 | 24.28 |
| TRELLIS (v2) | 17.58 | 24.81 | 24.36 | 26.18 | 24.55 | 27.08 | 24.09 |
| TRELLIS.2 (v2) | 17.32 | 27.83 | 25.57 | 27.54 | 25.02 | 29.58 | 25.48 |
| RecGen + SLAT (T1) | 18.39 | 28.42 | 26.42 | 27.20 | 31.26 | 30.29 | 27.00 |
| RecGen + cubefeats (T1) | 17.79 | 28.57 | 26.39 | 26.76 | 30.77 | 30.48 | 26.79 |
| RecGen + SLAT (T2) | 16.99 | -- | 26.56 | 27.00 | 31.50 | 29.14 | 26.24 |
| RecGen + o-voxel (T2) | 16.64 | 27.81 | 26.19 | 26.49 | 30.45 | 29.73 | 26.22 |
| SAM3D + SLAT (T1) | 19.84 | 28.50 | 26.08 | 26.73 | 28.39 | 29.93 | 26.58 |
| SAM3D + cubefeats (T1) | 19.56 | 27.23 | 24.93 | 26.83 | 25.86 | 28.73 | 25.52 |
| SAM3D + SLAT (T2) | 17.70 | 27.85 | 25.46 | 26.50 | 24.40 | 29.35 | 25.21 |
| SAM3D + o-voxel (T2) | 18.93 | 28.13 | 25.57 | 26.35 | 24.45 | 28.90 | 25.39 |
| TRELLIS + SLAT (T1) | 19.61 | 27.34 | 26.16 | 26.86 | 24.82 | 29.83 | 25.77 |
| TRELLIS + cubefeats (T1) | 19.01 | 27.00 | 25.98 | 26.91 | 28.38 | 26.92 | 25.70 |
| TRELLIS + SLAT (T2) | 17.65 | 27.11 | 25.33 | 26.49 | 24.62 | 25.83 | 24.51 |
| TRELLIS + o-voxel (T2) | 18.00 | 27.17 | 25.82 | 26.37 | 24.66 | 27.12 | 24.85 |
| TRELLIS.2 + SLAT (T1) | 20.01 | 28.69 | 26.35 | 27.31 | 30.02 | 29.73 | 27.02 |
| TRELLIS.2 + cubefeats (T1) | 20.39 | 29.05 | 26.19 | 27.13 | 30.72 | 29.30 | 27.13 |
| TRELLIS.2 + SLAT (T2) | 19.62 | 28.58 | 26.43 | 27.14 | 25.01 | 29.19 | 26.00 |
| TRELLIS.2 + o-voxel (T2) | 19.63 | 28.58 | 26.06 | 26.65 | 24.90 | 28.92 | 25.79 |

**view LPIPS**

| method | Birdhouse | Cup | Figurine | Gargoyle | Mallard | Mouse | mean |
|---|---|---|---|---|---|---|---|
| nvdiffrec (multi-view) | 0.0713 | 0.0198 | 0.0170 | 0.0193 | 0.0048 | 0.0083 | 0.0234 |
| RecGen (v2) | 0.1973 | 0.0384 | 0.0421 | 0.0657 | 0.0209 | 0.0204 | 0.0641 |
| SAM3D (v2) | 0.1876 | 0.0382 | 0.0497 | 0.0705 | 0.0408 | 0.0156 | 0.0671 |
| TRELLIS (v2) | 0.1878 | 0.0486 | 0.0461 | 0.0672 | 0.0314 | 0.0187 | 0.0666 |
| TRELLIS.2 (v2) | 0.1544 | 0.0343 | 0.0392 | 0.0519 | 0.0287 | 0.0241 | 0.0554 |
| RecGen + SLAT (T1) | 0.1285 | 0.0286 | 0.0334 | 0.0415 | 0.0159 | 0.0116 | 0.0433 |
| RecGen + cubefeats (T1) | 0.1377 | 0.0285 | 0.0332 | 0.0445 | 0.0156 | 0.0122 | 0.0453 |
| RecGen + SLAT (T2) | 0.1834 | -- | 0.0357 | 0.0504 | 0.0157 | 0.0180 | 0.0606 |
| RecGen + o-voxel (T2) | 0.1663 | 0.0371 | 0.0434 | 0.0583 | 0.0234 | 0.0195 | 0.0580 |
| SAM3D + SLAT (T1) | 0.1189 | 0.0290 | 0.0358 | 0.0509 | 0.0258 | 0.0118 | 0.0454 |
| SAM3D + cubefeats (T1) | 0.1351 | 0.0309 | 0.0366 | 0.0538 | 0.0253 | 0.0132 | 0.0491 |
| SAM3D + SLAT (T2) | 0.1994 | 0.0397 | 0.0512 | 0.0601 | 0.0468 | 0.0169 | 0.0690 |
| SAM3D + o-voxel (T2) | 0.1486 | 0.0393 | 0.0474 | 0.0665 | 0.0432 | 0.0213 | 0.0611 |
| TRELLIS + SLAT (T1) | 0.1165 | 0.0327 | 0.0371 | 0.0520 | 0.0332 | 0.0142 | 0.0476 |
| TRELLIS + cubefeats (T1) | 0.1322 | 0.0341 | 0.0358 | 0.0476 | 0.0198 | 0.0139 | 0.0472 |
| TRELLIS + SLAT (T2) | 0.1959 | 0.0422 | 0.0532 | 0.0660 | 0.0423 | 0.0270 | 0.0711 |
| TRELLIS + o-voxel (T2) | 0.1526 | 0.0429 | 0.0471 | 0.0601 | 0.0374 | 0.0221 | 0.0604 |
| TRELLIS.2 + SLAT (T1) | 0.1046 | 0.0273 | 0.0334 | 0.0438 | 0.0200 | 0.0123 | 0.0402 |
| TRELLIS.2 + cubefeats (T1) | 0.1085 | 0.0276 | 0.0336 | 0.0416 | 0.0153 | 0.0122 | 0.0398 |
| TRELLIS.2 + SLAT (T2) | 0.1183 | 0.0278 | 0.0340 | 0.0437 | 0.0286 | 0.0151 | 0.0446 |
| TRELLIS.2 + o-voxel (T2) | 0.1255 | 0.0344 | 0.0445 | 0.0560 | 0.0356 | 0.0198 | 0.0526 |

## A.6 Chamfer in millimetres: encoding cost against optimization gain

| variable | source | object | source mesh | after encode/decode | after optimization |
|---|---|---|---|---|---|
| SLAT (T1) | RecGen | Birdhouse | 2.63 | 2.51 | 1.80 |
| SLAT (T1) | RecGen | Cup | 1.20 | 1.32 | 1.15 |
| SLAT (T1) | RecGen | Figurine | 1.79 | 1.91 | 1.54 |
| SLAT (T1) | RecGen | Gargoyle | 2.61 | 2.61 | 2.05 |
| SLAT (T1) | RecGen | Mallard | 1.39 | 1.51 | 0.94 |
| SLAT (T1) | RecGen | Mouse | 1.56 | 1.58 | 1.32 |
| SLAT (T1) | SAM3D | Birdhouse | 2.94 | 3.08 | 2.44 |
| SLAT (T1) | SAM3D | Cup | 1.48 | 1.56 | 1.36 |
| SLAT (T1) | SAM3D | Figurine | 2.37 | 2.36 | 2.07 |
| SLAT (T1) | SAM3D | Gargoyle | 4.17 | 4.16 | 3.53 |
| SLAT (T1) | SAM3D | Mallard | 4.85 | 4.35 | 3.43 |
| SLAT (T1) | SAM3D | Mouse | 1.64 | 1.72 | 1.51 |
| SLAT (T1) | TRELLIS | Birdhouse | 2.47 | 2.50 | 1.92 |
| SLAT (T1) | TRELLIS | Cup | 1.45 | 1.61 | 1.41 |
| SLAT (T1) | TRELLIS | Figurine | 2.31 | 2.26 | 1.93 |
| SLAT (T1) | TRELLIS | Gargoyle | 3.42 | 3.16 | 2.71 |
| SLAT (T1) | TRELLIS | Mallard | 2.50 | 2.46 | 2.01 |
| SLAT (T1) | TRELLIS | Mouse | 1.57 | 1.55 | 1.26 |
| SLAT (T1) | TRELLIS.2 | Birdhouse | 2.42 | 2.56 | 2.28 |
| SLAT (T1) | TRELLIS.2 | Cup | 1.00 | 1.13 | 1.01 |
| SLAT (T1) | TRELLIS.2 | Figurine | 2.44 | 2.22 | 2.14 |
| SLAT (T1) | TRELLIS.2 | Gargoyle | 2.18 | 2.05 | 1.60 |
| SLAT (T1) | TRELLIS.2 | Mallard | 3.11 | 2.75 | 2.10 |
| SLAT (T1) | TRELLIS.2 | Mouse | 1.72 | 1.82 | 1.52 |
| cubefeats (T1) | RecGen | Birdhouse | 2.52 | 2.40 | 2.35 |
| cubefeats (T1) | RecGen | Cup | 1.21 | 1.32 | 1.31 |
| cubefeats (T1) | RecGen | Figurine | 1.77 | 1.87 | 1.85 |
| cubefeats (T1) | RecGen | Gargoyle | 2.61 | 2.63 | 2.59 |
| cubefeats (T1) | RecGen | Mallard | 1.37 | 1.50 | 1.47 |
| cubefeats (T1) | RecGen | Mouse | 1.57 | 1.57 | 1.55 |
| cubefeats (T1) | SAM3D | Birdhouse | 2.96 | 3.09 | 3.06 |
| cubefeats (T1) | SAM3D | Cup | 1.48 | 1.56 | 1.54 |
| cubefeats (T1) | SAM3D | Figurine | 2.36 | 2.38 | 2.35 |
| cubefeats (T1) | SAM3D | Gargoyle | 4.14 | 4.15 | 4.08 |
| cubefeats (T1) | SAM3D | Mallard | 4.27 | 3.68 | 3.67 |
| cubefeats (T1) | SAM3D | Mouse | 1.63 | 1.72 | 1.71 |
| cubefeats (T1) | TRELLIS | Birdhouse | 2.47 | 2.49 | 2.45 |
| cubefeats (T1) | TRELLIS | Cup | 1.45 | 1.62 | 1.61 |
| cubefeats (T1) | TRELLIS | Figurine | 2.34 | 2.28 | 2.25 |
| cubefeats (T1) | TRELLIS | Gargoyle | 3.40 | 3.17 | 3.13 |
| cubefeats (T1) | TRELLIS | Mallard | 3.52 | 3.16 | 3.12 |
| cubefeats (T1) | TRELLIS | Mouse | 1.57 | 1.54 | 1.50 |
| cubefeats (T1) | TRELLIS.2 | Birdhouse | 2.40 | 2.54 | 2.49 |
| cubefeats (T1) | TRELLIS.2 | Cup | 1.00 | 1.12 | 1.12 |
| cubefeats (T1) | TRELLIS.2 | Figurine | 2.45 | 2.25 | 2.20 |
| cubefeats (T1) | TRELLIS.2 | Gargoyle | 2.21 | 2.10 | 2.05 |
| cubefeats (T1) | TRELLIS.2 | Mallard | 3.15 | 2.80 | 2.79 |
| cubefeats (T1) | TRELLIS.2 | Mouse | 1.71 | 1.79 | 1.77 |

| variable | source | object | after encode/decode | after optimization |
|---|---|---|---|---|
| SLAT (T2) | RecGen | Birdhouse | 2.31 | 2.20 |
| SLAT (T2) | RecGen | Figurine | 1.74 | 1.63 |
| SLAT (T2) | RecGen | Gargoyle | 2.58 | 2.30 |
| SLAT (T2) | RecGen | Mallard | 1.37 | 1.09 |
| SLAT (T2) | RecGen | Mouse | 1.55 | 1.45 |
| SLAT (T2) | SAM3D | Birdhouse | 2.95 | 2.76 |
| SLAT (T2) | SAM3D | Cup | 1.49 | 1.39 |
| SLAT (T2) | SAM3D | Figurine | 2.36 | 2.21 |
| SLAT (T2) | SAM3D | Gargoyle | 4.17 | 3.79 |
| SLAT (T2) | SAM3D | Mallard | 4.14 | 3.54 |
| SLAT (T2) | SAM3D | Mouse | 1.65 | 1.53 |
| SLAT (T2) | TRELLIS | Birdhouse | 2.47 | 2.34 |
| SLAT (T2) | TRELLIS | Cup | 1.45 | 1.39 |
| SLAT (T2) | TRELLIS | Figurine | 2.32 | 2.07 |
| SLAT (T2) | TRELLIS | Gargoyle | 3.40 | 3.00 |
| SLAT (T2) | TRELLIS | Mallard | 2.54 | 2.11 |
| SLAT (T2) | TRELLIS | Mouse | 1.57 | 1.37 |
| SLAT (T2) | TRELLIS.2 | Birdhouse | 2.49 | 2.35 |
| SLAT (T2) | TRELLIS.2 | Cup | 1.00 | 0.95 |
| SLAT (T2) | TRELLIS.2 | Figurine | 2.26 | 2.03 |
| SLAT (T2) | TRELLIS.2 | Gargoyle | 2.25 | 1.94 |
| SLAT (T2) | TRELLIS.2 | Mallard | 2.47 | 2.21 |
| SLAT (T2) | TRELLIS.2 | Mouse | 1.74 | 1.55 |
| o-voxel (T2) | RecGen | Birdhouse | 2.31 | 2.31 |
| o-voxel (T2) | RecGen | Cup | 1.21 | 1.20 |
| o-voxel (T2) | RecGen | Figurine | 1.75 | 1.75 |
| o-voxel (T2) | RecGen | Gargoyle | 2.62 | 2.60 |
| o-voxel (T2) | RecGen | Mallard | 1.37 | 1.39 |
| o-voxel (T2) | RecGen | Mouse | 1.55 | 1.55 |
| o-voxel (T2) | SAM3D | Birdhouse | 2.96 | 2.96 |
| o-voxel (T2) | SAM3D | Cup | 1.48 | 1.48 |
| o-voxel (T2) | SAM3D | Figurine | 2.39 | 2.38 |
| o-voxel (T2) | SAM3D | Gargoyle | 4.18 | 4.18 |
| o-voxel (T2) | SAM3D | Mallard | 4.16 | 4.18 |
| o-voxel (T2) | SAM3D | Mouse | 1.64 | 1.64 |
| o-voxel (T2) | TRELLIS | Birdhouse | 2.44 | 2.45 |
| o-voxel (T2) | TRELLIS | Cup | 1.44 | 1.45 |
| o-voxel (T2) | TRELLIS | Figurine | 2.32 | 2.32 |
| o-voxel (T2) | TRELLIS | Gargoyle | 3.40 | 3.41 |
| o-voxel (T2) | TRELLIS | Mallard | 2.54 | 2.53 |
| o-voxel (T2) | TRELLIS | Mouse | 1.57 | 1.56 |
| o-voxel (T2) | TRELLIS.2 | Birdhouse | 2.49 | 2.49 |
| o-voxel (T2) | TRELLIS.2 | Cup | 1.00 | 1.00 |
| o-voxel (T2) | TRELLIS.2 | Figurine | 2.26 | 2.26 |
| o-voxel (T2) | TRELLIS.2 | Gargoyle | 2.24 | 2.25 |
| o-voxel (T2) | TRELLIS.2 | Mallard | 2.48 | 2.48 |
| o-voxel (T2) | TRELLIS.2 | Mouse | 1.74 | 1.73 |

| variable | source | objects | source mesh | after encode/decode | after optimization |
|---|---|---|---|---|---|
| SLAT (T1) | RecGen | 6 | 1.86 | 1.91 | 1.47 |
| SLAT (T1) | SAM3D | 6 | 2.91 | 2.87 | 2.39 |
| SLAT (T1) | TRELLIS | 6 | 2.29 | 2.26 | 1.87 |
| SLAT (T1) | TRELLIS.2 | 6 | 2.15 | 2.09 | 1.78 |
| cubefeats (T1) | RecGen | 6 | 1.84 | 1.88 | 1.85 |
| cubefeats (T1) | SAM3D | 6 | 2.81 | 2.76 | 2.73 |
| cubefeats (T1) | TRELLIS | 6 | 2.46 | 2.38 | 2.34 |
| cubefeats (T1) | TRELLIS.2 | 6 | 2.15 | 2.10 | 2.07 |
| SLAT (T2) | RecGen | 5 | -- | 1.91 | 1.74 |
| SLAT (T2) | SAM3D | 6 | -- | 2.79 | 2.54 |
| SLAT (T2) | TRELLIS | 6 | -- | 2.29 | 2.05 |
| SLAT (T2) | TRELLIS.2 | 6 | -- | 2.04 | 1.84 |
| o-voxel (T2) | RecGen | 6 | -- | 1.80 | 1.80 |
| o-voxel (T2) | SAM3D | 6 | -- | 2.80 | 2.80 |
| o-voxel (T2) | TRELLIS | 6 | -- | 2.29 | 2.29 |
| o-voxel (T2) | TRELLIS.2 | 6 | -- | 2.03 | 2.03 |

## A.7 TRELLIS.2: topology and gate changes per run

| variable | source | object | comp. init | comp. final | Euler init | Euler final | voxels + | voxels - | quad flips |
|---|---|---|---|---|---|---|---|---|---|
| SLAT (T2) | RecGen | Birdhouse | 12 | 373 | -1,816 | -3,463 | 838,086 | 973,332 | 407,333 |
| SLAT (T2) | RecGen | Figurine | 1 | 92 | -10,336 | -3,521 | 726,875 | 793,400 | 558,889 |
| SLAT (T2) | RecGen | Gargoyle | 15 | 334 | -2,908 | 43,002 | 1,339,016 | 1,349,619 | 783,527 |
| SLAT (T2) | RecGen | Mallard | 2 | 39 | -600 | 4,189 | 375,920 | 385,037 | 242,913 |
| SLAT (T2) | RecGen | Mouse | 29 | 202 | -1,051 | 5,087 | 645,711 | 654,658 | 337,853 |
| SLAT (T2) | SAM3D | Birdhouse | 1 | 527 | -579 | 1,123 | 818,969 | 783,967 | 264,676 |
| SLAT (T2) | SAM3D | Cup | 1 | 43 | -107 | -2,153 | 494,864 | 512,616 | 700,928 |
| SLAT (T2) | SAM3D | Figurine | 1 | 228 | -696 | 11,255 | 656,517 | 626,274 | 268,678 |
| SLAT (T2) | SAM3D | Gargoyle | 20 | 284 | -1,473 | 14,980 | 1,032,036 | 1,002,930 | 608,920 |
| SLAT (T2) | SAM3D | Mallard | 1 | 55 | -182 | 5,793 | 305,658 | 311,817 | 147,474 |
| SLAT (T2) | SAM3D | Mouse | 1 | 164 | -1,294 | 6,690 | 654,263 | 643,654 | 308,995 |
| SLAT (T2) | TRELLIS | Birdhouse | 2 | 390 | -621 | 1,222 | 772,937 | 751,604 | 272,167 |
| SLAT (T2) | TRELLIS | Cup | 1 | 86 | -275 | -4,607 | 653,297 | 697,009 | 853,777 |
| SLAT (T2) | TRELLIS | Figurine | 1 | 131 | -731 | 8,647 | 609,922 | 589,630 | 234,944 |
| SLAT (T2) | TRELLIS | Gargoyle | 1 | 141 | -1,446 | 17,871 | 706,243 | 674,011 | 382,120 |
| SLAT (T2) | TRELLIS | Mallard | 1 | 48 | -84 | 2,063 | 288,120 | 304,230 | 126,098 |
| SLAT (T2) | TRELLIS | Mouse | 1 | 67 | -745 | 14,791 | 386,339 | 327,789 | 160,117 |
| SLAT (T2) | TRELLIS.2 | Birdhouse | 6 | 225 | -30,422 | 31,425 | 735,224 | 1,315,460 | 951,085 |
| SLAT (T2) | TRELLIS.2 | Cup | 1 | 2 | -107,778 | -54,943 | 478,333 | 517,027 | 1,537,238 |
| SLAT (T2) | TRELLIS.2 | Figurine | 2 | 82 | -58,692 | 32,603 | 744,544 | 919,951 | 701,924 |
| SLAT (T2) | TRELLIS.2 | Gargoyle | 5 | 83 | -104,556 | 104,702 | 885,292 | 955,887 | 1,007,318 |
| SLAT (T2) | TRELLIS.2 | Mallard | 1 | 10 | -37,041 | 1,275 | 228,726 | 249,028 | 394,637 |
| SLAT (T2) | TRELLIS.2 | Mouse | 31 | 66 | -38,489 | 43,756 | 445,069 | 482,234 | 456,799 |
| o-voxel (T2) | RecGen | Birdhouse | 12 | 12 | -1,808 | -1,808 | 0 | 0 | 0 |
| o-voxel (T2) | RecGen | Cup | 1 | 1 | -2,992 | -2,992 | 0 | 0 | 0 |
| o-voxel (T2) | RecGen | Figurine | 1 | 1 | -10,324 | -10,324 | 0 | 0 | 0 |
| o-voxel (T2) | RecGen | Gargoyle | 16 | 16 | -2,920 | -2,920 | 0 | 0 | 0 |
| o-voxel (T2) | RecGen | Mallard | 2 | 2 | -599 | -599 | 0 | 0 | 0 |
| o-voxel (T2) | RecGen | Mouse | 29 | 29 | -1,045 | -1,045 | 0 | 0 | 0 |
| o-voxel (T2) | SAM3D | Birdhouse | 1 | 1 | -575 | -575 | 0 | 0 | 0 |
| o-voxel (T2) | SAM3D | Cup | 1 | 1 | -94 | -94 | 0 | 0 | 0 |
| o-voxel (T2) | SAM3D | Figurine | 1 | 1 | -694 | -694 | 0 | 0 | 0 |
| o-voxel (T2) | SAM3D | Gargoyle | 20 | 20 | -1,489 | -1,489 | 0 | 0 | 0 |
| o-voxel (T2) | SAM3D | Mallard | 1 | 1 | -171 | -171 | 0 | 0 | 0 |
| o-voxel (T2) | SAM3D | Mouse | 1 | 1 | -1,321 | -1,321 | 0 | 0 | 0 |
| o-voxel (T2) | TRELLIS | Birdhouse | 2 | 2 | -626 | -626 | 0 | 0 | 0 |
| o-voxel (T2) | TRELLIS | Cup | 1 | 1 | -275 | -275 | 0 | 0 | 0 |
| o-voxel (T2) | TRELLIS | Figurine | 1 | 1 | -720 | -720 | 0 | 0 | 0 |
| o-voxel (T2) | TRELLIS | Gargoyle | 1 | 1 | -1,450 | -1,450 | 0 | 0 | 0 |
| o-voxel (T2) | TRELLIS | Mallard | 1 | 1 | -86 | -86 | 0 | 0 | 0 |
| o-voxel (T2) | TRELLIS | Mouse | 1 | 1 | -742 | -742 | 0 | 0 | 0 |
| o-voxel (T2) | TRELLIS.2 | Birdhouse | 6 | 6 | -30,421 | -30,421 | 0 | 0 | 0 |
| o-voxel (T2) | TRELLIS.2 | Cup | 1 | 1 | -107,790 | -107,790 | 0 | 0 | 0 |
| o-voxel (T2) | TRELLIS.2 | Figurine | 2 | 2 | -58,647 | -58,647 | 0 | 0 | 0 |
| o-voxel (T2) | TRELLIS.2 | Gargoyle | 5 | 5 | -104,598 | -104,598 | 0 | 0 | 0 |
| o-voxel (T2) | TRELLIS.2 | Mallard | 1 | 1 | -37,044 | -37,044 | 0 | 0 | 0 |
| o-voxel (T2) | TRELLIS.2 | Mouse | 31 | 31 | -38,527 | -38,527 | 0 | 0 | 0 |

**Means**

| variable | source | objects | comp. init (mean) | comp. final (mean) | quad flips (mean) |
|---|---|---|---|---|---|
| SLAT (T2) | RecGen | 5 | 12 | 208 | 466,103 |
| SLAT (T2) | SAM3D | 6 | 4 | 217 | 383,278 |
| SLAT (T2) | TRELLIS | 6 | 1 | 144 | 338,204 |
| SLAT (T2) | TRELLIS.2 | 6 | 8 | 78 | 841,500 |
| o-voxel (T2) | RecGen | 6 | 10 | 10 | 0 |
| o-voxel (T2) | SAM3D | 6 | 4 | 4 | 0 |
| o-voxel (T2) | TRELLIS | 6 | 1 | 1 | 0 |
| o-voxel (T2) | TRELLIS.2 | 6 | 8 | 8 | 0 |

## A.8 TRELLIS: connected components before and after optimization

| variable | source | object | comp. init | comp. final | faces init | faces final |
|---|---|---|---|---|---|---|
| SLAT (T1) | RecGen | Birdhouse | 14 | 97 | 1,695,728 | 1,697,976 |
| SLAT (T1) | RecGen | Cup | 207 | 264 | 1,951,704 | 1,901,960 |
| SLAT (T1) | RecGen | Figurine | 3 | 151 | 1,332,016 | 1,303,680 |
| SLAT (T1) | RecGen | Gargoyle | 89 | 373 | 1,823,568 | 1,763,568 |
| SLAT (T1) | RecGen | Mallard | 13 | 56 | 566,936 | 537,232 |
| SLAT (T1) | RecGen | Mouse | 13 | 171 | 831,968 | 1,099,384 |
| SLAT (T1) | SAM3D | Birdhouse | 5 | 100 | 1,852,896 | 1,859,328 |
| SLAT (T1) | SAM3D | Cup | 196 | 244 | 1,582,520 | 1,509,456 |
| SLAT (T1) | SAM3D | Figurine | 3 | 104 | 1,551,208 | 1,519,992 |
| SLAT (T1) | SAM3D | Gargoyle | 35 | 215 | 1,345,616 | 1,287,904 |
| SLAT (T1) | SAM3D | Mallard | 1 | 111 | 737,544 | 650,408 |
| SLAT (T1) | SAM3D | Mouse | 32 | 81 | 966,112 | 1,028,576 |
| SLAT (T1) | TRELLIS | Birdhouse | 16 | 72 | 1,765,520 | 1,803,496 |
| SLAT (T1) | TRELLIS | Cup | 340 | 359 | 2,417,248 | 2,440,032 |
| SLAT (T1) | TRELLIS | Figurine | 2 | 124 | 1,478,760 | 1,429,712 |
| SLAT (T1) | TRELLIS | Gargoyle | 12 | 242 | 1,623,768 | 1,557,832 |
| SLAT (T1) | TRELLIS | Mallard | 1 | 43 | 725,176 | 689,200 |
| SLAT (T1) | TRELLIS | Mouse | 1 | 39 | 816,992 | 866,728 |
| SLAT (T1) | TRELLIS.2 | Birdhouse | 43 | 144 | 1,869,320 | 1,901,896 |
| SLAT (T1) | TRELLIS.2 | Cup | 45 | 108 | 2,523,984 | 2,570,592 |
| SLAT (T1) | TRELLIS.2 | Figurine | 9 | 197 | 1,472,560 | 1,511,192 |
| SLAT (T1) | TRELLIS.2 | Gargoyle | 30 | 197 | 1,811,048 | 1,673,848 |
| SLAT (T1) | TRELLIS.2 | Mallard | 2 | 88 | 678,936 | 620,336 |
| SLAT (T1) | TRELLIS.2 | Mouse | 9 | 60 | 769,832 | 843,624 |
| cubefeats (T1) | RecGen | Birdhouse | 13 | 16 | 1,695,344 | 1,726,992 |
| cubefeats (T1) | RecGen | Cup | 208 | 209 | 1,951,464 | 1,947,040 |
| cubefeats (T1) | RecGen | Figurine | 3 | 4 | 1,331,968 | 1,328,528 |
| cubefeats (T1) | RecGen | Gargoyle | 89 | 90 | 1,823,504 | 1,820,144 |
| cubefeats (T1) | RecGen | Mallard | 13 | 13 | 566,992 | 566,472 |
| cubefeats (T1) | RecGen | Mouse | 13 | 14 | 831,784 | 831,056 |
| cubefeats (T1) | SAM3D | Birdhouse | 5 | 6 | 1,852,872 | 1,872,720 |
| cubefeats (T1) | SAM3D | Cup | 196 | 196 | 1,582,536 | 1,581,904 |
| cubefeats (T1) | SAM3D | Figurine | 3 | 3 | 1,551,160 | 1,552,720 |
| cubefeats (T1) | SAM3D | Gargoyle | 35 | 35 | 1,346,096 | 1,347,256 |
| cubefeats (T1) | SAM3D | Mallard | 1 | 1 | 737,496 | 736,832 |
| cubefeats (T1) | SAM3D | Mouse | 32 | 31 | 966,072 | 970,256 |
| cubefeats (T1) | TRELLIS | Birdhouse | 16 | 16 | 1,765,488 | 1,785,944 |
| cubefeats (T1) | TRELLIS | Cup | 340 | 344 | 2,417,272 | 2,413,896 |
| cubefeats (T1) | TRELLIS | Figurine | 2 | 2 | 1,478,712 | 1,479,736 |
| cubefeats (T1) | TRELLIS | Gargoyle | 12 | 12 | 1,623,824 | 1,623,256 |
| cubefeats (T1) | TRELLIS | Mallard | 1 | 1 | 725,096 | 724,816 |
| cubefeats (T1) | TRELLIS | Mouse | 1 | 1 | 816,968 | 817,512 |
| cubefeats (T1) | TRELLIS.2 | Birdhouse | 42 | 39 | 1,869,776 | 1,891,712 |
| cubefeats (T1) | TRELLIS.2 | Cup | 47 | 47 | 2,523,920 | 2,510,496 |
| cubefeats (T1) | TRELLIS.2 | Figurine | 9 | 5 | 1,473,248 | 1,467,912 |
| cubefeats (T1) | TRELLIS.2 | Gargoyle | 30 | 17 | 1,811,536 | 1,794,944 |
| cubefeats (T1) | TRELLIS.2 | Mallard | 2 | 1 | 678,904 | 677,904 |
| cubefeats (T1) | TRELLIS.2 | Mouse | 9 | 7 | 769,936 | 766,544 |

**Means**

| variable | source | objects | comp. init (mean) | comp. final (mean) |
|---|---|---|---|---|
| SLAT (T1) | RecGen | 6 | 56 | 185 |
| SLAT (T1) | SAM3D | 6 | 45 | 142 |
| SLAT (T1) | TRELLIS | 6 | 62 | 146 |
| SLAT (T1) | TRELLIS.2 | 6 | 23 | 132 |
| cubefeats (T1) | RecGen | 6 | 56 | 58 |
| cubefeats (T1) | SAM3D | 6 | 45 | 45 |
| cubefeats (T1) | TRELLIS | 6 | 62 | 63 |
| cubefeats (T1) | TRELLIS.2 | 6 | 23 | 19 |

## A.9 Decoder noise floor (TRELLIS.2 SLAT runs)

The same latent decoded twice, so gate changes in A.7 are not decoder non-determinism.

| source | object | voxels added | voxels removed | quad flips |
|---|---|---|---|---|
| RecGen | Birdhouse | 0 | 0 | 0 |
| RecGen | Figurine | 0 | 0 | 0 |
| RecGen | Gargoyle | 0 | 0 | 0 |
| RecGen | Mallard | 0 | 0 | 0 |
| RecGen | Mouse | 0 | 0 | 0 |
| SAM3D | Birdhouse | 0 | 0 | 0 |
| SAM3D | Cup | 0 | 0 | 0 |
| SAM3D | Figurine | 0 | 0 | 0 |
| SAM3D | Gargoyle | 0 | 0 | 0 |
| SAM3D | Mallard | 0 | 0 | 0 |
| SAM3D | Mouse | 0 | 0 | 0 |
| TRELLIS | Birdhouse | 0 | 0 | 0 |
| TRELLIS | Cup | 0 | 0 | 0 |
| TRELLIS | Figurine | 0 | 0 | 0 |
| TRELLIS | Gargoyle | 0 | 0 | 0 |
| TRELLIS | Mallard | 0 | 0 | 0 |
| TRELLIS | Mouse | 0 | 0 | 0 |
| TRELLIS.2 | Birdhouse | 0 | 0 | 0 |
| TRELLIS.2 | Cup | 0 | 0 | 0 |
| TRELLIS.2 | Figurine | 0 | 0 | 0 |
| TRELLIS.2 | Gargoyle | 0 | 0 | 0 |
| TRELLIS.2 | Mallard | 0 | 0 | 0 |
| TRELLIS.2 | Mouse | 0 | 0 | 0 |
