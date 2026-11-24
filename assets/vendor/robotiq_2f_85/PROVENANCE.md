# Robotiq 2F-85 (vendored, unmodified)

Upstream: https://github.com/robotiq/isaacsim_assets
Path:     `grippers/Robotiq_2F_85/`
Commit:   `646e8a5dc0ff2a52ff8fa4bb4310c89484390129` (main, 2026-11-11)
License:  CC-BY-4.0 — see `PACKAGE-LICENSES/LICENSE` (Robotiq attribution kept intact)

## Why these files and only these

The shipped asset is used as `Physics = "Physx_parallel_grip"` + `Fingertip = "Standard"`,
which is the default. That variant composes the following payloads/layers; every
other file in the upstream package (compliant/Newton variants, tactile variants,
other grippers, demos, CI) is unused and is not vendored.

```
Robotiq_2F_85.usda                              root (defaultPrim /Robotiq_2F_85)
configuration/Robotiq_2F_85_robot.usda          sublayer: IsaacRobotAPI + joint/link lists
payloads/base.usda                              link tree, visuals, grip_frame
payloads/geometries.usd                         named-geometry slot (empty by design)
payloads/instances.usda                         instancing slot (empty by design)
payloads/Robotiq_2F_85_kinematics_physics.usda  neutral joint frames, limits, root_joint
payloads/Robotiq_2F_85_physx_common_physics.usda PhysX articulation + link colliders
payloads/Robotiq_2F_85_parallel_grip_physics.usda finger_joint drive + mimic table
payloads/Robotiq_2F_85_parallel_grip_dowel_pins.usda  visual dowel pins
payloads/Robotiq_2F_85_body_mass_physics.usda   base_link/knuckle mass
payloads/Robotiq_2F_85_fingertip_physics.usda   fingertip bodies + welds
materials/materials.usd                         sublayered by every parts/*.usd
parts/Defeatured_2F_85_PAD_OPEN_*.usd           the six CAD parts (git-LFS objects)
parts/Robotiq_2F_85_dowel_pin.usda              dowel pin part
PACKAGE-LICENSES/LICENSE                        CC-BY-4.0 text
```

## Byte-for-byte copy

Nothing was edited, so `sha256sum` is a complete provenance check. The two
layered text files carry no `#usda` header of our own on purpose: a modified
vendored file cannot be re-diffed against upstream, and the `.usd` parts are
binary (a header comment is not expressible). Read this file for provenance.

```
f60288330aedcd1682dcef7b9d50a5de914a7be818039a26b5da956effe2e3b6  ./configuration/Robotiq_2F_85_robot.usda
48eca813596e78486469edc70f825c9bb9913836f63e4007559920820e8e9d9d  ./materials/materials.usd
9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411  ./PACKAGE-LICENSES/LICENSE
a9082107ce6d92b69945ca81f22b271c24c5fba338655e6b838c517d81039826  ./parts/Defeatured_2F_85_PAD_OPEN_basestep_JFX.usd
1d8f971e8d9e6b140ae406adbe4ebf17831c8c298ce3ce07577b7fac3590bb8c  ./parts/Defeatured_2F_85_PAD_OPEN_Finger1step_JFT.usd
82d1355f767006bdcb1376e8fd495c222120b18776d6d9cccdb199c0ef3d240d  ./parts/Defeatured_2F_85_PAD_OPEN_finger2step_JFP.usd
2daaae8c788e7e74fa54a5850854b4dcd2ee6d739a2873db63c1fc5db157e87c  ./parts/Defeatured_2F_85_PAD_OPEN_finger3step_JFL.usd
8d7e3fce9d1c70ab65a676f3efdfa5fd1d161a66793a36bed408509e33b64fe5  ./parts/Defeatured_2F_85_PAD_OPEN_finger4step_JFH.usd
700409224e3d2c4e0a0e2ba4b4da4f644babd74207d660c034ce53b255b3e876  ./parts/Defeatured_2F_85_PAD_OPEN_fingertipsstep_JFD.usd
87bfc3752e993f8c89bc8a4218dda93918f6dca793b5c6d6441a4bbe3d904ce5  ./parts/Robotiq_2F_85_dowel_pin.usda
aad894840730a45edd7815b8ca8699cb9252791683af8e98d28fb2e604db2d71  ./payloads/base.usda
43f1718720ea76cf8c69d5031de8d215806c79a54618417c431c0f17fe0758f3  ./payloads/geometries.usd
f3354daa0ef1d865ac397413b732636e56d9affc5ce48240a174b30bea5309f6  ./payloads/instances.usda
21e46529d377d8e300d4a4e6bd3e09d9c40d1b905e3a3dfce27fb982453699db  ./payloads/Robotiq_2F_85_body_mass_physics.usda
ac7e592beb00e3263a9bae376a8c3cd49daa98986743c39c4bb02c004b8b475d  ./payloads/Robotiq_2F_85_fingertip_physics.usda
0ccc798f315e6e2b4baa7c11abfa9f58906d0c0462d3b404ea35560e135bb8a1  ./payloads/Robotiq_2F_85_kinematics_physics.usda
c4df764c154ef6086122d36694dfa4956c2f78b73ea4cf2f9749127a24a95854  ./payloads/Robotiq_2F_85_parallel_grip_dowel_pins.usda
0f7e98ea8d31f584121a9cc211351b3042ce451dcf898c383d470fe05c479b99  ./payloads/Robotiq_2F_85_parallel_grip_physics.usda
034630ddb2f47d1f1f65fa22c37031aadfa5395810754a211f19b80851566bd0  ./payloads/Robotiq_2F_85_physx_common_physics.usda
483b4b8e0dd237f9a5eaf92578f1b9b2d8a8e51bd65a5c662500bd6d780acd43  ./Robotiq_2F_85.usda
```
