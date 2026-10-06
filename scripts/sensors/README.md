# Sensor Tools

This directory contains sensor-specific diagnostics and offline post-processing scripts. General task collection and policy evaluation entry points remain directly under `scripts/`.

## Layout

- `xsense/`: XSense marker, physical-chain, and FEM-layer audits.
- `gelsight/`: GelSight/FEM force-field and gel-particle post-processing.
- `collect_contact.py`: General contact-data diagnostic collector shared by tactile backends.

Run scripts from the repository root, for example:

```bash
python scripts/sensors/xsense/audit_marker_tasks.py OUTPUT_DIR task=EPISODE.hdf5
python scripts/sensors/gelsight/contact_force_to_field.py RUN_DIR_OR_HDF5
```
