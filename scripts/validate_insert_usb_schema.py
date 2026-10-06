"""Validate a raw handmade-task episode against a real Insert_USB HDF5.

Episode length, task actor names and fixed-string storage widths may differ.
All other saved field paths, numeric dtypes and per-frame shapes must match.
This checks raw collection files, before ACT's shared data conversion.
"""
import argparse
import json
from pathlib import Path
import cv2
import h5py
import numpy as np


def datasets(handle):
    found={}
    handle.visititems(lambda name,obj: found.update({name:obj}) if isinstance(obj,h5py.Dataset) else None)
    return found


def validate(reference,episode, *, strict_terminal=False, metadata=None):
    errors=[];images=0
    with h5py.File(reference,'r') as ref,h5py.File(episode,'r') as data:
        expected=datasets(ref);actual=datasets(data)
        n=len(data['step']) if 'step' in data else 0
        common_expected={k for k in expected if not k.startswith('actor/')}
        common_actual={k for k in actual if not k.startswith('actor/')}
        for key in sorted(common_expected-common_actual):errors.append('missing field: '+key)
        for key in sorted(common_actual-common_expected):errors.append('unexpected field: '+key)
        actors=[k for k in actual if k.startswith('actor/')]
        if not actors:errors.append('missing task actor poses')
        reference_actor=next(v for k,v in expected.items() if k.startswith('actor/'))
        for key,value in actual.items():
            prototype=reference_actor if key.startswith('actor/') else expected.get(key)
            if prototype is None:continue
            if value.shape!=(n,*prototype.shape[1:]):
                errors.append(f'{key}: shape {value.shape} must be {(n,*prototype.shape[1:])}')
            string=prototype.dtype.kind=='S'
            if string:
                if value.dtype.kind!='S':errors.append(f'{key}: expected fixed bytes, got {value.dtype}')
            elif value.dtype!=prototype.dtype:
                errors.append(f'{key}: dtype {value.dtype} must be {prototype.dtype}')
            if value.dtype.kind=='f':
                for offset in range(0,n,16):
                    if not np.isfinite(value[offset:offset+16]).all():
                        errors.append(f'{key}: nonfinite numeric values');break
            if key.endswith(('/rgb','/rgb_marker')) and len(value):
                exemplar=cv2.imdecode(np.frombuffer(prototype[0],np.uint8),cv2.IMREAD_COLOR)
                for index,encoded in enumerate(value):
                    frame=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR)
                    if frame is None or frame.shape!=exemplar.shape:
                        errors.append(f'{key}[{index}]: JPEG does not match {exemplar.shape}');break
                    images+=1
        for key in ref.attrs:
            if key not in data.attrs:errors.append('missing root attribute: '+key)
        if 'phase' in data:
            for key in ref['phase'].attrs:
                if key not in data['phase'].attrs:errors.append('missing phase attribute: '+key)
            phases=np.asarray(data['phase/id'])
            if not np.isin(phases,[0,1,2]).all() or np.any(np.diff(phases)<0):
                errors.append('invalid phase sequence')
        if n<2:errors.append('episode has fewer than two observations')
        elif np.any(np.diff(np.asarray(data['step']))<=0):
            errors.append('step field must increase strictly')
        if strict_terminal:
            errors.extend(validate_phase_contract(data, require_terminal=True, metadata=metadata))
        result=dict(valid=not errors,episode=str(episode),reference=str(reference),
                    frames=n,decoded_images=images,actor_names=actors,errors=errors)
    return result


def validate_phase_contract(data, *, require_terminal=False, metadata=None):
    errors = []
    required = ["step", "phase/id", "phase/name", "phase/sim_step", "phase/policy_step", "phase/is_boundary"]
    if any(key not in data for key in required):
        return ["phase contract cannot be checked: required fields are missing"]
    step = np.asarray(data["step"])
    phase = np.asarray(data["phase/id"])
    if not len(step):
        return ["phase contract requires observations"]
    attrs = data["phase"].attrs
    if not np.array_equal(step, np.asarray(data["phase/sim_step"])):
        errors.append("phase/sim_step differs from step")
    names = {0: "pre_move", 1: "policy", 2: "terminal"}
    # The canonical BaseTask phase name for id 1 is policy.
    for identifier, name in names.items():
        values = np.asarray(data["phase/name"])[phase == identifier]
        if any(value.decode() != name for value in values):
            errors.append(f"phase/name inconsistent with id {identifier}")
    counts = {identifier: int(np.sum(phase == identifier)) for identifier in names}
    for key, identifier in [("pre_move_saved_frames", 0), ("policy_saved_frames", 1),
                            ("action_saved_frames", 1), ("terminal_saved_frames", 2)]:
        if int(attrs.get(key, -1)) != counts[identifier]:
            errors.append(f"{key} does not match actual phase frames")
    boundaries = np.r_[True, phase[1:] != phase[:-1]].astype(np.int64)
    if not np.array_equal(boundaries, np.asarray(data["phase/is_boundary"])):
        errors.append("phase/is_boundary does not match phase transitions")
    policy = np.flatnonzero(phase == 1)
    first_policy = int(policy[0]) if len(policy) else -1
    if int(attrs.get("policy_start_saved_index", -1)) != first_policy:
        errors.append("policy_start_saved_index does not match the first policy frame")
    if len(policy) > 1 and require_terminal:
        stride = int(attrs.get("save_frequency", 0))
        if stride <= 0 or not np.all(np.diff(step[policy]) == stride):
            errors.append("policy observations are not uniformly sampled at save_frequency")
    policy_step = np.asarray(data["phase/policy_step"])
    if not np.all(policy_step[phase == 0] == -1):
        errors.append("pre_move policy_step must be -1")
    reason = str(attrs.get("terminal_reason", ""))
    if require_terminal:
        if phase[-1] != 2 or counts[2] != 1:
            errors.append("strict episode must end in exactly one actual terminal observation")
        if not reason or reason == "terminal":
            errors.append("strict terminal_reason is missing or ambiguous")
        start = int(attrs.get("policy_start_sim_step", -1))
        if not np.array_equal(policy_step[phase != 0], step[phase != 0] - start):
            errors.append("policy_step must reflect actual elapsed physics steps")
        if metadata is not None:
            diagnostics = metadata.get("success_diagnostics", {})
            if reason != metadata.get("terminal_reason") or reason != diagnostics.get("terminal_reason"):
                errors.append("HDF5 and metadata terminal reasons disagree")
            if (reason == "success") != bool(diagnostics.get("success", False)):
                errors.append("success flag and terminal reason disagree")
            if int(diagnostics.get("physical_step", -1)) != int(step[-1]):
                errors.append("terminal diagnostics do not describe the last physical step")
    return errors


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('reference',type=Path)
    parser.add_argument('episodes',nargs='+',type=Path)
    parser.add_argument('--strict-terminal', action='store_true')
    args=parser.parse_args()
    reports=[validate(args.reference,p,strict_terminal=args.strict_terminal) for p in args.episodes]
    print(json.dumps(reports,indent=2))
    raise SystemExit(0 if all(r['valid'] for r in reports) else 1)
