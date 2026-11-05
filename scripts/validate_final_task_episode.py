"""Check grasp_chip cadence, physical verdict and raw Insert_USB compatibility."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
try:
    from .validate_insert_usb_schema import validate, validate_phase_contract
except ImportError:  # Direct script invocation.
    from validate_insert_usb_schema import validate, validate_phase_contract

TASK_SECONDS={"grasp_chip":60}

def validate_final(episode,reference=None,expect=None):
    episode=Path(episode);errors=[];raw=None
    if reference is not None:
        raw=validate(reference,episode)
        errors.extend(raw["errors"])
    with h5py.File(episode,"r") as data:
        context=json.loads(data.attrs["episode_context_json"])
        task=context["task"];timing=context.get("timing_contract",{})
        metadata_path=episode.parent.parent/"metadata.json"
        meta=json.loads(metadata_path.read_text())[str(context["seed"])]
        if task not in TASK_SECONDS:errors.append("not a grasp_chip episode")
        errors.extend(validate_phase_contract(data))
        for key,wanted in {"physics_dt_s":1/120,"decimation":1,"save_frequency":2,
                "policy_observation_dt_s":1/60,"policy_action_repeat":2,
                "policy_action_dt_s":1/60,"video_frequency":2,"video_fps":60}.items():
            if not np.isclose(timing.get(key,-1),wanted,atol=1e-10,rtol=0):
                errors.append("timing mismatch: "+key)
        for key,wanted in {"quaternion_order":"wxyz","position_units":"meters",
                "pose_frame":"world","zero_velocity_targets":True}.items():
            if timing.get(key)!=wanted:errors.append("action/pose contract mismatch: "+key)
        phases=data["phase/id"][:];steps=data["step"][:]
        policy=np.flatnonzero(phases==1);terminal=np.flatnonzero(phases==2)
        reason=str(data["phase"].attrs.get("terminal_reason",""))
        start=int(data["phase"].attrs.get("policy_start_sim_step",-1))
        if len(policy)<2 or not np.all(np.diff(steps[policy])==2):
            errors.append("POLICY must be a uniform60Hz sequence")
        if not len(terminal) or phases[-1]!=2:errors.append("actual terminal observations are missing")
        if not reason or reason=="terminal":errors.append("terminal reason is ambiguous")
        policy_counter=data["phase/policy_step"][:][phases!=0]
        if np.any(policy_counter<0) or np.any(np.diff(policy_counter)<0):
            errors.append("policy_step must be a nonnegative monotonic policy counter")
        # The existing Insert_USB expert leaves its policy-loop counter at0.
        # Physical time is always step/phase.sim_step; do not reinterpret that
        # legacy counter as physical ticks for BaseTask exporters.
        if task=="grasp_chip" and not np.array_equal(policy_counter,steps[phases!=0]-start):
            errors.append("chip elapsed-physics policy counter is inconsistent")
        if len(terminal) and (int(steps[terminal[0]])-start)/120>TASK_SECONDS.get(task,0)+2/120:
            errors.append("physical verdict exceeds the declared policy timeout")
        accepted=reason=="success"
        if meta.get("result") not in ("success","fail"):
            errors.append("collector result is missing")
        elif (meta["result"]=="success")!=accepted:
            errors.append("collector result differs from terminal reason")
        if expect is not None and accepted!=(expect=="success"):
            errors.append("physical result differs from requested expectation")
        physical={}
        if task=="grasp_chip":
            physical=meta.get("success_diagnostics",{})
            if physical.get("terminal_reason")!=reason or bool(physical.get("success"))!=accepted:
                errors.append("chip verdict differs between HDF5 and physical scorer")
            if not len(terminal) or physical.get("physical_step")!=int(steps[terminal[0]]):
                errors.append("chip physical verdict must match the first actual terminal observation")
            if accepted and len(terminal)!=1:
                errors.append("successful chip must freeze at one actual terminal endpoint")
            if not accepted:
                transitions=[row for row in physical.get("transitions",[]) if row.get("stage")=="terminal"]
                if transitions and len(terminal) and transitions[0].get("step")!=int(steps[terminal[0]]):
                    errors.append("chip failure aftermath was recorded as POLICY instead of TERMINAL")
            if accepted and not all(physical.get(k,False) for k in ("grasp_verified","lift_verified","release_supported")):
                errors.append("successful chip is missing a required physical stage")
        return dict(valid=not errors,task=task,seed=context["seed"],episode=str(episode),
            physical_outcome="success" if accepted else "failure",terminal_reason=reason,
            frames=len(steps),policy_frames=len(policy),terminal_frames=len(terminal),
            first_terminal_physical_step=None if not len(terminal) else int(steps[terminal[0]]),
            raw_schema=raw,physical=physical,errors=errors)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes",nargs="+",type=Path)
    parser.add_argument("--reference",type=Path)
    parser.add_argument("--expect",choices=("success","failure"))
    args=parser.parse_args()
    reports=[validate_final(path,args.reference,args.expect) for path in args.episodes]
    print(json.dumps(reports,indent=2))
    raise SystemExit(0 if all(row["valid"] for row in reports) else 1)
