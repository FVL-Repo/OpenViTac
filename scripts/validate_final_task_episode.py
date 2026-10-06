"""Check final-task cadence, physical verdict and raw Insert_USB compatibility."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
try:
    from .validate_insert_usb_schema import validate, validate_phase_contract
except ImportError:  # Direct script invocation.
    from validate_insert_usb_schema import validate, validate_phase_contract

TASK_SECONDS={"grasp_fragile_chip":60,"tension_strap":60,
              "bulb_tightening":180,"wipe_vase":160}

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
        if task not in TASK_SECONDS:errors.append("not one of the four final tasks")
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
        # legacy counter as physical ticks for the three BaseTask exporters.
        if task=="grasp_fragile_chip" and not np.array_equal(policy_counter,steps[phases!=0]-start):
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
        if task=="grasp_fragile_chip":
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
        elif task=="tension_strap":
            physical=meta.get("strap_acceptance",{})
            if bool(meta.get("strap_accepted"))!=accepted or bool(physical.get("success"))!=accepted:
                errors.append("strap verdict differs from independent two-stage scorer")
            for key,value in {"target_sequence_N":[12.,18.],"tolerance_N":.5,
                    "hold_seconds_per_stage":3.,"timeout_seconds":60.}.items():
                if physical.get(key)!=value:errors.append("strap acceptance mismatch: "+key)
            if accepted and (physical.get("stage_index")!=2 or len(physical.get("completed_steps",[]))!=2):
                errors.append("strap success did not complete both stages")
        elif task=="wipe_vase":
            physical=meta.get("vase_wiping_final",{})
            required_clean=meta.get("required_clean_fraction",.95)
            configured_clean=meta.get("vase_wiping_parameters",{}).get("required_clean_fraction",.95)
            if required_clean not in (.925,.95) or configured_clean!=required_clean:
                errors.append("vase cleaning threshold is unsupported or differs from its recorded configuration")
            if physical.get("required_clean_fraction",required_clean)!=required_clean:
                errors.append("vase physical cleaning threshold differs from episode metadata")
            if (physical.get("failure") is None)!=accepted:
                errors.append("vase failure differs from terminal verdict")
            if accepted and not (physical.get("cleaned_fraction",0)>=required_clean
                    and physical.get("pressure_band_fraction",0)>=.95
                    and physical.get("peak_normal_force_N",1e6)<=12.5
                    and not meta.get("glaze_damage")):
                errors.append("vase success violates cleaning, force or glaze gate")
            if not np.isclose(meta.get("stain_area",{}).get("ratio",0),1.10,atol=1e-5):
                errors.append("initial stain area is not same-seed baseline plus10percent")
        elif task=="bulb_tightening":
            physical=meta.get("bulb_final",{})
            if bool(meta.get("bulb_accepted"))!=accepted or bool(meta.get("bulb_failure"))==accepted:
                errors.append("bulb verdict differs between metadata and HDF5")
            if accepted:
                if (meta.get("bulb_submission_source")!="stationary_joint_command"
                        or meta.get("bulb_submission_stationary_steps",0)<180):
                    errors.append("bulb expert did not submit through the same public stationary END")
                hand=meta.get("thread_physical_spec",{}).get("handedness",0)
                if not (physical.get("probe_complete") and 1<=physical.get("formal_net_turns",0)<=3
                        and meta.get("bulb_hold_steps",0)>=132
                        and abs(physical.get("torque_Nm",0)-.12)<=.012
                        and physical.get("signed_torque_Nm",0)*hand>0
                        and physical.get("formal_advance_m",physical.get("advance_m",0))>=
                            (.055-.006-meta.get("thread_physical_spec",{}).get("initial_depth_m",.014)
                             -meta.get("thread_physical_spec",{}).get("seat_top_m",.022)-.002)
                        and physical.get("phase_error_m",1)<.002
                        and physical.get("center_drift_m",1)<.002 and physical.get("tilt_deg",99)<6
                        and physical.get("seat_force_N",0)>3):
                    errors.append("bulb success violates probing, actual turns, torque or seating gate")
                light=meta.get("bulb_light",{})
                if light.get("phase")!="terminal" or light.get("glass_emission") is not False:
                    errors.append("successful bulb did not use terminal-only filament lighting")
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
