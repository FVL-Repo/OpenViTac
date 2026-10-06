"""生成 USB 配套底座(带孔 + 导入倒角)。孔 = 插口截面 + 单边间隙。
用法: python scripts/asset_tools/build_usb_slot.py [out.obj]   (默认 /tmp/usb_assets/USBSlot.obj)
"""
import sys, os, trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/usb_assets/USBSlot.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

CW, CT = 12.0, 4.5            # 插口截面 (mm)
CLR = 1.5                     # 单边间隙: 必须 > IPC 的 d_hat(0.5mm), 否则插入会卡死
hw, ht = CW + 2 * CLR, CT + 2 * CLR
H, D = 13.0, 11.0            # 底座高 / 孔深
LEAD = 0.8                    # 导入倒角: 顶部加宽段单边 0.8mm, 深 1.5mm

outer = trimesh.creation.box([26, 16, H]);              outer.apply_translation([0, 0, H / 2])
main = trimesh.creation.box([hw, ht, 2 * D]);           main.apply_translation([0, 0, H])
lead = trimesh.creation.box([hw + 2 * LEAD, ht + 2 * LEAD, 3.0]); lead.apply_translation([0, 0, H])
slot = trimesh.boolean.difference([outer, main, lead])
slot.apply_scale(0.001)
slot.export(out)
print(f"saved {out}  hole(mm)={hw}x{ht} depth={D}  watertight={slot.is_watertight}")
