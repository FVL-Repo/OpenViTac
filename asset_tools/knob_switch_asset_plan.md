# Knob Switch Asset Plan

目标是生成一个简化版“假旋转”旋钮开关资产，用于后续转换成带 tet 的 USD，并能作为 UIPC 物体稳定加载。由于当前 OpenViTac 任务里不打算做 joint，旋钮拆成两个独立资产：

- 固定底座：类似 slot，是静止的承载件，包含一个方形底板和中间竖直圆柱轴。
- 旋钮套子：类似一个套在圆柱轴外面的 knob cap，被夹爪抓住后通过 task 里的 motion plan 假装旋转。

第一版优先保证几何简单、水密、单连通、原点清楚，不追求复杂外观细节。

## 坐标约定

- 单位：脚本内部先用 mm 建模，导出前统一 `apply_scale(0.001)` 转成 m。
- 旋钮轴线：沿 `+Z`。
- 资产最低点：放在 `z = 0`，便于在场景里用 pose.z 控制接触桌面或底座。
- 原点：放在底部中心，也就是 `x = 0, y = 0, z = 0`。
- 正面指示方向：默认指向 `+X`，后续 task 里可以通过 yaw 控制旋钮初始角度。

## 固定底座几何

固定底座建议单独导出为：

```text
assets/objects/knob_switch_base.obj
```

几何组成：

1. 方形底板
   - 尺寸：`80 mm x 80 mm x 5 mm`。
   - 位置：底面 `z = 0`，顶面 `z = 5 mm`。
   - 作用：固定底座，类似 slot 的静止物体。

2. 中心立柱
   - 几何体：圆柱。
   - 尺寸：直径 `20 mm`，高度 `17 mm`。
   - 位置：立在底板中心，底面 `z = 5 mm`，顶面 `z = 22 mm`。
   - 作用：给旋钮套子提供视觉上的转轴/定位结构。

3. 底板圆角
   - 需求：xy 平面四个角做圆角，圆角直径 `8 mm`，也就是半径 `4 mm`。
   - 推荐做法：用 2D rounded rectangle 截面 extrude 成 3D 底板，而不是对 box 做 3D bevel。
   - 原因：只需要 xy 平面的四个角圆，不需要上下边缘也倒角；2D 外轮廓挤出更可控，也更容易保持水密。

底座必须是一个水密单体。底板和中心圆柱可以 boolean union，圆柱底部和底板顶面最好有 `0.2-0.5 mm` 的重叠，避免只是相切导致 union 失败。

## 旋钮套子几何

旋钮套子建议单独导出为：

```text
assets/objects/knob_switch_cap.obj
```

第一版可以做成一个厚壁圆柱套：

1. 外圆柱
   - 尺寸建议：外径 `23 mm`，高度 `20 mm`。
   - 底面可以放在 `z = 5 mm` 或 `z = 6 mm`，让它看起来套在底座立柱上。

2. 中心盲孔或通孔
   - 中心孔直径必须大于底座立柱直径。
   - 立柱直径是 `20 mm`，建议孔直径 `22-24 mm`，给 UIPC 接触留间隙。
   - 如果做盲孔，孔深建议 `16-18 mm`；如果做通孔，几何更简单，但视觉上不一定像真实旋钮。

3. 顶部指示条
   - 几何体：小长方体凸起。
   - 尺寸建议：长 `18 mm`，宽 `3 mm`，高 `2 mm`。
   - 方向：沿 `+X`，用于显示旋钮角度。

旋钮套子是运动物体，和底座不是同一个 mesh。任务里“旋转”时，本质是抓住 `knob_switch_cap` 绕底座中心轴运动；底座保持静止。这样不需要 joint，但视觉和接触上可以近似一个可旋转旋钮。

## 为什么不先做凹槽

凹槽、刻线、十字槽这类细节通常需要 boolean difference。difference 也可以保持水密，但更容易产生非流形边、小碎面、极薄三角形。UIPC tet 转换时更关心体网格质量，所以第一版用“凸起指示条”替代“凹进去的刻线”。

## lead 是什么

你给的 slot 示例里：

```python
main = trimesh.creation.box([hw, ht, 2 * D])
main.apply_translation([0, 0, H])

lead = trimesh.creation.box([hw + 2 * LEAD, ht + 2 * LEAD, 3.0])
lead.apply_translation([0, 0, H])

slot = trimesh.boolean.difference([outer, main, lead])
```

这里的 `lead` 不是“圆角”。它是在入口处额外减掉一个更宽、更浅的 box，相当于给孔口做一个导入扩口/入口倒角的近似。效果是：

- `main` 切出主要孔道；
- `lead` 在孔口附近切出更大的浅层开口；
- 插头刚接触孔口时更容易进入，不会被锐利直角边卡住。

严格的倒角应该是斜面；这个 `lead` 写法更像“台阶式导入口”。它对插入任务有用，但不是 xy 平面四角圆角。底座四角圆角应该用 rounded rectangle extrusion 或 polygon offset 来做。

## 水密单体策略

固定底座推荐使用 `trimesh` primitives + boolean union：

```python
base = rounded_box_xy(width=80, depth=80, height=5, radius=0.5)
post = trimesh.creation.cylinder(radius=10, height=17, sections=64)
post.apply_translation([0, 0, 5 + 17 / 2 - 0.2])

# 平移到预期位置后：
mesh = trimesh.boolean.union([base, post])
```

旋钮套子推荐使用 boolean difference + union：

```python
outer = trimesh.creation.cylinder(radius=15, height=20, sections=64)
hole = trimesh.creation.cylinder(radius=11.5, height=24, sections=64)
cap = trimesh.boolean.difference([outer, hole])

indicator = trimesh.creation.box(extents=[18, 3, 2])
cap = trimesh.boolean.union([cap, indicator])
```

生成后必须做检查：

```python
mesh.remove_duplicate_faces()
mesh.remove_degenerate_faces()
mesh.remove_unreferenced_vertices()
mesh.fix_normals()

assert mesh.is_watertight
assert len(mesh.split(only_watertight=False)) == 1
assert mesh.volume > 0
```

如果 boolean 后不是单体，说明某些几何只是相切或没有足够重叠。解决方式是让相邻部件有 `0.2-1.0 mm` 的重叠，例如中心立柱底部嵌入底板 `0.2 mm`，顶部指示条嵌入旋钮顶部 `0.2-0.5 mm`。

## 导出文件

建议脚本名：

```text
scripts/asset_tools/build_knob_switch.py
```

默认输出：

```text
assets/objects/knob_switch_base.obj
assets/objects/knob_switch_base.mtl
assets/objects/knob_switch_cap.obj
assets/objects/knob_switch_cap.mtl
```

如果后续希望材质分区明显，可以先导出单色 OBJ，等 USD 转换后再用 USD material binding 或单独 face groups 做颜色。第一版先避免多材质带来的面分组复杂度。

## 后续转换

生成 OBJ 后，沿用已有 convert 流程：

```bash
python scripts/convert.py assets/objects/knob_switch_base.obj assets/objects/knob_switch_base.usd
python scripts/convert.py assets/objects/knob_switch_cap.obj assets/objects/knob_switch_cap.usd
```

转换前建议先打印：

- `mesh.extents`
- `mesh.bounds`
- `mesh.is_watertight`
- `mesh.euler_number`
- connected components 数量

## 为什么这个写法会比当前 build_usb 简单

你给的 USB 示例短，是因为它使用了 `trimesh.creation.box` 和 `trimesh.boolean.union`。也就是说：

- 顶点和面由 trimesh 自动生成；
- 相交部件的内部面由 boolean union 自动删除；
- 水密性主要交给 boolean 后端保证；
- OBJ 导出也由 trimesh 处理。

当前 `build_usb_0_obj.py` 和 `build_usb_slot_0_obj.py` 更长，是因为它们手工构造网格：

- 手写坐标分割线 `xs/ys/zs`；
- 用 `occupied()` 判断每个小体素是否属于实体；
- 只给实体和空体之间的边界生成面；
- 手动写 OBJ 顶点、面、MTL 材质。

手写版本更啰嗦，但优点是确定性强、不依赖 boolean 后端，适合 USB slot 这种带盲孔的网格。trimesh boolean 版本更短，但需要可用的 boolean backend，例如 `manifold3d`、Blender 或 OpenSCAD；如果 backend 不可用或输入几何刚好相切，可能会失败或产生非水密网格。
