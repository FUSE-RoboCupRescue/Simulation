import rowan
import numpy as np

wheel_count = 10
base_dims = [.5, .3, .1]
flipper_wheel_count = 8
flipper_dims = [base_dims[0]*.8, .05, base_dims[2]*.7]
wheel_space = base_dims[0] / wheel_count
wheel_diameter = wheel_space * .9
wheel_depth = .1

flipper_dims[0] = wheel_space * flipper_wheel_count
flipper_dims[2] = base_dims[2]

wheel_max_torque = 10
flipper_max_torque = 100

obstacle_color = [.8, .8, .0]


def obstacles():
    x_start = 20
    x_layer_count = 50
    y_layer_count = 10
    text = ""
    for i in range(x_layer_count):
        for j in range(y_layer_count):
            pos_x = x_start + i-x_layer_count*.5
            pos_y = j-y_layer_count*.5
            pos_z = np.random.random()*.1
            text += f"""<geom type="box" size=".5 .5 .5" rgba="{obstacle_color} 1" pos ="{pos_x} {pos_y} {pos_z}"/>"""
    return text.replace(",", "").replace("[", "").replace("]", "")


def begin(timestep: float=.002):
    text = f"""<mujoco model="wheely">
    <size njmax="10000" nconmax="5000"/>
    <option timestep="{timestep}" />
    <option gravity="0 0 -9.81"/>
    <default>
        <geom size="0.1" mass="1" friction="5"/>
    </default>
    <visual>
        <quality shadowsize="{2**12}"/>
    </visual>
    <asset>
        <material name="shiny_material" shininess="0.9" reflectance="0.3" specular="1"/>
    </asset>

    <worldbody>
        <!--<geom type="box" size="10 10 0.1" rgba="1 1 0 1"/>-->
        <body name="goal" pos="40 0 3">
            <geom type="box" size=".5 .5 .5" rgba="1 0 0 0.5"/>
        </body>
        {obstacles()}
        """
    light_count = 3
    for i in range(light_count):
        text += f"""<light name="sunlight{i}" pos="{i*(50/light_count)} 10 10" dir="0 -1 -1" diffuse="1 1 1" specular="0.9 0.9 0.9" ambient="0.2 0.2 0.2" castshadow="true" />"""
    text += f"""
        <!-- Car chassis -->
        <body name="chassis" pos="0 0 2">
            <joint name="chasis_joint" type="free"/>

            <geom type="box" size="{base_dims[0]} {base_dims[1]} {base_dims[2]}" rgba="0.3 0.3 0.3 1" material="shiny_material"/>
        """
    return text


def end():
    return """    </actuator>
</mujoco>"""


def make_flipper(idx, pos, offset, front):
    text = f"""<body name="flipper{idx}" pos="{pos[0]} {pos[1]} {pos[2]}">
                <joint name="jointf{idx}" stiffness="100" damping="1"  type="hinge" axis="0 1 0"/>
                <body name="flipper_piece{idx}" pos="{offset} 0 0">
                    <geom type="box" pos="{-flipper_dims[0]*.5 if front else flipper_dims[0]*.5} 0 0" size="{flipper_dims[0]*.5} {flipper_dims[1]*.5} {flipper_dims[2]*.3}" rgba="0.3 0.3 0.3 1" material="shiny_material"/>""".replace(",", "").replace("[", "").replace("]", "")
    z_start = -wheel_space
    z_end = -flipper_dims[2] - wheel_space
    for i in range(flipper_wheel_count):
        pos_x = (flipper_dims[0]/flipper_wheel_count)*i*2 - flipper_dims[0] + wheel_diameter
        if front:
            pos_z = (z_end-z_start) * (flipper_wheel_count-i)/flipper_wheel_count + z_start
        else:
            pos_z = (z_end-z_start) * i/flipper_wheel_count + z_start
        text += f"""<body name="wheelf{idx}_{i}d" pos="{pos_x} 0 {pos_z}">
                        <joint name="jointf_{idx}_{i}d" type="hinge" axis="0 1 0"/>
                        <body name="wheel_piecef_{idx}_{i}d" quat="0.5 -0.5 0.5 0.5">
                            <geom type="cylinder" size="{wheel_diameter} {wheel_depth}" rgba="0.0 0.2 0.01 1"/>
                            <!-- <geom type="cylinder" size="{wheel_diameter*.1} {wheel_depth*1.05}" rgba="0.8 0.0 0.0 1" pos="{wheel_diameter*.8} 0 0"/> -->
                        </body>
                    </body>"""
        text += f"""<body name="wheelf{idx}_{i}u" pos="{pos_x} 0 {-pos_z}">
                        <joint name="jointf_{idx}_{i}u" type="hinge" axis="0 1 0"/>
                        <body name="wheel_piecef_{idx}_{i}u" quat="0.5 -0.5 0.5 0.5">
                            <geom type="cylinder" size="{wheel_diameter} {wheel_depth}" rgba="0.0 0.2 0.01 1"/>
                        </body>
                    </body>"""
    text += "</body></body>"
    return text


def gen(timestep: float=.002) -> str:
    generated_xml_path = "./xml_files/wheely.xml"
    gen_file = open(generated_xml_path, 'w')
    gen_file.write(begin(timestep))

    for i in range(wheel_count):
        pos_x = (base_dims[0]/wheel_count)*i*2 - base_dims[0] + wheel_diameter
        pos_y = base_dims[1] - wheel_depth*.6
        pos_z = -base_dims[2]*1.5 - wheel_diameter*.5

        gen_file.write(f"""<body name="wheel{i}l" pos="{pos_x} {pos_y} {pos_z}">
                <joint name="joint{i}l" type="hinge" axis="0 1 0"/>
                <body name="wheel_piece{i}l" quat="0.5 -0.5 0.5 0.5">
                    <geom type="cylinder" size="{wheel_diameter} {wheel_depth}" rgba="0.0 0.2 0.01 1"/>
                </body>
            </body>""")
        gen_file.write(f"""<body name="wheel{i}r" pos="{pos_x} {-pos_y} {pos_z}">
                <joint name="joint{i}r" type="hinge" axis="0 1 0"/>
                <body name="wheel_piece{i}r" quat="0.5 -0.5 0.5 0.5">
                    <geom type="cylinder" size="{wheel_diameter} {wheel_depth}" rgba="0.0 0.2 0.01 1"/>
                </body>
            </body>""")
        gen_file.write(f"""<body name="wheel{i}tl" pos="{pos_x} {pos_y} {-pos_z}">
                <joint name="joint{i}tl" type="hinge" axis="0 1 0"/>
                <body name="wheel_piece{i}tl" quat="0.5 -0.5 0.5 0.5">
                    <geom type="cylinder" size="{wheel_diameter} {wheel_depth}" rgba="0.0 0.2 0.01 1"/>
                </body>
            </body>""")
        gen_file.write(f"""<body name="wheel{i}tr" pos="{pos_x} {-pos_y} {-pos_z}">
                <joint name="joint{i}tr" type="hinge" axis="0 1 0"/>
                <body name="wheel_piece{i}tr" quat="0.5 -0.5 0.5 0.5">
                    <geom type="cylinder" size="{wheel_diameter} {wheel_depth}" rgba="0.0 0.2 0.01 1"/>
                </body>
            </body>""")
        
    # Flipper
    gen_file.write(make_flipper(idx=0, pos=[ base_dims[0]*.7, -0.45, 0], offset=(base_dims[0]*.5), front=True))
    gen_file.write(make_flipper(idx=1, pos=[-base_dims[0]*.7, -0.45, 0], offset=(-base_dims[0]*.5), front=False))
    gen_file.write(make_flipper(idx=2, pos=[ base_dims[0]*.7,  0.45, 0], offset=(base_dims[0]*.5), front=True))
    gen_file.write(make_flipper(idx=3, pos=[-base_dims[0]*.7,  0.45, 0], offset=(-base_dims[0]*.5), front=False))
        
    gen_file.write("""        </body>
    </worldbody>
    <actuator>""")

    main_indices_l = []
    main_indices_r = []
    flipper_indices_0 = []
    flipper_indices_1 = []
    flipper_indices_2 = []
    flipper_indices_3 = []

    index = 0
    for i in range(wheel_count):
        gen_file.write(f'<motor joint="joint{i}r" ctrlrange="{-wheel_max_torque} {wheel_max_torque}"/>')
        main_indices_r.append(index)
        index += 1
        gen_file.write(f'<motor joint="joint{i}tr" ctrlrange="{-wheel_max_torque} {wheel_max_torque}"/>')
        main_indices_r.append(index)
        index += 1
        gen_file.write(f'<motor joint="joint{i}l" ctrlrange="{-wheel_max_torque} {wheel_max_torque}"/>')
        main_indices_l.append(index)
        index += 1
        gen_file.write(f'<motor joint="joint{i}tl" ctrlrange="{-wheel_max_torque} {wheel_max_torque}"/>')
        main_indices_l.append(index)
        index += 1

    for i in range(flipper_wheel_count):
        gen_file.write(f'<motor joint="jointf_0_{i}d" ctrlrange="{-wheel_max_torque} {wheel_max_torque}"/>')
        flipper_indices_0.append(index)
        index += 1
        gen_file.write(f'<motor joint="jointf_0_{i}u" ctrlrange="{-wheel_max_torque} {wheel_max_torque}"/>')
        flipper_indices_0.append(index)
        index += 1
        gen_file.write(f'<motor joint="jointf_1_{i}d" ctrlrange="{-wheel_max_torque} {wheel_max_torque}"/>')
        flipper_indices_1.append(index)
        index += 1
        gen_file.write(f'<motor joint="jointf_1_{i}u" ctrlrange="{-wheel_max_torque} {wheel_max_torque}"/>')
        flipper_indices_1.append(index)
        index += 1
        gen_file.write(f'<motor joint="jointf_2_{i}d" ctrlrange="{-wheel_max_torque} {wheel_max_torque}"/>')
        flipper_indices_2.append(index)
        index += 1
        gen_file.write(f'<motor joint="jointf_2_{i}u" ctrlrange="{-wheel_max_torque} {wheel_max_torque}"/>')
        flipper_indices_2.append(index)
        index += 1
        gen_file.write(f'<motor joint="jointf_3_{i}d" ctrlrange="{-wheel_max_torque} {wheel_max_torque}"/>')
        flipper_indices_3.append(index)
        index += 1
        gen_file.write(f'<motor joint="jointf_3_{i}u" ctrlrange="{-wheel_max_torque} {wheel_max_torque}"/>')
        flipper_indices_3.append(index)
        index += 1

    gen_file.write(f'<motor joint="jointf0" ctrlrange="{-flipper_max_torque} {flipper_max_torque}"/>')
    gen_file.write(f'<motor joint="jointf1" ctrlrange="{-flipper_max_torque} {flipper_max_torque}"/>')
    gen_file.write(f'<motor joint="jointf2" ctrlrange="{-flipper_max_torque} {flipper_max_torque}"/>')
    gen_file.write(f'<motor joint="jointf3" ctrlrange="{-flipper_max_torque} {flipper_max_torque}"/>')

    gen_file.write(end())

    info = {
        "base_wheel_count": wheel_count,
        "flipper_wheel_count": flipper_wheel_count,
        "main_indices_l": main_indices_l,
        "main_indices_r": main_indices_r,
        "flipper_indices_0": flipper_indices_0,
        "flipper_indices_1": flipper_indices_1,
        "flipper_indices_2": flipper_indices_2,
        "flipper_indices_3": flipper_indices_3,
        "flipper_max_torque": flipper_max_torque,
        "track_max_torque": wheel_max_torque
    }

    return generated_xml_path, info


if __name__ == "__main__":
    gen()
