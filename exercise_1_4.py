from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

#  parameters
m =   0.25
r_m = 0.01
J_m = 1e-5
L_m = 0.01
R_m = 2.0
K_m = 0.05
mu_r= 0.1
c_x = 1.0
c_y = 1.0

J_eff = J_m + m * r_m**2 / 2
c_d = r_m**2 * (3 * mu_r + (c_x + c_y) / 4)
c_c= r_m**2 * (mu_r + (c_x - c_y) / 4)
axis_damping = np.array([c_d + c_c, c_d - c_c])

duration  = 20.0

centre = np.array( [0.10, 0.10])
angles = np.pi / 2 + 2 * np.pi * np.arange(6) / 5
vertices = centre + 0.04 * np.column_stack((np.cos(angles), np.sin(angles)))
vertices[-1]= vertices[0]
output_directory = Path(__file__).resolve().parent


def reference(t, shape):
    if shape == "ellipse":
        w = 2 * np.pi / duration
        phase = w * t
        a, b = 0.04, 0.025
        p = centre + np.array([a *  np.cos(phase), b * np.sin(phase)])
        v = np.array([-a * w * np.sin(phase), b *  w * np.cos(phase)])
        acc = np.array([-a * w**2 * np.cos(phase), -b * w**2 * np.sin(phase)])
        jerk = np.array([a * w**3 *  np.sin(phase), -b * w**3 * np.cos(phase)])
        
        return p, v, acc, jerk

    edge_time = duration /  5
    edge = min(int(t  / edge_time), 4)
    s = np.clip((t -  edge * edge_time) / edge_time, 0.0, 1.0)
    delta = vertices[edge + 1] - vertices[edge]
    
    # 0 velocity and aceleration at both ends of side
    h = 10 * s**3 - 15 * s**4 + 6 * s**5
    dh = (30 * s**2 - 60 * s**3 + 30 * s**4) / edge_time
    ddh = (60 * s - 180 * s**2 + 120 * s**3) / edge_time**2
    dddh = (60 - 360 * s + 360 * s**2) / edge_time**3
    return vertices[edge] + delta * h, delta * dh, delta * ddh, delta * dddh


def motor_voltages(t, shape):
    _, v, acc, jerk = reference(t, shape)
    current = 2 * (J_eff * acc + axis_damping * v) / (K_m * r_m)
    current_rate = 2 * (J_eff * jerk + axis_damping * acc) / (K_m * r_m)
    u_x, u_y = L_m * current_rate + R_m * current + 2 * K_m * v / r_m
    return np.array([(u_x + u_y) / 2, (u_x - u_y) / 2])


def motor_model(t, state, shape):
    theta_l, theta_r, w_l, w_r, i_l, i_r = state
    u_l, u_r = motor_voltages(t, shape)
    return [
        w_l,
        w_r,
        (K_m * i_l - c_d * w_l - c_c * w_r) / J_eff,
        (K_m * i_r - c_c * w_l - c_d * w_r) / J_eff,
        (u_l - R_m * i_l - K_m * w_l) / L_m,
        (u_r - R_m * i_r - K_m * w_r) / L_m,
    ]


def simulate_and_plot(shape):
    p, v, acc, _ = reference(0.0, shape)
    i_x, i_y = 2 * (J_eff * acc + axis_damping * v) / (K_m * r_m)


    initial_state = [
        (p[0] + p[1]) / r_m,
        (p[0] - p[1]) / r_m,
        (v[0] + v[1]) / r_m,
        (v[0] - v[1]) / r_m,
        (i_x + i_y) / 2,
        (i_x - i_y) / 2,
    ]
    time = np.linspace(0.0, duration, 2001)
    result = solve_ivp(
        motor_model, (0.0, duration), initial_state, args=(shape,),
        t_eval=time, method="DOP853", rtol=1e-9, atol=1e-11, max_step=0.02,
    )
    if not result.success:
        raise RuntimeError(result.message)

    actual = r_m / 2 * np.column_stack(
        (result.y[0] + result.y[1], result.y[0] - result.y[1])
    )
    desired = np.array([reference(t, shape)[0] for t in time])
    voltages = np.array([motor_voltages(t, shape) for t in time])
    error = np.max(np.linalg.norm(actual - desired, axis=1))
    closing_error = np.linalg.norm(actual[-1] - actual[0])
    print(
        f"{shape}: maximum position error = {error:.3e} m, "
        f"closing error = {closing_error:.3e} m, "
        f"maximum motor voltage = {np.max(np.abs(voltages)):.3f} V"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    axes[0].plot(*actual.T * 1000, color="#0072B2", lw=1.4, label="Simulated")
    axes[0].plot(*desired.T * 1000, "--", color="grey", lw=1.0, label="Desired")
    if shape == "pentagon":
        axes[0].plot(*vertices.T * 1000, "o", color="#222222", ms=3)
    axes[0].set(
        xlabel="x (mm)", ylabel="y (mm)", title=f"{shape.title()}",
        xlim=(45, 155), ylim=(45, 155),
    )
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend()
    axes[1].plot(time, voltages[:, 0], color="#0072B2", label="Left motor")
    axes[1].plot(time, voltages[:, 1], color="#D55E00", label="Right motor")
    axes[1].set(xlabel="Time (s)", ylabel="Voltage (V)", title="Motor voltages")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(output_directory / f"{shape}.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    for shape in ("ellipse", "pentagon"):
        simulate_and_plot(shape)
