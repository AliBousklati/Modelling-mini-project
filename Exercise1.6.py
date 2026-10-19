from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.sparse import csc_matrix, diags, eye
from scipy.sparse.linalg import factorized


#  PLA specs
rho = 1240.0
c_p = 1800.0
k =  0.13

# Assumed  parameters
width = 0.0004
h_air =  10.0
h_bed = 100.0
T_air = 20.0
T_bed = 60.0
T_deposit = 200.0
print_time = 20.0
a = 0.04
b= 0.025
tolerance = 1.0

alpha = k /(rho *c_p)
loss_rate = (h_bed  + 3 * h_air) / (rho * c_p * width)
T_equilibrium = (h_bed *  T_bed + 3 * h_air * T_air) / (h_bed + 3 * h_air)


def geometry(n_cells):
    angle = np.linspace(0.0 ,  2 * np.pi, 20001)
    arc = cumulative_trapezoid(
        np.sqrt((a * np.sin(angle))**2 + (b *np.cos(angle))**2),
        angle, initial=0.0,
    )
    length = arc[-1]
    ds = length / n_cells
    #activeer wnr de cell boundary reached
    birth_times = np.interp(
        np.arange(n_cells  + 1) * ds, arc, print_time * angle / (2* np.pi)
    )
    centres = (np.arange(n_cells) + 0.5) *  ds
    return centres, ds, birth_times


def stepper(n_active, ds, dt, closed):
    if n_active == 1:
        laplacian = csc_matrix((1, 1))
    else:
        laplacian = diags(
            [np.ones(n_active - 1), -2  * np.ones(n_active), np.ones(n_active - 1)],
            [-1, 0, 1], format="lil",
        )
        if closed:
            laplacian[0, -1] = 1.0
            laplacian[-1, 0] = 1.0
        else:
             # geen heat flux bij ends
            laplacian[0, 0] = -1.0
            laplacian[-1, -1] = -1.0
        laplacian = laplacian.tocsc()
    operator = alpha * laplacian / ds**2 - loss_rate * eye(n_active, format="csc")
    return factorized(eye(n_active, format="csc") - dt * operator)

#hou op .2
def simulate(n_cells=200,  max_step=0.2):
    centres, ds, birth_times = geometry(n_cells)
    times, history = [], []
    temperature = np.empty(0)

    def record(t):
        field = np.full(n_cells, np.nan)
        field[:len(temperature)] = temperature
        if times and abs(t - times[-1])  < 1e-10:
            history[-1] = field
        else:
            times.append(t)
            history.append(field)

    for cell in range(n_cells):
        temperature = np.append(temperature, T_deposit)
        start, finish = birth_times[cell:cell + 2]
        record(start)
        n_steps = max(1, int(np.ceil((finish - start) / max_step)))
        dt = (finish - start) / n_steps
        solve = stepper(cell + 1, ds, dt, closed=False)
        for j in range(n_steps):
            temperature = solve(temperature + dt * loss_rate * T_equilibrium)
            record(start + (j + 1) * dt)

    # eerst en laatste cell connecten
    time = print_time
    solve = stepper(n_cells, ds, max_step, closed=True)
    while time < print_time +  120:
        temperature = solve(temperature  + max_step * loss_rate * T_equilibrium)
        time += max_step
        record(time)
        nearly_uniform = np.ptp(temperature)  <= tolerance
        near_equilibrium = np.max(np.abs(temperature- T_equilibrium)) <= tolerance
        if nearly_uniform and near_equilibrium:
            break
    else:
        raise RuntimeError("cooling niet gehaald")

    field = np.asarray(history)
    return {
        "time": np.asarray(times), "temperature": field, "s": centres,
        "cooling_time": time, "ds": ds,
    }


def plot(result):
    time, field = result["time"], result["temperature"]
    minimum = np.nanmin(field, axis=1)
    maximum = np.nanmax(field, axis=1)
    mean = np.nanmean(field, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#eeeeee")
    image = axes[0].pcolormesh(
        time, result["s"] * 1000, np.ma.masked_invalid(field.T),
        shading="auto", cmap=cmap, vmin=T_equilibrium, vmax=T_deposit,
    )
    fig.colorbar(image, ax=axes[0], label="Temperature (deg C)")
    axes[0].set(
        xlabel="Time since printing starts (s)", ylabel="Arc length s (mm)",
        title="Temperature along the filament",
    )
    axes[0].text(
        0.6, 0.95, "Gray: not yet deposited",
        transform=axes[0].transAxes, va="top",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "black"},
    )

    axes[1].fill_between(time, minimum, maximum, color="#dce8f2")
    axes[1].plot(time, maximum, color="#c44932", label="Maximum")
    axes[1].plot(time, mean, "--", color="#222222", label="Mean")
    axes[1].plot(time, minimum, color="#1872b8", label="Minimum")
    axes[1].axhline(T_equilibrium, color="#4d8057", ls=":", label="Equilibrium")
    axes[1].set(
        xlabel="Time since printing starts (s)", ylabel="Temperature (deg C)",
        title="Cooling after deposition", ylim=(45, 205),
    )
    axes[1].axvline(print_time, color="#777777", ls="--", label="Printing ends")
    axes[1].axvline(
        result["cooling_time"], color="#76528e", ls="--", label="Cooling criterion"
    )
    axes[0].axvline(print_time, color="#777777", ls="--")
    for ax in axes:
        ax.set_xlim(0.0, time[-1] + 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="upper right")
    fig.savefig(Path(__file__).resolve().parent / "filament_cooling.png", dpi=180)
    plt.show()



if __name__ == "__main__":
    result = simulate()
    final = result["temperature"][-1]
    print(f"Printed length: {len(result['s']) * result['ds'] * 1000:.2f} mm")
    print(f"Equilibrium temperature: {T_equilibrium:.2f} deg C")
    print(f"Printing finishes at: {print_time:.2f} s")
    print(f"Cooling criterion reached at: {result['cooling_time']:.2f} s")
    print(f"Additional cooling time: {result['cooling_time'] - print_time:.2f} s")
    print(f"Final temperature range: {final.min():.2f} to {final.max():.2f} deg C")
    plot(result)
