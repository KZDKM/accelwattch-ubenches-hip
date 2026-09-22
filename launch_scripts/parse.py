import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

static_power = 310.0;

def main():
    # parse the cmd arguments (only for specifying input files for now)
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help = "path to profiling result CSV file")
    args = parser.parse_args();

    # read csv and sanity check
    df = pd.read_csv(args.file, skiprows=1)
    required = ['timestamp_ns', 'current_socket_power_W', 'inst_power_W', 'gfx_clock_MHz']
    if not all(col in df.columns for col in required):
        print("CSV malformed!")
        sys.exit(1)
        
    power = df['current_socket_power_W'].values
    static_power_val = np.max(power)
    timestamps_ns = df['timestamp_ns'].values

    t_0 = timestamps_ns[0]
    time_sec = (timestamps_ns - t_0) / 1e9 # normalize time to seconds
    duration = time_sec[-1]

    dt = np.diff(time_sec)
    dt = np.append(dt, dt[-1])
    
    is_dynamic = power > static_power
    dyn_energy = np.sum(power[is_dynamic] * dt[is_dynamic]) 

    print(args.file)
    print(f"max instantaneous power: {static_power_val:.2f}W")
    print(f"dynamic energy (nominal static power {static_power}W): {dyn_energy:.2f}J")
    print(f"run duration: {duration:.2f}s")
    print(f"average dynamic power: {(dyn_energy / duration):.2f}W")

    # plot
    fig, axis = plt.subplots(figsize=(10,6))
    axis.plot(time_sec, power, label='Power (W)', color='blue')
    axis.set_xlabel('Time (s)')
    axis.set_ylabel('Power (W)')
    axis.set_title(args.file)
    axis.axhline(y=static_power, color='green', linestyle='--', label=f'Static PWR = {static_power:.2f} W')
    axis.legend()
    axis.grid(True, alpha=0.3)
    plt.savefig(args.file + "_plot.jpg", dpi=150)
    

if __name__ == "__main__":
    main()