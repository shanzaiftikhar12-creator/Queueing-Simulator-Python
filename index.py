import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import itertools


# --- Manual Poisson PMF For CP ---
def poisson_pmf(k, lam):
    return (lam**k * math.exp(-lam)) / math.factorial(k)

def mm1_simulation(lambda_val, mu_val):
    # Safety parameters
    threshold = 0.99999
    max_k = 500  # safety upper bound
    pmf = []
    cp_cum = []
    no_between_arrivals = []
    p0 = math.exp(-lambda_val)
    current_p = p0
    cum = 0.0

    k = 0
    while True:
        pmf.append(current_p)
        cum += current_p
        cp_cum.append(round(cum, 5))
        no_between_arrivals.append(k)
        if cum >= threshold or k >= max_k:
            break
        k += 1
        current_p = current_p * (lambda_val / k)

    if cp_cum:
        cp_cum[-1] = min(cp_cum[-1], 1.0)

    cp_lookup = [0.0] + cp_cum[:-1]

    inter_arrivals = [0]
    n_ranges = len(cp_cum)
    for _ in range(1, n_ranges):
        r = np.random.rand()
        found = False
        for i in range(n_ranges):
            if cp_lookup[i] <= r <= cp_cum[i]:
                inter_arrivals.append(i)
                found = True
                break
        if not found:
            inter_arrivals.append(n_ranges - 1)

    arrival_times = np.cumsum(inter_arrivals)
    obs_numbers = list(range(1, len(arrival_times) + 1))

    # --- New Columns: Service Time, Start Time, End Time ---
    service_times = [max(1, math.ceil(-mu_val * math.log(np.random.rand()))) for _ in range(len(arrival_times))]
    service_starts = [0] * len(arrival_times)
    service_ends = [0] * len(arrival_times)

    for i in range(len(arrival_times)):
        if i == 0:
            service_starts[i] = arrival_times[i]
        else:
            service_starts[i] = max(arrival_times[i], service_ends[i - 1])
        service_ends[i] = service_starts[i] + service_times[i]

    # --- Additional Performance Columns ---
    turnaround_times = [service_ends[i] - arrival_times[i] for i in range(len(arrival_times))]
    wait_times = [turnaround_times[i] - service_times[i] for i in range(len(arrival_times))]
    response_times = [service_starts[i] - arrival_times[i] for i in range(len(arrival_times))]

    # --- Create DataFrame ---
    df = pd.DataFrame({
        "Observation": obs_numbers,
        "CP": cp_cum,
        "CP Lookup": cp_lookup,
        "No Between Arrivals": no_between_arrivals,
        "Inter-arrivals": inter_arrivals,
        "Arrival Times": arrival_times.astype(int),
        "Service Time": service_times,
        "Service Start": service_starts,
        "Service End": service_ends,
        "Turnaround Time": turnaround_times,
        "Wait Time": wait_times,
        "Response Time": response_times
    })

    df = df.astype({
        "Observation": int,
        "No Between Arrivals": int,
        "Inter-arrivals": int,
        "Arrival Times": int,
        "Service Time": int,
        "Service Start": int,
        "Service End": int,
        "Turnaround Time": int,
        "Wait Time": int,
        "Response Time": int
    })

    return df

def mms_simulation(lambda_val, mu_val, servers):
    # --- Safety & setup ---
    threshold = 0.99999
    max_k = 500  # safety upper bound to avoid infinite loops
    pmf = []
    cp_cum = []
    no_between_arrivals = []

    # --- Iterative Poisson PMF (no factorial overflow) ---
    p0 = math.exp(-lambda_val)
    current_p = p0
    cum = 0.0
    k = 0

    while True:
        pmf.append(current_p)
        cum += current_p
        cp_cum.append(round(cum, 5))
        no_between_arrivals.append(k)

        # stop when CP reaches threshold or we hit max_k
        if cum >= threshold or k >= max_k:
            break

        k += 1
        current_p = current_p * (lambda_val / k)

    # ensure CP doesn’t exceed 1.0 due to rounding
    if cp_cum:
        cp_cum[-1] = min(cp_cum[-1], 1.0)

    # CP lookup (lower bounds)
    cp_lookup = [0.0] + cp_cum[:-1]

    # --- Generate inter-arrival times ---
    inter_arrivals = [0]
    n_ranges = len(cp_cum)
    for _ in range(1, n_ranges):
        r = np.random.rand()
        found = False
        for i in range(n_ranges):
            if cp_lookup[i] <= r <= cp_cum[i]:
                inter_arrivals.append(i)
                found = True
                break
        if not found:
            inter_arrivals.append(n_ranges - 1)

    # --- Arrival times ---
    arrival_times = np.cumsum(inter_arrivals)
    obs_numbers = list(range(1, len(arrival_times) + 1))

    # --- Service times (Exponential Distribution) ---
    service_times = [max(1, math.ceil(-mu_val * math.log(np.random.rand()))) for _ in range(len(arrival_times))]

    # --- Multi-server scheduling logic ---
    server_end_times = [0] * servers  # track end time of each server
    service_start = []
    service_end = []
    server_assigned = []

    for i in range(len(arrival_times)):
        # 1. Identify which servers are currently idle (free)
        free_servers = [idx for idx, end_time in enumerate(server_end_times) if end_time <= arrival_times[i]]

        if free_servers:
            # 2. If servers are free, pick the one with the lowest index (Priority: S1 > S2 > S3)
            server_idx = min(free_servers)
            next_available = server_end_times[server_idx]
        else:
            # 3. If all busy, pick the one that finishes earliest
            next_available = min(server_end_times)
            server_idx = server_end_times.index(next_available)

        # Start time is max(arrival time, server free time)
        start_time = max(arrival_times[i], next_available)
        end_time = start_time + service_times[i]

        # Update that server’s end time
        server_end_times[server_idx] = end_time

        # Record info
        service_start.append(start_time)
        service_end.append(end_time)
        server_assigned.append(f"S{server_idx + 1}")

    # --- Additional Performance Metrics ---
    turnaround_times = [service_end[i] - arrival_times[i] for i in range(len(arrival_times))]
    wait_times = [turnaround_times[i] - service_times[i] for i in range(len(arrival_times))]
    response_times = [service_start[i] - arrival_times[i] for i in range(len(arrival_times))]

    # --- Build DataFrame ---
    df = pd.DataFrame({
        "Observation": obs_numbers,
        "CP": cp_cum,
        "CP Lookup": cp_lookup,
        "No. Between Arrivals": no_between_arrivals,
        "Inter-arrivals": inter_arrivals,
        "Arrival Times": arrival_times.astype(int),
        "Service Time": service_times,
        "Service Start": service_start,
        "Service End": service_end,
        "Server": server_assigned,
        "Turnaround Time": turnaround_times,
        "Wait Time": wait_times,
        "Response Time": response_times
    })

    # ensure int columns
    int_cols = ["Observation", "No. Between Arrivals", "Inter-arrivals",
                "Arrival Times", "Service Time", "Service Start", "Service End",
                "Turnaround Time", "Wait Time", "Response Time"]
    df[int_cols] = df[int_cols].astype(int)

    return df

# Simulation Table Container
def show_table(parent_frame, df, lambda_val, mu_val, servers):
    for widget in parent_frame.winfo_children():
        widget.destroy()

    # --- MAIN CONTAINER (scrollable) ---
    container = tk.Frame(parent_frame, bg="#666633")
    container.pack(fill="both", expand=True)

    # Canvas + scrollbar
    canvas = tk.Canvas(container, bg="#666633", highlightthickness=0)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    # Scrollable frame inside canvas
    scrollable_frame = tk.Frame(canvas, bg="#666633")
    canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="n")

    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    scrollable_frame.bind("<Configure>", on_frame_configure)

    def on_canvas_configure(event):
        canvas.itemconfig(canvas_window, width=event.width)
    canvas.bind("<Configure>", on_canvas_configure)

    # --- BACK BUTTON (top-left) ---
    shadow_back = tk.Frame(scrollable_frame, bg="#333311")
    shadow_back.pack(anchor="w", pady=15, padx=15)
    btn_back = tk.Button(
        shadow_back,
        text="← Back",
        font=("Arial", 14, "bold"),
        bg="#888844",
        fg="white",
        bd=0,
        relief="flat",
        activebackground="#777733",
        activeforeground="white",
        command=lambda: raise_frame(main_frame)  # Replace with your raise_frame(main_frame)
    )
    btn_back.pack(padx=(0, 5), pady=(0, 5), ipadx=15, ipady=8)
    btn_back.bind("<Enter>", lambda e: btn_back.config(bg="#494920"))
    btn_back.bind("<Leave>", lambda e: btn_back.config(bg="#888844"))

    # --- HEADING SECTION (centered full width) ---
    header_frame = tk.Frame(scrollable_frame, bg="#666633")
    header_frame.pack(pady=(10, 20), fill="x")

    tk.Label(
        header_frame,
        text="Performance Analysis",
        font=("Arial", 22, "bold"),
        fg="white",
        bg="#666633"
    ).pack(pady=(10, 5))

    # --- Dynamic Parameters ---
    if lambda_val is not None and mu_val is not None:
        queue_type = f"M/M/{servers} Queue" if servers and servers != 1 else "M/M/1 Queue"
        params_frame = tk.Frame(header_frame, bg="#666633")
        params_frame.pack(pady=(5, 10))

        tk.Label(params_frame, text=queue_type, font=("Arial", 16, "bold"), fg="#eeeecc", bg="#666633").pack(anchor="center", pady=1)
        tk.Label(params_frame, text=f"λ = {lambda_val}", font=("Arial", 14, "bold"), fg="#eeeecc", bg="#666633").pack(anchor="center", pady=1)
        tk.Label(params_frame, text=f"μ = {mu_val}", font=("Arial", 14, "bold"), fg="#eeeecc", bg="#666633").pack(anchor="center", pady=1)
        if servers and servers != 1:
            tk.Label(params_frame, text=f"Servers = {servers}", font=("Arial", 14, "bold"), fg="#eeeecc", bg="#666633").pack(anchor="center", pady=1)

    tk.Label(
        header_frame,
        text="Customer Data Chart",
        font=("Arial", 18, "bold"),
        fg="white",
        bg="#666633"
    ).pack(pady=(10, 5))

    # --- TABLE (centered) ---
    table_wrapper = tk.Frame(scrollable_frame, bg="#666633")
    table_wrapper.pack(pady=(10, 50), fill="x")
    table_frame = tk.Frame(table_wrapper, bg="#666633")
    table_frame.pack(anchor="center")

    cols = list(df.columns)
    total_cols = len(cols)

    # Table header
    for j, col in enumerate(cols):
        tk.Label(
            table_frame,
            text=col,
            font=("Arial", 12, "bold"),
            bg="#999966",
            fg="white",
            pady=6,
            padx=10,
            relief="flat",
            highlightthickness=0,
            bd=1
        ).grid(row=0, column=j, sticky="nsew")

    # Table data rows
    for i, row in enumerate(df.itertuples(index=False), start=1):
        for j, val in enumerate(row):
            tk.Label(
                table_frame,
                text=val,
                font=("Arial", 11),
                bg="#999966" if i % 2 == 0 else "#aaa977",
                fg="white",
                relief="flat",
                highlightthickness=0,
                bd=0
            ).grid(row=i, column=j, sticky="nsew")

    for j in range(total_cols):
        table_frame.grid_columnconfigure(j, weight=1)

    # If it's MM1
    if servers is None or servers == 1:
        create_averages_frame(scrollable_frame, df, lambda_val, mu_val)
    else:
    # MMS case
        create_averages_frame(scrollable_frame, df, lambda_val, mu_val, servers=servers)

    if servers is not None and servers > 1:
        draw_mms_gantt(df, scrollable_frame)
    else:
        draw_mm1_gantt(df, scrollable_frame)

    # Spacer at bottom
    tk.Label(scrollable_frame, text="", bg="#666633").pack(pady=40)


#M/M/1 AND M/M/S SIMULATION RESULTS 
def create_averages_frame(parent_frame, df, lambda_val, mu_val, servers=None):
    """
    Creates a labeled frame under the table to show simulation averages,
    updates the values immediately.
    """
    averages_frame = tk.LabelFrame(
        parent_frame, text="Simulation Results",
        font=("Arial", 16, "bold"), fg="white", bg="#666633",
        bd=2, relief="groove", padx=20, pady=10,
        width=int(parent_frame.winfo_screenwidth() * 0.4)
    )
    averages_frame.pack(pady=(0,20), anchor="center")  # bottom center under the table

    # Create labels
    lbl_avg_turnaround = tk.Label(averages_frame, text="0",font=("Arial", 13), bg="#666633", fg="white")
    lbl_avg_wait = tk.Label(averages_frame, text="0",font=("Arial", 13), bg="#666633", fg="white")
    lbl_avg_response = tk.Label(averages_frame, text="0",font=("Arial", 13), bg="#666633", fg="white")
    lbl_avg_interarrival = tk.Label(averages_frame, text="0",font=("Arial", 13), bg="#666633", fg="white")
    lbl_avg_service = tk.Label(averages_frame, text="0",font=("Arial", 13), bg="#666633", fg="white")
    lbl_utilization = tk.Label(averages_frame, text="0",font=("Arial", 13), bg="#666633", fg="white")
    lbl_total_customers = tk.Label(averages_frame, text="0",font=("Arial", 13), bg="#666633", fg="white")

    # Labels grid
    labels = [
        ("Avg Turnaround Time:", lbl_avg_turnaround),
        ("Avg Wait Time:", lbl_avg_wait),
        ("Avg Response Time:", lbl_avg_response),
        ("Avg Inter-arrival Time:", lbl_avg_interarrival),
        ("Avg Service Time:", lbl_avg_service),
        ("Server Utilization:", lbl_utilization),
        ("Total Customers:", lbl_total_customers)
    ]

    for i, (text_label, value_label) in enumerate(labels):
        tk.Label(averages_frame, text=text_label, bg="#666633", fg="white").grid(
            row=i, column=0, sticky="w", padx=5, pady=2
        )
        value_label.grid(row=i, column=1, sticky="w", padx=5, pady=2)

    # --- Update values immediately ---
    if servers and servers > 1:
        # MMS averages (if you have a separate function)
        update_mms_averages(
            df, lambda_val, mu_val, servers,
            lbl_avg_turnaround,
            lbl_avg_wait,
            lbl_avg_response,
            lbl_avg_interarrival,
            lbl_avg_service,
            lbl_utilization,
            lbl_total_customers
        )
    else:
        # MM1 averages
        update_simulation_averages(
            df, lambda_val, mu_val,
            lbl_avg_turnaround,
            lbl_avg_wait,
            lbl_avg_response,
            lbl_avg_interarrival,
            lbl_avg_service,
            lbl_utilization,
            lbl_total_customers
        )

    return averages_frame  # in case you need to reference it later



#MM1 Gant Chart
def draw_mm1_gantt(df, scrollable_frame):
    # --- Gantt Chart (Equal box width, no outer border) ---
    fig, ax = plt.subplots(figsize=(10, 2.8), facecolor="#666633")
    ax.set_facecolor("#666633")

    if 'Service Start' in df.columns and 'Service End' in df.columns:
        # Define 20 customer colors (distinct shades)
        customer_colors = [
            "#A56A64", "#7D719B", "#818F6D", "#D18685", "#6379A1",
            "#A36E6E", "#AF7EA6", "#80B8AB", "#A0655B", "#85CEC6",
            "#DDC48B", "#7677A3", "#916269", "#B89775", "#7CB4B1",
            "#AF6C5F", "#609C9C", "#AA6C76", "#AD778F", "#6989AD"
        ]
        color_cycle = itertools.cycle(customer_colors)

        # Build timeline (add idle gaps if any)
        timeline = []
        current_time = df['Service Start'].min()

        for i in range(len(df)):
            s_start = df.loc[i, 'Service Start']
            s_end = df.loc[i, 'Service End']
            cust = f"C{df.loc[i, 'Observation']}"

            # Add idle time if there’s a gap
            if s_start > current_time:
                timeline.append(("Idle", current_time, s_start))
                current_time = s_start

            # Add service time
            timeline.append((cust, s_start, s_end))
            current_time = s_end

        total_boxes = len(timeline)
        box_width = 1.5  # fixed equal width
        x_pos = 0

        for label, start, end in timeline:
            if "C" in label:
                color = next(color_cycle)  # get next customer color
            else:
                color = "#BBBBA0"  # idle box color

            rect = plt.Rectangle((x_pos, 0), box_width, 1,
                                 facecolor=color, edgecolor="white", lw=1.5)
            ax.add_patch(rect)

            # Box label (C1, Idle, etc.)
            ax.text(x_pos + box_width / 2, 0.5, label,
                    color="white", fontsize=9, fontweight="bold", ha="center", va="center")

            # Start time (below left edge)
            ax.text(x_pos, -0.25, str(int(start)),
                    color="white", fontsize=8, ha="center", va="top")
            # End time (below right edge)
            ax.text(x_pos + box_width, -0.25, str(int(end)),
                    color="white", fontsize=8, ha="center", va="top")

            x_pos += box_width

        # Aesthetics
        ax.set_xlim(0, total_boxes * box_width)
        ax.set_ylim(-0.5, 1.2)
        ax.axis('off')
        ax.set_title(
            "Server Utilization Gantt Chart",
            fontdict={'family': 'Arial', 'size': 16, 'weight': 'bold', 'color': 'white'},
            pad=10
        )

    # Display inside scrollable frame
    gantt_canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
    gantt_canvas.draw()
    gantt_canvas.get_tk_widget().pack(pady=5, anchor="center")

    # Spacer at bottom
    tk.Label(scrollable_frame, text="", bg="#666633").pack(pady=40)

#MMS Gant Chart
def draw_mms_gantt(df, scrollable_frame):
    # Define your custom 20 colors
    customer_colors = [
        "#A56A64", "#7D719B", "#818F6D", "#D18685", "#6379A1",
        "#A36E6E", "#AF7EA6", "#80B8AB", "#A0655B", "#85CEC6",
        "#DDC48B", "#7677A3", "#916269", "#B89775", "#7CB4B1",
        "#AF6C5F", "#609C9C", "#AA6C76", "#AD778F", "#6989AD"
    ]
    color_cycle = itertools.cycle(customer_colors)

    if 'Server' in df.columns:
        servers = df['Server'].unique()
    else:
        servers = ['Server 1']  # fallback

    for srv in servers:
        srv_df = df[df['Server'] == srv].reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(10, 2.8), facecolor="#666633")
        ax.set_facecolor("#666633")

        timeline = []
        if len(srv_df) > 0:
            current_time = srv_df['Service Start'].min()
            for i in range(len(srv_df)):
                s_start = srv_df.loc[i, 'Service Start']
                s_end = srv_df.loc[i, 'Service End']
                cust = f"C{srv_df.loc[i, 'Observation']}"

                # Add idle period if any
                if s_start > current_time:
                    timeline.append(("Idle", current_time, s_start))
                    current_time = s_start

                timeline.append((cust, s_start, s_end))
                current_time = s_end

        total_boxes = len(timeline)
        box_width = 1.5
        x_pos = 0

        # Assign colors to customers
        cust_color_map = {}
        for label, start, end in timeline:
            if "C" in label and label not in cust_color_map:
                cust_color_map[label] = next(color_cycle)

        for label, start, end in timeline:
            if "Idle" in label:
                color = "#BBBBA0"
            else:
                color = cust_color_map.get(label, "#75754B")

            rect = plt.Rectangle((x_pos, 0), box_width, 1,
                                 facecolor=color, edgecolor="white", lw=1.5)
            ax.add_patch(rect)

            # Texts
            ax.text(x_pos + box_width / 2, 0.5, label,
                    color="white", fontsize=9, fontweight="bold",
                    ha="center", va="center")

            ax.text(x_pos, -0.25, str(int(start)),
                    color="white", fontsize=8, ha="center", va="top")

            ax.text(x_pos + box_width, -0.25, str(int(end)),
                    color="white", fontsize=8, ha="center", va="top")

            x_pos += box_width

        # Aesthetics
        ax.set_xlim(0, total_boxes * box_width if total_boxes > 0 else 1)
        ax.set_ylim(-0.5, 1.2)
        ax.axis('off')
        ax.set_title(f"{srv} Utilization Gantt Chart",
                     fontdict={'family': 'Arial', 'size': 16, 'weight': 'bold', 'color': 'white'},
                     pad=10)

        gantt_canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
        gantt_canvas.draw()
        gantt_canvas.get_tk_widget().pack(pady=5, anchor="center")

    tk.Label(scrollable_frame, text="", bg="#666633").pack(pady=40)

# M/M/1 SIMULATION RESULTS
def update_simulation_averages(
    df, lambda_val, mu_val,
    lbl_avg_turnaround,
    lbl_avg_wait,
    lbl_avg_response,
    lbl_avg_interarrival,
    lbl_avg_service,
    lbl_utilization,
    lbl_total_customers
):
    avg_turnaround = df["Turnaround Time"].mean()
    avg_wait = df["Wait Time"].mean()
    avg_response = df["Response Time"].mean()
    avg_interarrival = df["Inter-arrivals"].mean()
    avg_service = df["Service Time"].mean()
    total_busy = df['Service Time'].sum()
    total_time = df['Service End'].max() - df['Service Start'].min()
    server_utilization = (total_busy / total_time) * 100  # in %
    total_customers = len(df)

    lbl_avg_turnaround.config(text=f"{avg_turnaround:.2f}")
    lbl_avg_wait.config(text=f"{avg_wait:.2f}")
    lbl_avg_response.config(text=f"{avg_response:.2f}")
    lbl_avg_interarrival.config(text=f"{avg_interarrival:.2f}")
    lbl_avg_service.config(text=f"{avg_service:.2f}")
    lbl_utilization.config(text=f"{server_utilization:.2f}%")
    lbl_total_customers.config(text=str(total_customers))

# M/M/S SIMULATION RESULTS
def update_mms_averages(
    df, lambda_val, mu_val, servers,
    lbl_avg_turnaround,
    lbl_avg_wait,
    lbl_avg_response,
    lbl_avg_interarrival,
    lbl_avg_service,
    lbl_utilization,
    lbl_total_customers
):
    # --- Calculate averages ---
    avg_turnaround = df["Turnaround Time"].mean()
    avg_wait = df["Wait Time"].mean()
    avg_response = df["Response Time"].mean()
    avg_interarrival = df["Inter-arrivals"].mean()
    avg_service = df["Service Time"].mean()
    servers = df['Server'].nunique()
    total_time_observed = df['Service End'].max() - df['Service Start'].min()  # overall simulation duration
    # Calculate total busy time across all servers
    total_busy_time = 0
    for srv in df['Server'].unique():
        srv_df = df[df['Server'] == srv]
        total_busy_time += srv_df['Service End'].sum() - srv_df['Service Start'].sum()
    # True utilization in %
    server_utilization = (total_busy_time / (total_time_observed * servers)) * 100
    total_customers = len(df)

    # --- Update labels ---
    lbl_avg_turnaround.config(text=f"{avg_turnaround:.2f}")
    lbl_avg_wait.config(text=f"{avg_wait:.2f}")
    lbl_avg_response.config(text=f"{avg_response:.2f}")
    lbl_avg_interarrival.config(text=f"{avg_interarrival:.2f}")
    lbl_avg_service.config(text=f"{avg_service:.2f}")
    lbl_utilization.config(text=f"{server_utilization:.2f}%")
    lbl_total_customers.config(text=str(total_customers))


# MM1 Input page container
def mm1_input_page():
    raise_frame(mm1_frame)
    for widget in mm1_frame.winfo_children():
        widget.destroy()

    # Central container to center all widgets
    container = tk.Frame(mm1_frame, bg="#666633")
    container.pack(expand=True)

    # Header Label
    tk.Label(container, text="Input M/M/1 System Parameters", font=("Arial", 22, "bold"),
             fg="white", bg="#666633").pack(pady=30)
    # Lambda input
    tk.Label(container, text="Lambda (λ):", font=("Arial", 13, "bold"), fg="white", bg="#666633").pack(pady=5)
    shadow_lambda = tk.Frame(container, bg="#333311")
    shadow_lambda.pack(pady=5)
    lambda_entry = tk.Entry(shadow_lambda, bg="#888844", fg="white", bd=0,
                            font=("Arial", 12, "bold"), justify="center", insertbackground="white")
    lambda_entry.pack(padx=(0,5), pady=(0,5), ipady=15, ipadx=50)
    # Mu input
    tk.Label(container, text="Mu (μ):", font=("Arial", 13, "bold"), fg="white", bg="#666633").pack(pady=5)
    shadow_mu = tk.Frame(container, bg="#333311")
    shadow_mu.pack(pady=5)
    mu_entry = tk.Entry(shadow_mu, bg="#888844", fg="white", bd=0,
                        font=("Arial", 12, "bold"), justify="center", insertbackground="white")
    mu_entry.pack(padx=(0,5), pady=(0,5), ipady=15, ipadx=50)

    # Simulation function
    def run_simulation():
        try:
            lam = float(lambda_entry.get())
            mu = float(mu_entry.get())

            # ❌ Negative or zero checks
            if lam <= 0:
                messagebox.showerror("Error", "Arrival rate (λ) must be greater than 0.")
                return
            if mu <= 0:
                messagebox.showerror("Error", "Service rate (μ) must be greater than 0.")
                return

            rho = lam / mu  # utilization factor

            if rho >= 1:
                messagebox.showwarning(
                    "System Unstable",
                    "The queue will grow infinitely because λ ≥ μ.\n\n"
                    "Please enter values where λ < μ to have a stable system."
                )
                return

            # If stable, run simulation
            df = mm1_simulation(lam, mu)
            show_table(mm1_frame, df, lam, mu, servers=1)

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

    # --- Simulate Button ---
    shadow_simulate = tk.Frame(container, bg="#333311")
    shadow_simulate.pack(pady=15)
    btn_simulate = tk.Button(shadow_simulate, text="Run Simulation", font=("Arial", 16, "bold"),
                             bg="#888844", fg="white", bd=0, relief="flat",
                             activebackground="#777733", activeforeground="white",
                             command=run_simulation)
    btn_simulate.pack(padx=(0,5), pady=(0,5), ipadx=90, ipady=15)
    # Hover effect for Simulate
    btn_simulate.bind("<Enter>", lambda e: btn_simulate.config(bg="#494920"))
    btn_simulate.bind("<Leave>", lambda e: btn_simulate.config(bg="#888844"))
    # --- Back Button ---
    shadow_back = tk.Frame(container, bg="#333311")
    shadow_back.pack(pady=5)
    btn_back = tk.Button(shadow_back, text="Back", font=("Arial", 16, "bold"),
                         bg="#888844", fg="white", bd=0, relief="flat",
                         activebackground="#777733", activeforeground="white",
                         command=lambda: raise_frame(main_frame))
    btn_back.pack(padx=(0,5), pady=(0,5), ipadx=107, ipady=15)
    # Hover effect for Back
    btn_back.bind("<Enter>", lambda e: btn_back.config(bg="#494920"))
    btn_back.bind("<Leave>", lambda e: btn_back.config(bg="#888844"))

# MMS Input page container
def mms_input_page():
    raise_frame(mms_frame)
    for widget in mms_frame.winfo_children():
        widget.destroy()
    
    # Central container to center all widgets
    container = tk.Frame(mms_frame, bg="#666633")
    container.pack(expand=True)

    # Header Label
    tk.Label(container, text="Input M/M/S System Parameters", font=("Arial", 22, "bold"),
             fg="white", bg="#666633").pack(pady=30)
    # Lambda input
    tk.Label(container, text="Lambda (λ):", font=("Arial", 13, "bold"), fg="white", bg="#666633").pack(pady=(0,5))
    shadow_lambda = tk.Frame(container, bg="#333311")
    shadow_lambda.pack(pady=(0,5))
    lambda_entry = tk.Entry(shadow_lambda, bg="#888844", fg="white", bd=0,
                            font=("Arial", 12, "bold"), justify="center", insertbackground="white")
    lambda_entry.pack(padx=(0,5), pady=(0,5), ipady=15, ipadx=50)
    # Mu input
    tk.Label(container, text="Mu (μ):", font=("Arial", 13, "bold"), fg="white", bg="#666633").pack(pady=(0,5))
    shadow_mu = tk.Frame(container, bg="#333311")
    shadow_mu.pack(pady=(0,5))
    mu_entry = tk.Entry(shadow_mu, bg="#888844", fg="white", bd=0,
                        font=("Arial", 12, "bold"), justify="center", insertbackground="white")
    mu_entry.pack(padx=(0,5), pady=(0,5), ipady=15, ipadx=50)
    # Number of Servers
    tk.Label(container, text="Number of Servers:", font=("Arial", 13, "bold"), fg="white", bg="#666633").pack(pady=(0,5))
    shadow_servers = tk.Frame(container, bg="#333311")
    shadow_servers.pack(pady=(0,5))
    servers_entry = tk.Entry(shadow_servers, bg="#888844", fg="white", bd=0,
                             font=("Arial", 12, "bold"), justify="center", insertbackground="white")
    servers_entry.pack(padx=(0,5), pady=(0,5), ipady=15, ipadx=50)
    
    # Simulation function
    def run_simulation():
        try:
            lam = float(lambda_entry.get())
            mu = float(mu_entry.get())
            servers = int(servers_entry.get())
            # ❌ Negative or zero checks
            if lam <= 0:
                messagebox.showerror("Error", "Arrival rate (λ) must be greater than 0.")
                return
            if mu <= 0:
                messagebox.showerror("Error", "Service rate (μ) must be greater than 0.")
                return
            if servers <= 0:
                messagebox.showerror("Error", "Number of Servers must be greater than 0.")
                return

            rho = lam / (servers * mu)  # utilization factor for M/M/s

            if rho >= 1:
                messagebox.showwarning(
                    "System Unstable",
                    "The queue will grow infinitely because λ ≥ s × μ.\n\n"
                    "Please enter values where λ < s × μ to have a stable system."
                )
                return

         # If stable, run simulation
            df = mms_simulation(lam, mu, servers)
            show_table(mms_frame, df, lam, mu, servers)

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

    # --- Simulate Button ---
    shadow_simulate = tk.Frame(container, bg="#333311")
    shadow_simulate.pack(pady=15)
    btn_simulate = tk.Button(shadow_simulate, text="Run Simulation", font=("Arial", 16, "bold"),
                             bg="#888844", fg="white", bd=0, relief="flat",
                             activebackground="#777733", activeforeground="white",
                             command=run_simulation)
    btn_simulate.pack(padx=(0,5), pady=(0,5), ipadx=90, ipady=15)
    # Hover effect for Simulate
    btn_simulate.bind("<Enter>", lambda e: btn_simulate.config(bg="#494920"))
    btn_simulate.bind("<Leave>", lambda e: btn_simulate.config(bg="#888844"))
    # --- Back Button ---
    shadow_back = tk.Frame(container, bg="#333311")
    shadow_back.pack(pady=5)
    btn_back = tk.Button(shadow_back, text="Back", font=("Arial", 16, "bold"),
                         bg="#888844", fg="white", bd=0, relief="flat",
                         activebackground="#777733", activeforeground="white",
                         command=lambda: raise_frame(main_frame))
    btn_back.pack(padx=(0,5), pady=(0,5), ipadx=107, ipady=15)
    # Hover effect for Back
    btn_back.bind("<Enter>", lambda e: btn_back.config(bg="#494920"))
    btn_back.bind("<Leave>", lambda e: btn_back.config(bg="#888844"))


# --- Frame Helper ---
def raise_frame(frame):
    frame.tkraise()

# --- Main Window ---
root = tk.Tk()
root.title("Queue Simulator")
root.state('zoomed')  # Full screen mode

main_frame = tk.Frame(root)
mm1_frame = tk.Frame(root)
mms_frame = tk.Frame(root)

for frame in (main_frame, mm1_frame, mms_frame):
    frame.place(relwidth=1, relheight=1)
    frame.configure(bg="#666633")

# Main page container
main_container = tk.Frame(main_frame, bg="#666633")
main_container.pack(expand=True)
# Label
tk.Label(main_container, text="QUEUEING MODEL SELECTION", font=("Arial", 22, "bold"), fg="white", bg="#666633").pack(pady=30)
#Button for M/M/1
shadow1 = tk.Frame(main_container, bg="#333311")  # deeper shadow
shadow1.pack(pady=25)
btn_mm1 = tk.Button(shadow1, text="M/M/1 MODEL", font=("Arial", 16, "bold"),
                    bg="#888844", fg="white", bd=0, relief="flat",
                    activebackground="#777733", activeforeground="white")
btn_mm1.pack(padx=(0,5), pady=(0,5), ipadx=90, ipady=15)  # increased width and height
btn_mm1.configure(command=mm1_input_page)
# Hover effect for btn_mm1
btn_mm1.bind("<Enter>", lambda e: btn_mm1.config(bg="#494920"))
btn_mm1.bind("<Leave>", lambda e: btn_mm1.config(bg="#888844"))

#Button for M/M/S
shadow2 = tk.Frame(main_container, bg="#333311")  # deeper shadow
shadow2.pack(pady=25)
btn_mms = tk.Button(shadow2, text="M/M/S MODEL", font=("Arial", 16, "bold"),
                    bg="#888844", fg="white", bd=0, relief="flat",
                    activebackground="#777733", activeforeground="white")
btn_mms.pack(padx=(0,5), pady=(0,5), ipadx=90, ipady=15)  # increased width and height
btn_mms.configure(command=mms_input_page)
# Hover effect for btn_mms
btn_mms.bind("<Enter>", lambda e: btn_mms.config(bg="#494920"))
btn_mms.bind("<Leave>", lambda e: btn_mms.config(bg="#888844"))

raise_frame(main_frame)
root.mainloop()
