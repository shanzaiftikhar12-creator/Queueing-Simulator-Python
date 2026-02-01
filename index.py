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
    service_times = [max(1, math.ceil(-mu_val * math.log(np.random.rand()))) for _ in range(len(arrival_times))]
    # NEW: Generate Priorities 1-3 (1=High, 3=Low)
    priorities = [np.random.randint(1, 4) for _ in range(len(arrival_times))]
    # NEW STEP 1: Track remaining time to allow for interrupts
    remaining_times = list(service_times)
    service_starts = [0] * len(arrival_times)
    service_ends = [0] * len(arrival_times)
    service_chunks = [] # To store (Customer, Start, End) for the Gantt chart

    # NEW STEP 2: Preemptive Logic for M/M/1
    service_starts = [-1] * len(arrival_times)
    service_ends = [0] * len(arrival_times)
    served_mask = [False] * len(arrival_times)
    
    current_time = 0
    active_customer = None # Who is on the server right now?
    
    # We create a timeline of all arrival times to check for interrupts
    all_events = sorted(list(set(arrival_times)))
    

    while not all(served_mask):
        # 1. Identify who has arrived and is waiting
        waiting_pool = [i for i in range(len(arrival_times)) 
                        if not served_mask[i] and arrival_times[i] <= current_time]
        
        if not waiting_pool:
            current_time = min([arrival_times[i] for i in range(len(arrival_times)) if not served_mask[i]])
            continue

        # 2. Selection: Pick the highest superiority (1 is best)
        if use_priority_var.get():
            chosen_idx = min(waiting_pool, key=lambda x: priorities[x])
        else:
            chosen_idx = min(waiting_pool)

        # Record First Touch (for Response Time)
        if service_starts[chosen_idx] == -1:
            service_starts[chosen_idx] = current_time

        # 3. Hijack Logic: Check if a superior priority is coming
        interruptors = [i for i in range(len(arrival_times)) 
                        if arrival_times[i] > current_time and priorities[i] < priorities[chosen_idx]]
        
        if interruptors and use_priority_var.get():
            next_interrupt_time = min([arrival_times[i] for i in interruptors])
            time_to_event = next_interrupt_time - current_time
        else:
            time_to_event = remaining_times[chosen_idx]
        
        # 4. Process the work chunk
        work_duration = min(remaining_times[chosen_idx], time_to_event)
        
        start_chunk = current_time
        current_time += work_duration
        remaining_times[chosen_idx] -= work_duration
        
        # RECORD CHUNK: This ensures the Gantt Chart shows the split
        service_chunks.append((f"C{chosen_idx + 1}", start_chunk, current_time))
        
        # 5. Finalize if done
        if remaining_times[chosen_idx] == 0:
            service_ends[chosen_idx] = current_time
            served_mask[chosen_idx] = True

    # --- Performance Metrics for Preemptive Logic ---
    # These must be calculated after the loop finishes so the final values are available
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
        "Priority": priorities,
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

    return df, service_chunks

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
    # Fixed the zero service time issue
    service_times = [max(1, math.ceil(-mu_val * math.log(np.random.rand()))) for _ in range(len(arrival_times))]
    # NEW: Generate Priorities 1-3
    priorities = [np.random.randint(1, 4) for _ in range(len(arrival_times))]
    # NEW: Track remaining time and chunks for M/M/S preemption
    remaining_times = list(service_times)
    service_chunks = [] # To store (Customer, Start, End, ServerID)
    last_chunk_start = [0] * len(arrival_times) # Tracks when the current server session started

    # --- Initialize variables for Preemptive Logic ---
    service_start = [-1] * len(arrival_times)
    service_end = [0] * len(arrival_times)
    server_assigned = [""] * len(arrival_times)
    served_mask = [False] * len(arrival_times)
    
    # Track what each server is doing: {server_idx: customer_idx or None}
    server_occupancy = [None] * servers
    current_time = 0

    while not all(served_mask):
        # 1. Identify who is waiting in the system
        waiting_pool = [i for i in range(len(arrival_times)) 
                        if not served_mask[i] and arrival_times[i] <= current_time 
                        and i not in server_occupancy]
        
        # 2. Assign idle servers to waiting customers
        for s_idx in range(servers):
            if server_occupancy[s_idx] is None and waiting_pool:
                chosen = min(waiting_pool, key=lambda x: priorities[x]) if use_priority_var.get() else min(waiting_pool)
                server_occupancy[s_idx] = chosen
                waiting_pool.remove(chosen)
                if service_start[chosen] == -1: 
                    service_start[chosen] = current_time
                server_assigned[chosen] = f"S{s_idx + 1}"
                last_chunk_start[chosen] = current_time # Start the clock for this chunk

        # 3. HIJACK LOGIC: Can a high-priority newcomer kick someone off?
        if use_priority_var.get() and waiting_pool:
            for s_idx in range(servers):
                curr_cust = server_occupancy[s_idx]
                if curr_cust is not None:
                    best_in_pool = min(waiting_pool, key=lambda x: priorities[x])
                    if priorities[best_in_pool] < priorities[curr_cust]:
                        # Record the partial work done by the person being kicked off
                        duration = current_time - last_chunk_start[curr_cust]
                        if duration > 0:
                            service_chunks.append((f"C{curr_cust + 1}", last_chunk_start[curr_cust], current_time, f"S{s_idx+1}"))
                        
                        # Perform the swap
                        waiting_pool.append(curr_cust)
                        server_occupancy[s_idx] = best_in_pool
                        waiting_pool.remove(best_in_pool)
                        if service_start[best_in_pool] == -1: 
                            service_start[best_in_pool] = current_time
                        server_assigned[best_in_pool] = f"S{s_idx + 1}"
                        last_chunk_start[best_in_pool] = current_time

        # 4. CALCULATE THE JUMP: Find the next arrival or completion
        possible_events = []
        next_arrivals = [arrival_times[i] for i in range(len(arrival_times)) if arrival_times[i] > current_time]
        if next_arrivals: possible_events.append(min(next_arrivals))
        
        for s_idx in range(servers):
            c_idx = server_occupancy[s_idx]
            if c_idx is not None:
                possible_events.append(current_time + remaining_times[c_idx])
        
        if not possible_events: break 
        
        next_event_time = min(possible_events)
        time_jump = next_event_time - current_time

        # 5. Execute work for this duration
        for s_idx in range(servers):
            c_idx = server_occupancy[s_idx]
            if c_idx is not None:
                remaining_times[c_idx] -= time_jump
                # If they finish exactly now, close the chunk and the server slot
                if remaining_times[c_idx] <= 0:
                    service_chunks.append((f"C{c_idx + 1}", last_chunk_start[c_idx], next_event_time, f"S{s_idx+1}"))
                    service_end[c_idx] = next_event_time
                    served_mask[c_idx] = True
                    server_occupancy[s_idx] = None
        
        current_time = next_event_time

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
        "Priority": priorities,
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

    return df, service_chunks

# Simulation Table Container
def show_table(parent_frame, df, lambda_val, mu_val, servers, chunks=None):
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

    # --- TABLE (centered) ---
    table_wrapper = tk.Frame(scrollable_frame, bg="#666633")
    table_wrapper.pack(pady=(10, 50), fill="x")
    table_frame = tk.Frame(table_wrapper, bg="#666633")
    table_frame.pack(anchor="center")

# Filter columns: If Priority is disabled, remove the Priority column
    cols = list(df.columns)
    if not use_priority_var.get() and "Priority" in cols:
        cols.remove("Priority")
    
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

    # Table data rows (Using only the filtered columns)
    for i, row in enumerate(df.to_dict('records'), start=1):
        for j, col_name in enumerate(cols):
            val = row[col_name]
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
        # We must pass 'chunks' and the 'servers' count
        draw_mms_gantt(chunks, scrollable_frame, servers) 
    else:
        draw_mm1_gantt(chunks, scrollable_frame)

    # Spacer at bottom
    tk.Label(scrollable_frame, text="", bg="#666633").pack(pady=40)

    # Force the scrollable area to recalculate after all charts are added
    scrollable_frame.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))

def draw_mm1_gantt(chunks, scrollable_frame):
    if not chunks:
        return

    # 1. Build the full timeline including idle gaps
    timeline = []
    current_time = chunks[0][1]
    for label, start, end in chunks:
        if start > current_time:
            timeline.append(("Idle", current_time, start))
        timeline.append((label, start, end))
        current_time = end

    # 2. Setup colors
    customer_colors = ["#A56A64", "#7D719B", "#818F6D", "#D18685", "#6379A1", "#A36E6E", "#AF7EA6"]
    color_cycle = itertools.cycle(customer_colors)
    cust_color_map = {}

    # 3. Parameters for Wrapping
    boxes_per_row = 10
    box_width = 1.5
    row_height_gap = 1.5 # Space between lines
    total_boxes = len(timeline)
    num_rows = math.ceil(total_boxes / boxes_per_row)

    # Adjust figure size based on the number of rows
    fig, ax = plt.subplots(figsize=(12, 2 * num_rows), facecolor="#666633")
    ax.set_facecolor("#666633")

    # 4. Draw boxes in rows
    for i, (label, start, end) in enumerate(timeline):
        row_idx = i // boxes_per_row
        col_idx = i % boxes_per_row
        
        # Calculate coordinates
        x = col_idx * box_width
        y = -row_idx * row_height_gap # Move down for each new row
        
        color = "#BBBBA0" if "Idle" in label else cust_color_map.setdefault(label, next(color_cycle))
        
        # Draw box
        rect = plt.Rectangle((x, y), box_width, 1, facecolor=color, edgecolor="white", lw=1.5)
        ax.add_patch(rect)
        
        # Add labels inside box
        ax.text(x + box_width/2, y + 0.5, label, color="white", fontweight="bold", 
                ha="center", va="center", fontsize=9)
        
        # Add start/end times below the box
        ax.text(x, y - 0.3, str(int(start)), color="white", fontsize=8, ha="center")
        ax.text(x + box_width, y - 0.3, str(int(end)), color="white", fontsize=8, ha="center")

    # 5. Aesthetics
    ax.set_xlim(-0.5, boxes_per_row * box_width + 0.5)
    ax.set_ylim(-num_rows * row_height_gap + 1, 2)
    ax.axis('off')
    ax.set_title("Server Utilization Timeline", color="white", fontweight="bold", pad=0)

    canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=0, fill="x")

def draw_mms_gantt(chunks, scrollable_frame, num_servers):
    if not chunks:
        return

    # 1. Setup colors
    customer_colors = ["#A56A64", "#7D719B", "#818F6D", "#D18685", "#6379A1", "#A36E6E", "#AF7EA6"]
    color_cycle = itertools.cycle(customer_colors)
    cust_color_map = {}

    # 2. Parameters for Layout
    boxes_per_row = 10
    box_width = 1.5
    row_height_gap = 1.5 

    # 3. Draw a separate chart for each server
    for s_idx in range(num_servers):
        srv_name = f"S{s_idx + 1}"
        srv_chunks = [c for c in chunks if c[3] == srv_name]
        
        if not srv_chunks:
            continue

        # Build timeline for this server including idle gaps
        timeline = []
        current_time = srv_chunks[0][1]
        for label, start, end, _ in srv_chunks:
            if start > current_time:
                timeline.append(("Idle", current_time, start))
            timeline.append((label, start, end))
            current_time = end

        num_rows = math.ceil(len(timeline) / boxes_per_row)
        fig, ax = plt.subplots(figsize=(12, 2 * num_rows), facecolor="#666633")
        ax.set_facecolor("#666633")

        # 4. Draw boxes in rows for the current server
        for i, (label, start, end) in enumerate(timeline):
            row_idx = i // boxes_per_row
            col_idx = i % boxes_per_row
            x = col_idx * box_width
            y = -row_idx * row_height_gap
            
            color = "#BBBBA0" if "Idle" in label else cust_color_map.setdefault(label, next(color_cycle))
            
            ax.add_patch(plt.Rectangle((x, y), box_width, 1, facecolor=color, edgecolor="white", lw=1.5))
            ax.text(x + box_width/2, y + 0.5, label, color="white", fontweight="bold", ha="center", va="center", fontsize=9)
            ax.text(x, y - 0.3, str(int(start)), color="white", fontsize=8, ha="center")
            ax.text(x + box_width, y - 0.3, str(int(end)), color="white", fontsize=8, ha="center")

        ax.set_xlim(-0.5, boxes_per_row * box_width + 0.5)
        ax.set_ylim(-num_rows * row_height_gap + 1, 2)
        ax.axis('off')

        # --- FIX: ADD NATIVE TKINTER LABEL BEFORE THE CHART ---
        tk.Label(
            scrollable_frame,
            text=f"Server {s_idx + 1} Utilization Timeline",
            font=("Arial", 16, "bold"),
            fg="white",
            bg="#666633"
        ).pack(pady=(5, 0))

        canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=(0, 0), fill="x")

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
    tk.Label(container, text="M/M/1 Parameters", font=("Arial", 22, "bold"),
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
            df, chunks = mm1_simulation(lam, mu)
            show_table(mm1_frame, df, lam, mu, servers=1, chunks=chunks)

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

    # --- Simulate Button ---
    # check button added for mm1
    tk.Checkbutton(container, text="Enable Priority Scheduling", 
                   variable=use_priority_var, font=("Arial", 12, "bold"),
                   bg="#666633", fg="white", selectcolor="#333311",
                   activebackground="#666633", activeforeground="white").pack(pady=10)
    shadow_simulate = tk.Frame(container, bg="#333311")
    shadow_simulate.pack(pady=15)
    btn_simulate = tk.Button(shadow_simulate, text="Simulate", font=("Arial", 16, "bold"),
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
    tk.Label(container, text="M/M/S Parameters", font=("Arial", 22, "bold"),
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
            df, chunks = mms_simulation(lam, mu, servers)
            show_table(mms_frame, df, lam, mu, servers, chunks=chunks)

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

    # --- Simulate Button ---
    # check button added for mms
    tk.Checkbutton(container, text="Enable Priority Scheduling", 
                   variable=use_priority_var, font=("Arial", 12, "bold"),
                   bg="#666633", fg="white", selectcolor="#333311",
                   activebackground="#666633", activeforeground="white").pack(pady=10)
    shadow_simulate = tk.Frame(container, bg="#333311")
    shadow_simulate.pack(pady=15)
    btn_simulate = tk.Button(shadow_simulate, text="Simulate", font=("Arial", 16, "bold"),
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
root.attributes('-fullscreen', True)  # Full screen on Linux, Windows, macOS
#root.state('zoomed')  # Full screen mode
# check box added
use_priority_var = tk.BooleanVar(value=False)

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
tk.Label(main_container, text="CHOOSE SIMULATION MODEL", font=("Arial", 22, "bold"), fg="white", bg="#666633").pack(pady=30)
#Button for M/M/1
shadow1 = tk.Frame(main_container, bg="#333311")  # deeper shadow
shadow1.pack(pady=25)
btn_mm1 = tk.Button(shadow1, text="M/M/1", font=("Arial", 16, "bold"),
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
btn_mms = tk.Button(shadow2, text="M/M/S", font=("Arial", 16, "bold"),
                    bg="#888844", fg="white", bd=0, relief="flat",
                    activebackground="#777733", activeforeground="white")
btn_mms.pack(padx=(0,5), pady=(0,5), ipadx=90, ipady=15)  # increased width and height
btn_mms.configure(command=mms_input_page)
# Hover effect for btn_mms
btn_mms.bind("<Enter>", lambda e: btn_mms.config(bg="#494920"))
btn_mms.bind("<Leave>", lambda e: btn_mms.config(bg="#888844"))

# --- ADD THIS: Quit Button ---
shadow_quit = tk.Frame(main_container, bg="#333311")
shadow_quit.pack(pady=25)
btn_quit = tk.Button(
    shadow_quit, 
    text="QUIT", 
    font=("Arial", 16, "bold"),
    bg="#802020", # Dark red for a clear exit action
    fg="white", 
    bd=0, 
    relief="flat",
    activebackground="#601010", 
    activeforeground="white",
    command=root.destroy # This command exits the program
)
btn_quit.pack(padx=(0,5), pady=(0,5), ipadx=95, ipady=15)

# Hover effect for Quit Button
btn_quit.bind("<Enter>", lambda e: btn_quit.config(bg="#b03030"))
btn_quit.bind("<Leave>", lambda e: btn_quit.config(bg="#802020"))

raise_frame(main_frame)
root.mainloop()