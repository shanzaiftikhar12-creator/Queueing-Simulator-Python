A compact Python simulator for MM1 and MMS queueing systems that both computes analytical results and visually demonstrates simulated behavior using a simple Python UI.

The program:

builds a table of arrivals and per-customer events,

calculates performance metrics such as waiting time, turnaround time, and response time,

and displays a Gantt chart of server activity inside a Python UI window.

Features

Generates a detailed arrival / event table (arrival time, service start, service end, waiting time, response time, turnaround).

Computes and prints queueing metrics:

Waiting time (time customer waits before service starts),

Turnaround time (time from arrival to service completion),

Response time (time to first response/start of service),

Summary statistics (averages, totals).

Supports both MM1 (single-server) and MMS (multi-server) configurations.

Produces a Gantt chart showing which server handled which customer and when — displayed in a Python UI screen.

Single-file project for easy sharing: index.py.
