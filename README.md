# **MM1 and MMS Queueing System Simulator**

## **Description**
A compact **Python simulator** for **MM1** and **MMS** queueing systems that both computes **analytical results** and visually demonstrates simulated behavior using a simple **Python UI**.

The program:

- Builds a **table of arrivals and per-customer events**.
- Calculates **performance metrics** such as **waiting time**, **turnaround time**, and **response time**.
- Displays a **Gantt chart** of server activity inside a Python UI window.

## **Features**
- **Detailed Arrival/Event Table:** Shows **arrival time**, **service start**, **service end**, **waiting time**, **response time**, and **turnaround time** for each customer.
- **Queueing Metrics Computation:**
  - **Waiting Time:** Time a customer waits before service starts.
  - **Turnaround Time:** Time from arrival to service completion.
  - **Response Time:** Time to first response/start of service.
  - **Summary Statistics:** Averages and totals of the above metrics.
- **Supports MM1 and MMS Configurations:** Single-server (**MM1**) and multi-server (**MMS**) setups.
- **Gantt Chart Visualization:** Displays which server handled which customer and when in a **Python UI screen**.
- **Single-File Project:** Easy to share, only requires **`index.py`**.

## **Setup and Usage**
### **Prerequisites**
- Python 3.x installed
- Required libraries: 
  ```bash
  pip install matplotlib numpy
