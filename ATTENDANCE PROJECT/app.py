import streamlit as st
import pandas as pd
from datetime import datetime
import time

ts = time.time()
dt = datetime.fromtimestamp(ts)
date = dt.strftime("%d-%m-%y")
timestamp = dt.strftime("%H:%M:%S")
from streamlit_autorefresh import st_autorefresh

count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

if count == 0:
    st.write("Count is zero")
elif count % 3 == 0 and count % 5 == 0:
    st.write("FizzBuzz")
elif count % 3 == 0:
    st.write("Fizz")
elif count % 5 == 0:
    st.write("Buzz")
else:
    st.write(f"Count: {count}")
try:
    df = pd.read_csv("Attendance/Attendance_" + date + ".csv")
    st.dataframe(df.style.highlight_max(axis=0))
except FileNotFoundError:
    st.error("File not found: Attendance_" + date + ".csv")
