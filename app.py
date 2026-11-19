"""Streamlit interface - run with:  streamlit run app.py

Runs on the same computer as the webcam. The webcam is read by Python (OpenCV),
and the live preview is drawn inside the page.
"""
import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st

import config
from database import setup, known_regs, run_sql, query_df, load_known_faces
from faces import register_student
from attendance import process_clip, recognize_array, now_parts
from camera import record_clip, grab_photo
from reports import daily_report, monthly_report

st.set_page_config(page_title="Student Attendance System", page_icon="🎓", layout="wide")
setup()


# ---------- helpers ----------
def preview_to(placeholder):
    """Return a callback that shows webcam frames in a placeholder (about 12 per second)."""
    last = [0.0]

    def show(img):
        if time.time() - last[0] >= 0.08:
            last[0] = time.time()
            placeholder.image(img, channels="BGR", width=480, output_format="JPEG")
    return show


def register_bytes(reg_no, name, data, suffix=".jpg"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        path = f.name
    try:
        return register_student(reg_no, name, path)
    finally:
        os.remove(path)


def say(message):
    (st.success if message.startswith("Registered") else st.error)(message)


def show_outcome(info):
    left, right = st.columns(2)
    left.image(info["image"], channels="BGR", width=480)
    live, phone = info["liveness"], info["phone"]
    if live:
        (right.success if live["live"] else right.error)(
            f"Blink check: {'PASS' if live['live'] else 'FAIL'} - "
            f"{live['blinks']} blink(s), {live['reason']}")
    if phone is not None:
        if phone["found"]:
            right.error(f"Phone check: FAIL - phone/screen detected ({phone['confidence']:.2f})")
        else:
            right.success("Phone check: PASS")
    for r in info["results"]:
        who = f"{r['name']} ({r['reg_no']})" if r["reg_no"] else r["name"]
        if r["status"].startswith("Present"):
            right.success(f"{who}: {r['status']}")
        elif r["status"].startswith("Duplicate"):
            right.info(f"{who}: {r['status']}")
        else:
            right.error(f"{who}: {r['status']}")


# ---------- sidebar ----------
st.sidebar.title("🎓 Attendance")
page = st.sidebar.radio("Menu", ["Dashboard", "Take attendance", "Register students", "Reports"])
with st.sidebar.expander("Settings"):
    config.LIVENESS_ENABLED = st.toggle("Blink check", config.LIVENESS_ENABLED)
    config.PHONE_CHECK_ENABLED = st.toggle("Phone check", config.PHONE_CHECK_ENABLED)
    config.TOLERANCE = st.slider("Face match tolerance (lower = stricter)", 0.3, 0.7,
                                 float(config.TOLERANCE), 0.01)


# ---------- pages ----------
if page == "Dashboard":
    st.title("Dashboard")
    today = now_parts()[0]
    present = daily_report(today, save_csv=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Registered students", len(known_regs))
    c2.metric("Present today", len(present))
    c3.metric("Not yet marked", max(len(known_regs) - len(present), 0))
    st.subheader(f"Today's attendance ({today})")
    st.dataframe(present, hide_index=True)
    students = query_df("SELECT reg_no, name FROM students")
    missing = students[~students["reg_no"].isin(
        present["Registration No"] if len(present) else [])]
    with st.expander(f"Not yet marked ({len(missing)})"):
        st.dataframe(missing.rename(columns={"reg_no": "Registration No", "name": "Student Name"}),
                     hide_index=True)

elif page == "Take attendance":
    st.title("Take attendance")
    live_tab, photo_tab = st.tabs(["Live webcam", "Classroom photo"])

    with live_tab:
        st.write("One student at a time. Click the button, look at the camera and "
                 f"**blink naturally** for {config.CLIP_SECONDS} seconds. Photos and phone "
                 "screens are rejected.")
        if st.button("Start live check", type="primary"):
            box = st.empty()
            try:
                frames = record_clip(config.CLIP_SECONDS, config.CLIP_FPS,
                                     on_frame=preview_to(box))
                box.empty()
                with st.spinner("Checking..."):
                    info = process_clip(frames)
                show_outcome(info)
            except RuntimeError as e:
                box.empty()
                st.error(str(e))

    with photo_tab:
        st.warning("Photo mode has no blink or phone check, so it can be fooled by a photo. "
                   "Use the live webcam for secure attendance.")
        up = st.file_uploader("Upload a classroom photo", type=["jpg", "jpeg", "png"])
        if up is not None:
            st.image(up, width=480)
            if st.button("Recognize and mark attendance"):
                bgr = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                annotated, results = recognize_array(bgr, mark=True)
                st.image(annotated, channels="BGR", width=700)
                if results:
                    st.dataframe([{"Name": r["name"], "Registration No": r["reg_no"],
                                   "Status": r["status"]} for r in results], hide_index=True)
                else:
                    st.info("No face detected in this image.")

elif page == "Register students":
    st.title("Register students")
    cam_tab, up_tab, list_tab = st.tabs(["Webcam", "Upload photos", "Registered students"])

    with cam_tab:
        reg = st.text_input("Registration number", key="cam_reg")
        name = st.text_input("Student name", key="cam_name")
        st.caption("A 3-second countdown runs, then one photo is taken. Face the camera.")
        if st.button("Capture and register", type="primary"):
            if not reg.strip() or not name.strip():
                st.error("Enter both the registration number and the name.")
            else:
                box = st.empty()
                try:
                    frame = grab_photo(3, on_frame=preview_to(box))
                    box.empty()
                    st.image(frame, channels="BGR", width=320)
                    ok, buf = cv2.imencode(".jpg", frame)
                    say(register_bytes(reg.strip(), name.strip(), buf.tobytes()))
                except RuntimeError as e:
                    box.empty()
                    st.error(str(e))

    with up_tab:
        st.subheader("One student")
        reg2 = st.text_input("Registration number", key="up_reg")
        name2 = st.text_input("Student name", key="up_name")
        photo = st.file_uploader("Photo (exactly one face)", type=["jpg", "jpeg", "png"], key="one")
        if st.button("Register this photo"):
            if not (reg2.strip() and name2.strip() and photo):
                st.error("Enter the registration number, the name and a photo.")
            else:
                say(register_bytes(reg2.strip(), name2.strip(), photo.getvalue(),
                                   os.path.splitext(photo.name)[1] or ".jpg"))
        st.subheader("Many students at once")
        st.caption("Name each file like  AP26122040014_Phani_Trinadh.jpg  (RegNo_Name.jpg)")
        many = st.file_uploader("Photos", type=["jpg", "jpeg", "png"],
                                accept_multiple_files=True, key="many")
        if st.button("Register all photos") and many:
            for f in many:
                stem = os.path.splitext(f.name)[0]
                r, _, n = stem.partition("_")
                say(register_bytes(r, n.replace("_", " "), f.getvalue(),
                                   os.path.splitext(f.name)[1]))

    with list_tab:
        df = query_df("SELECT reg_no AS 'Registration No', name AS 'Student Name' "
                      "FROM students ORDER BY reg_no")
        st.dataframe(df, hide_index=True)
        if len(df):
            pick = st.selectbox("Remove a student", df["Registration No"])
            if st.button("Delete selected student"):
                run_sql("DELETE FROM students WHERE reg_no=?", (pick,))
                load_known_faces()
                st.rerun()

elif page == "Reports":
    st.title("Reports")
    d_tab, m_tab = st.tabs(["Daily", "Monthly"])

    with d_tab:
        day = st.date_input("Date")
        df = daily_report(day.strftime("%Y-%m-%d"), save_csv=False)
        st.dataframe(df, hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False),
                           file_name=f"daily_{day}.csv", mime="text/csv")

    with m_tab:
        month_day = st.date_input("Pick any day in the month", key="month")
        ym = month_day.strftime("%Y-%m")
        mdf = monthly_report(ym, save_csv=False)
        st.dataframe(mdf, hide_index=True)
        if len(mdf):
            st.bar_chart(mdf.set_index("Student Name")["Attendance %"])
        st.download_button("Download CSV", mdf.to_csv(index=False),
                           file_name=f"monthly_{ym}.csv", mime="text/csv")
