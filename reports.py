"""Daily and monthly attendance reports."""
from database import query_df
from attendance import now_parts


def daily_report(date=None, save_csv=True):
    date = date or now_parts()[0]
    df = query_df("SELECT name, reg_no, date, time, status, liveness FROM attendance "
                  "WHERE date=? ORDER BY time", (date,))
    df.columns = ["Student Name", "Registration No", "Date", "Time", "Status", "Liveness"]
    if save_csv:
        df.to_csv("daily_report.csv", index=False)
    return df


def monthly_report(year_month=None, save_csv=True):
    """year_month looks like '2026-11'."""
    year_month = year_month or now_parts()[0][:7]
    records = query_df("SELECT reg_no, date FROM attendance WHERE date LIKE ?", (year_month + "%",))
    students = query_df("SELECT reg_no, name FROM students")
    held = records["date"].nunique()              # days on which attendance was taken
    present = records.groupby("reg_no").size().rename("Days Present").reset_index()
    report = students.merge(present, on="reg_no", how="left").fillna({"Days Present": 0})
    report["Days Present"] = report["Days Present"].astype(int)
    report["Classes Held"] = held
    report["Attendance %"] = (report["Days Present"] / held * 100).round(1) if held else 0.0
    report.columns = ["Registration No", "Student Name", "Days Present", "Classes Held", "Attendance %"]
    if save_csv:
        report.to_csv("monthly_report.csv", index=False)
    return report


def show_table(df):
    """Print a DataFrame as a plain text table."""
    print(df.to_string(index=False) if len(df) else "(no records)")
