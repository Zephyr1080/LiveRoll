"""Student attendance system - run with:  python main.py"""
import config
from database import setup
from faces import register_from_webcam, register_from_folder
from attendance import run_attendance_session, recognize_photo
from reports import daily_report, monthly_report, show_table
from tests import run_demo, run_test_cases, run_liveness_tests, run_phone_tests


def ask(prompt):
    return input(prompt).strip()


def register_webcam():
    reg, name = ask("Registration no: "), ask("Student name: ")
    if reg and name:
        register_from_webcam(reg, name)


def register_folder():
    register_from_folder(ask("Folder with photos named RegNo_Name.jpg: "))


def attendance_webcam():
    n = ask("How many students (Enter = 1): ")
    run_attendance_session(int(n) if n.isdigit() and int(n) > 0 else 1)


def attendance_photo():
    recognize_photo(ask("Path to the photo: ").strip('"'))


def report_daily():
    show_table(daily_report())
    print("Saved: daily_report.csv")


def report_monthly():
    ym = ask("Month as YYYY-MM (Enter = this month): ") or None
    show_table(monthly_report(ym))
    print("Saved: monthly_report.csv")


def tests():
    run_liveness_tests()
    run_phone_tests()
    run_test_cases()


MENU = [
    ("Register a student (webcam)", register_webcam),
    ("Register students from a folder of photos", register_folder),
    ("Take attendance (webcam: blink + phone check)", attendance_webcam),
    ("Take attendance from a photo (no liveness check)", attendance_photo),
    ("Daily report", report_daily),
    ("Monthly report", report_monthly),
    ("Run tests", tests),
]


def main():
    setup()
    while True:
        print("\n=== Student Attendance System ===")
        for i, (label, _) in enumerate(MENU, 1):
            print(f" {i}. {label}")
        print(" 0. Exit")
        choice = ask("Choose: ")
        if choice == "0":
            break
        if choice.isdigit() and 1 <= int(choice) <= len(MENU):
            try:
                MENU[int(choice) - 1][1]()
            except Exception as e:               # keep the menu alive on errors
                print("Error:", e)
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
