from typing import List, Tuple

from pydantic.dataclasses import dataclass


@dataclass
class XArg:
    x_table: List[str]
    x_key: List[str]
    x_how: List[str]
    meta_id_escapes: List[str]
    meta_datetime_escapes: List[Tuple[str, str]]
    meta_time_escapes: List[Tuple[str, str]]

    def copy(self):
        return XArg(
            self.x_table, self.x_key, self.x_how,
            self.meta_id_escapes, self.meta_datetime_escapes, self.meta_time_escapes)


class XArgs:
    tables_6 = XArg(
        x_table=["BookLoan", "Book", "Library", "Student", "Enrollment", "Submission"],
        x_key=['book_id', "library_id", "student_id", "student_id", "student_id"],
        x_how=['inner' for _ in range(5)],
        meta_id_escapes=["assignment_id", "course_id"],
        meta_datetime_escapes=[("Submission", "submission_date")],
        meta_time_escapes=[],
    )
    tables_11 = XArg(
        x_table=["BookLoan",
                 "Book", "Library", "Student",
                 "Enrollment", "Submission", "Course",
                 # "Assignment"
                 "CourseTextbook", "Textbook",
                 "Schedule", "Professor"],
        x_key=[
            'book_id', "library_id", "student_id",
            "student_id", "student_id", "course_id",
            # "assignment_id",
            "course_id", "textbook_id",
            "course_id", "professor_id"],
        x_how=['inner' for _ in range(10)],
        meta_id_escapes=["assignment_id"],
        meta_datetime_escapes=[("Submission", "submission_date")],
        meta_time_escapes=[("Schedule", "time_slot")]
    )

    tables_14 = XArg(
        x_table=["BookLoan",
                 "Book", "Library", "Student",
                 "Enrollment", "Submission", "Course",
                 # "Assignment"
                 "CourseTextbook", "Textbook",
                 "Schedule", "Professor", 'ProjectMember',
                 'ResearchProject', 'ResearchGroup'],
        x_key=[
            'book_id', "library_id", "student_id",
            "student_id", "student_id", "course_id",
            # "assignment_id",
            "course_id", "textbook_id",
            "course_id", "professor_id", "professor_id",
            "project_id", "group_id"],
        x_how=['inner' for _ in range(13)],
        meta_id_escapes=["assignment_id"],
        meta_datetime_escapes=[("Submission", "submission_date")],
        meta_time_escapes=[("Schedule", "time_slot")]
    )
