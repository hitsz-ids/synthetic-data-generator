from typing import NamedTuple, List, Tuple

x_args_type = NamedTuple('x_args_type', [
    ('x_table', List[str]),
    ('x_key', List[str]),
    ('x_how', List[str]),
    ('meta_id_escapes', List[str]),
    ('meta_datetime_escapes', List[Tuple[str, str]]),
    ('meta_time_escapes', List[Tuple[str, str]])
])


class XArgs:
    tables_6 = x_args_type(
        x_table=["BookLoan", "Book", "Library", "Student", "Enrollment", "Submission"],
        x_key=['book_id', "library_id", "student_id", "student_id", "student_id"],
        x_how=['inner' for _ in range(5)],
        meta_id_escapes=["assignment_id", "course_id"],
        meta_datetime_escapes=[("Submission", "submission_date")],
        meta_time_escapes=[],
    )
    tables_11 = x_args_type(
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

    tables_14 = x_args_type(
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